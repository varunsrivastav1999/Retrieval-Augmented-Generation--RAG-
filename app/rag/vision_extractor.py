import os
import base64
import json
import requests
import io
import pdfplumber
from typing import Dict, Any

from app.rag.model_loader import get_ollama_generate_url

# We allow overriding the vision model via env var, default to llama3.2-vision
VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "llama3.2-vision")

VISION_SYSTEM_PROMPT = """You are an elite AI vision architect and enterprise data extractor. Your task is to look at the provided image of a document (like a complex catalogue table or specification sheet) and extract its tabular contents into perfectly structured Markdown or JSON.

CRITICAL RULES:
1. Extract ONLY information explicitly visible in the image.
2. Maintain perfect row/column alignment. If there are nested headers (like 'Dimensions' spanning 'H', 'W', 'D'), ensure the data maps to the correct sub-column.
3. Resolve clustered or merged cells into distinct row items.
4. Output your result strictly as valid JSON or clean Markdown tables so it can be parsed.

Use this JSON schema if you detect a complex table:
{
  "Document_Type": "Hardware_Catalog_Table",
  "Structured_Payload": [
    {
      "Identifier": "Part Number / Model Name",
      "Attributes": {
         "Column_Name_1": "Value",
         "Column_Name_2": "Value"
      }
    }
  ]
}
If the table is simple, a Markdown grid is acceptable. Return ONLY the JSON or Markdown, no conversational text.
"""

def extract_table_with_vision(pdf_path: str, page_index: int) -> Dict[str, Any]:
    """
    Extracts structured table data from a specific PDF page using a local Vision LLM via Ollama.
    """
    if not os.path.exists(pdf_path):
        return {}

    # 1. Extract image of the page using pdfplumber
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_index >= len(pdf.pages):
                return {}
            page = pdf.pages[page_index]
            # Convert page to a PIL Image (high resolution)
            im = page.to_image(resolution=300)
            pil_image = im.original
            
            # Save to bytes
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"[Vision] Error converting PDF page to image: {e}")
        return {}

    # 2. Send image to Ollama Vision Model
    payload = {
        "model": VISION_MODEL,
        "system": VISION_SYSTEM_PROMPT,
        "prompt": "Extract the table in this image into structured JSON format.",
        "images": [img_str],
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": -1
        },
        "format": "json"
    }

    last_error = ""
    for attempt in range(2):
        try:
            response = requests.post(get_ollama_generate_url(), json=payload, timeout=300)
            if response.status_code == 200:
                result_text = response.json().get("response", "").strip()
                
                # Cleanup potential Markdown wrappings around JSON
                if result_text.startswith("```json"):
                    result_text = result_text[7:]
                if result_text.startswith("```"):
                    result_text = result_text[3:]
                if result_text.endswith("```"):
                    result_text = result_text[:-3]
                    
                result_text = result_text.strip()
                
                if not result_text:
                    last_error = "Empty response from vision model"
                    continue
                
                try:
                    parsed_data = json.loads(result_text)
                    if isinstance(parsed_data, dict):
                        return parsed_data
                except json.JSONDecodeError as je:
                    last_error = f"JSON decode error: {je}"
            else:
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                print(f"[Vision] Ollama API Error (attempt {attempt+1}): {last_error}")
        except Exception as e:
            last_error = str(e)
            print(f"[Vision] Request failed (attempt {attempt+1}): {e}")
            
    print(f"[Vision] Failed to extract via Vision Model: {last_error}")
    return {}
