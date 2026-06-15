import json
import re
import requests
from typing import List, Dict, Any

from app.rag.model_loader import get_ollama_generate_url, OLLAMA_MODEL

# The system prompt provided by the user for electrical catalog extraction
EXTRACTION_SYSTEM_PROMPT = """You are an expert technical data extractor specializing in electrical engineering, loadcentres, and industrial control specifications. Your task is to process raw, poorly formatted, flattened OCR text from product catalogs and convert it into a perfectly structured, flat JSON array of objects.

Follow these strict rules to resolve misaligned and clustered data:

1. RESOLVE CLUSTERS: If you see clustered model numbers and amperages (e.g., "SEQ24100SMD SEQ24100SMK" followed by "100 125"), you must logically separate them into individual JSON objects. Each catalog number must have its own distinct object.
2. FIX DIMENSIONS: Reformat poorly parsed fractions into standard readable formats. Convert strings like "167/8" to "16 7/8", "131/32" to "13 1/32", and maintain the millimeter conversions in parentheses if present.
3. RETAIN HIERARCHY AS METADATA: Identify the overarching category for the section (e.g., "EQL Loadcentres with main lugs only 1 phase 3 wire 240 V AC max") and apply this exact string as a "Category_Context" field to EVERY object extracted from that section.
4. CAPTURE MODIFICATIONS: If rules like "Factory Modifications" (e.g., "White door and trim add suffix...W") appear, apply this context to a "Notes" or "Modifications" field for the relevant models.
5. STRICT JSON: Output ONLY valid JSON. Do not include markdown formatting, conversational text, or explanations.

Use the following JSON schema for each item:
{
  "Category_Context": "string",
  "Number_of_Circuits": "string",
  "Catalogue_Number": "string",
  "Skid_Qty": "integer or null",
  "Main_Amps": "integer or null",
  "Dimensions_Inches": {
    "Height": "string",
    "Width": "string",
    "Depth": "string"
  },
  "Dimensions_mm": {
    "Height": "integer or null",
    "Width": "integer or null",
    "Depth": "integer or null"
  },
  "Lug_Data": "string",
  "Mounting_Trim": "string",
  "Door_Kit_Catalogue_Number": "string",
  "Notes_and_Modifications": "string"
}
"""

def looks_like_catalog_page(text: str) -> bool:
    """
    Heuristic to determine if a page contains electrical catalog/table data.
    Looks for high density of electrical specifications and catalog terminology.
    """
    text_lower = text.lower()
    keywords = ["amps", "dimensions", "loadcentre", "catalogue", "catalog", "breaker", "suffix", "circuits", "skid"]
    
    # Check if at least 3 keywords are present
    match_count = sum(1 for kw in keywords if kw in text_lower)
    
    # Also check for product codes (e.g., sequences of caps and numbers)
    code_matches = len(re.findall(r'\b[A-Z]{2,4}[0-9]{3,5}[A-Z]*\b', text))
    
    return match_count >= 3 or code_matches >= 5

def extract_catalog_data_from_page(page_text: str) -> List[Dict[str, Any]]:
    """
    Sends the raw page text to the LLM to extract perfectly structured JSON arrays
    representing the electrical components.
    """
    if not page_text or not page_text.strip():
        return []
        
    prompt = f"Raw OCR Text:\n{page_text}\n\nReturn the JSON array of objects now."
    
    payload = {
        "model": OLLAMA_MODEL,
        "system": EXTRACTION_SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 4096,
            "num_gpu": 99,
        },
        "format": "json" # Force JSON mode if supported by the model
    }
    
    try:
        response = requests.post(get_ollama_generate_url(), json=payload, timeout=120)
        if response.status_code == 200:
            result_text = response.json().get("response", "").strip()
            
            # Clean up markdown formatting if the model still includes it
            if result_text.startswith("```json"):
                result_text = result_text[7:]
            if result_text.startswith("```"):
                result_text = result_text[3:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
                
            result_text = result_text.strip()
            
            try:
                parsed_data = json.loads(result_text)
                if isinstance(parsed_data, list):
                    return parsed_data
                elif isinstance(parsed_data, dict):
                    # Sometimes the model wraps it in an object like {"components": [...]}
                    for key, val in parsed_data.items():
                        if isinstance(val, list):
                            return val
                    return [parsed_data] # If it's just a single object
                return []
            except json.JSONDecodeError as je:
                print(f"[Extraction] JSON Decode Error: {je}. Raw output: {result_text[:200]}...")
                return []
        else:
            print(f"[Extraction] Ollama API Error: {response.text}")
            return []
    except Exception as e:
        print(f"[Extraction] Error calling LLM: {e}")
        return []

def format_json_object_for_embedding(obj: Dict[str, Any]) -> str:
    """
    Converts a structured JSON object into a highly readable, key-value string format.
    This format yields much higher semantic similarity scores during vector retrieval
    compared to raw JSON brackets.
    """
    lines = []
    
    # Prioritize key identifiers
    if obj.get("Category_Context"):
        lines.append(f"Category Context: {obj.get('Category_Context')}")
    if obj.get("Catalogue_Number"):
        lines.append(f"Catalogue Number (Model): {obj.get('Catalogue_Number')}")
        
    for key, value in obj.items():
        if key in ["Category_Context", "Catalogue_Number"]:
            continue
            
        if value is None or value == "":
            continue
            
        formatted_key = key.replace("_", " ")
        
        if isinstance(value, dict):
            # E.g., Dimensions_Inches -> Height: 10, Width: 5
            sub_items = [f"{k}: {v}" for k, v in value.items() if v is not None and v != ""]
            if sub_items:
                lines.append(f"{formatted_key}: " + ", ".join(sub_items))
        else:
            lines.append(f"{formatted_key}: {value}")
            
    return "\n".join(lines)
