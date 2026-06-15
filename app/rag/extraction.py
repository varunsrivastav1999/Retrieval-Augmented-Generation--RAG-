import json
import re
import requests
from typing import List, Dict, Any

from app.rag.model_loader import get_ollama_generate_url, OLLAMA_MODEL

# The system prompt provided by the user for electrical catalog extraction
EXTRACTION_SYSTEM_PROMPT = """You are an elite AI data architect and technical data extractor. Your task is to process raw, poorly formatted, flattened OCR text from various enterprise documents (electrical product catalogs, SCADA/robotics manuals, complex financial reports, and HR postings) and convert it into a perfectly structured JSON object for a RAG vector database.

Follow these strict rules based on the document type you detect:

1. IF CATALOG (Hardware/Specs):
   - RESOLVE CLUSTERS: If you see clustered model numbers and amperages (e.g., "SEQ24100SMD SEQ24100SMK" followed by "100 125"), you must logically separate them into individual JSON objects. Each catalogue number must have its own distinct object.
   - FIX DIMENSIONS: Reformat poorly parsed fractions (e.g., "167/8" to "16 7/8", "131/32" to "13 1/32").
   - RETAIN HIERARCHY: Identify the overarching category for the section (e.g., "EQL Loadcentres 1 phase 3 wire") and apply this exact string as a "Category_Context" field to EVERY object extracted from that section.

2. IF SCADA/ROBOTICS MANUAL:
   - Extract step-by-step procedures into ordered arrays. 
   - If UI elements (like Red/Green status indicators, Faults, or specific dashboard screens) are mentioned, tag them in a `UI_Elements` array for easy fault-resolution retrieval.

3. IF FINANCIAL/CORPORATE REPORT:
   - Scan for multi-hop reasoning requirements, numerical discrepancies, and footnotes. 
   - If a metric (like revenue or net profit) is reported differently across pages, extract both values and include the explanatory footnote in a `Discrepancy_Context` field.

4. IF HR/JOB POSTING:
   - Extract the role, company, salary/CTC, location, and exact experience requirements into simple key-value pairs.

STRICT JSON OUTPUT:
Return ONLY valid JSON. Do not include markdown formatting, conversational text, or explanations. Use the following schema:

{
  "Document_Type": "Catalog | SCADA_Manual | Financial_Report | HR_Document",
  "Primary_Entities": ["List of main subjects, models, or companies"],
  "Category_Context": "string (Overarching category or section name)",
  "Extracted_Data": [
    {
      "Item_Name_or_Catalogue_Number": "string",
      "Attributes": {
        "Main_Amps": "integer or null",
        "Dimensions_Inches": "string or null",
        "Role_or_Procedure_Step": "string or null"
      },
      "UI_Elements": ["string"],
      "Discrepancy_Context": "string or null",
      "Notes_and_Modifications": "string or null"
    }
  ]
}
"""

def looks_like_extractable_page(text: str) -> bool:
    """
    Heuristic to determine if a page contains dense extractable data 
    (catalogs, SCADA manuals, financial reports, HR postings).
    """
    text_lower = text.lower()
    keywords = [
        # Catalog
        "amps", "dimensions", "loadcentre", "catalogue", "catalog", "breaker", "suffix", "circuits", "skid",
        # SCADA / Manual
        "fault", "indicator", "status", "dashboard", "procedure", "step", "warning",
        # Finance
        "revenue", "net profit", "fiscal", "quarter", "ebitda", "footnote", "discrepancy",
        # HR
        "salary", "ctc", "experience", "requirements", "responsibilities"
    ]
    
    # Check if at least 4 keywords are present
    match_count = sum(1 for kw in keywords if kw in text_lower)
    
    # Also check for product codes (e.g., sequences of caps and numbers)
    code_matches = len(re.findall(r'\b[A-Z]{2,4}[0-9]{3,5}[A-Z]*\b', text))
    
    return match_count >= 4 or code_matches >= 5

def extract_structured_data_from_page(page_text: str) -> Dict[str, Any]:
    """
    Sends the raw page text to the LLM to extract perfectly structured JSON
    representing the enterprise document contents.
    """
    if not page_text or not page_text.strip():
        return {}
        
    prompt = f"Raw OCR Text:\n{page_text}\n\nReturn the JSON object now."
    
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
                if isinstance(parsed_data, dict):
                    return parsed_data
                return {}
            except json.JSONDecodeError as je:
                print(f"[Extraction] JSON Decode Error: {je}. Raw output: {result_text[:200]}...")
                return {}
        else:
            print(f"[Extraction] Ollama API Error: {response.text}")
            return {}
    except Exception as e:
        print(f"[Extraction] Error calling LLM: {e}")
        return {}

def format_structured_data_for_embedding(root_obj: Dict[str, Any]) -> List[str]:
    """
    Converts the structured root JSON object into a list of highly readable, key-value string chunks.
    Each item in 'Extracted_Data' becomes its own chunk, enriched with the root metadata.
    """
    if not root_obj or not isinstance(root_obj, dict):
        return []
        
    doc_type = root_obj.get("Document_Type", "Unknown")
    primary_entities = root_obj.get("Primary_Entities", [])
    category_context = root_obj.get("Category_Context", "")
    
    entities_str = ", ".join(primary_entities) if isinstance(primary_entities, list) else str(primary_entities)
    
    chunks = []
    extracted_data = root_obj.get("Extracted_Data", [])
    if not isinstance(extracted_data, list):
        # Fallback if the LLM returned a single object instead of a list
        extracted_data = [extracted_data]
        
    for item in extracted_data:
        if not isinstance(item, dict):
            continue
            
        lines = []
        lines.append(f"Document Type: {doc_type}")
        if category_context:
            lines.append(f"Category Context: {category_context}")
        if entities_str:
            lines.append(f"Primary Entities: {entities_str}")
            
        item_name = item.get("Item_Name_or_Catalogue_Number")
        if item_name:
            lines.append(f"Item / Catalogue Number: {item_name}")
            
        attributes = item.get("Attributes", {})
        if isinstance(attributes, dict):
            sub_items = [f"{k}: {v}" for k, v in attributes.items() if v is not None and v != ""]
            if sub_items:
                lines.append(f"Attributes: " + ", ".join(sub_items))
                
        ui_elements = item.get("UI_Elements", [])
        if ui_elements and isinstance(ui_elements, list):
            lines.append(f"UI Elements: " + ", ".join(str(u) for u in ui_elements))
            
        for key in ["Discrepancy_Context", "Notes_and_Modifications"]:
            val = item.get(key)
            if val and val != "null":
                lines.append(f"{key.replace('_', ' ')}: {val}")
                
        chunks.append("\n".join(lines))
        
    return chunks
