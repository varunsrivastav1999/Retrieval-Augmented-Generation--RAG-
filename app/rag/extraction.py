import json
import re
import requests
from typing import List, Dict, Any

from app.rag.model_loader import get_ollama_generate_url, OLLAMA_MODEL

EXTRACTION_SYSTEM_PROMPT = """You are an elite AI data architect and universal enterprise data extractor. Your task is to process raw, poorly formatted, flattened OCR text from ANY industrial or corporate document (Financial Reports, HR Postings, Hardware Catalogs, SCADA/Robotics Manuals, SOPs, Maintenance Logs, and Defect Sheets) and convert it into a perfectly structured JSON array optimized for advanced RAG retrieval.

CRITICAL RULE: Extract ONLY information that is explicitly stated in the provided text.
Do NOT add, invent, infer, or assume any data not present in the source.
Do NOT fabricate part numbers, values, measurements, or specifications.
When resolving clusters or fixing formatting, do not change the underlying data values.

Follow these strict extraction and conditioning rules based on the document types you detect:

1. IF FINANCIAL, CORPORATE, OR COMPLIANCE REPORTS:
   - Extract numerical data precisely (e.g., Revenue, YoY Growth, Margins, Budgets, Headcount).
   - RECONCILE DISCREPANCIES: If a metric is reported differently across sections (e.g., cash-basis vs. ASC 606 adjusted), extract both values and capture the explanatory footnote in a `Discrepancy_Context` field.
   - Extract Risk Registers (Likelihood/Impact scores) and Compliance Certifications (Status/Renewal dates).

2. IF SCADA / ROBOTICS MANUALS & USER GUIDES:
   - Extract real-time screen configurations, tag mappings, and PLC addresses verbatim.
   - Map exact valve names, flow rates (cc/min), turbine speeds (rpm), and shaping air levels (NL/min).
   - Capture UI/Safety components (e.g., Red=Fault, Green=Healthy, Emergency Stops, Light Curtains) into a `UI_Safety_Signals` array.

3. IF SOPS, MAINTENANCE & PAINT DEFECT LOGS:
   - Convert sequential text into an ordered array under `Procedural_Steps`. Separated prerequisites from active steps.
   - For Quality/Defect logs, structure data into `Defect_Name`, `Potential_Root_Causes` (e.g., clogged CCV), and `Corrective_Actions` (e.g., activate Flush Pattern).

4. IF HARDWARE SPECIFICATION CATALOGS:
   - RESOLVE CLUSTERS: Disentangle clustered models and amperages into individual, isolated JSON objects so every part number has its own distinct row.
   - FIX DIMENSIONS: Normalize fragmented fractions into clean string formats (e.g., convert "167/8" to "16 7/8").

5. IF HR & ORGANIZATIONAL DATA:
   - Extract job roles, required experience, compensation (LPA/Salary), location, and departmental performance scores into standard key-value pairs.

STRICT JSON OUTPUT FORMAT:
Return ONLY valid JSON. Do not include markdown formatting, conversational prose, or block explanations. Use this universal schema to handle any scenario:

{
  "Document_Type": "Financial_Report | HR_Document | SCADA_Specification | Robotics_Manual | SOP | Quality_Log | Hardware_Catalog",
  "Metadata_Context": {
    "Overarching_Category_or_Department": "string",
    "System_Model_or_Fiscal_Period": "string or null",
    "Primary_Entities": ["string"]
  },
  "Structured_Payload": [
    {
      "Identifier": "Part Number / Step Name / Defect / Financial Metric / Job Title",
      "Core_Attributes": {
        "Financial_or_HR_Value": "string/numeric or null",
        "PLC_Address_or_Tag": "string or null",
        "Dimensions_or_Limits": "string or null",
        "Status_or_Compliance": "string or null"
      },
      "Procedural_Steps": ["string"],
      "UI_Safety_Signals": [
        {
          "Element": "string",
          "State_Condition": "string"
        }
      ],
      "Troubleshooting_and_Discrepancies": {
        "Root_Cause_or_Conflicting_Data": ["string"],
        "Remedy_or_Footnote_Explanation": ["string"]
      },
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
        "fault", "indicator", "status", "dashboard", "procedure", "step", "warning", "plc", "valve", "turbine", "nl/min",
        # SOPs / Maintenance
        "prerequisite", "defect", "root cause", "corrective action", "maintenance", "log",
        # Finance
        "revenue", "net profit", "fiscal", "quarter", "ebitda", "footnote", "discrepancy", "compliance", "yoy",
        # HR
        "salary", "ctc", "experience", "requirements", "responsibilities"
    ]
    
    # Check if at least 3 keywords are present
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
    Converts the structured universal JSON object into a list of highly readable, key-value string chunks.
    Each item in 'Structured_Payload' becomes its own chunk, enriched with the global metadata.
    """
    if not root_obj or not isinstance(root_obj, dict):
        return []
        
    doc_type = root_obj.get("Document_Type", "Unknown")
    
    metadata = root_obj.get("Metadata_Context", {})
    category = metadata.get("Overarching_Category_or_Department", "")
    system_model = metadata.get("System_Model_or_Fiscal_Period", "")
    primary_entities = metadata.get("Primary_Entities", [])
    
    entities_str = ", ".join(primary_entities) if isinstance(primary_entities, list) else str(primary_entities)
    
    chunks = []
    payload = root_obj.get("Structured_Payload", [])
    if not isinstance(payload, list):
        payload = [payload]
        
    for item in payload:
        if not isinstance(item, dict):
            continue
            
        lines = []
        lines.append(f"Document Type: {doc_type}")
        if category and str(category).lower() != "null":
            lines.append(f"Category/Department: {category}")
        if system_model and str(system_model).lower() != "null":
            lines.append(f"System/Model/Period: {system_model}")
        if entities_str and str(entities_str).lower() != "null" and entities_str != "[]":
            lines.append(f"Primary Entities: {entities_str}")
            
        identifier = item.get("Identifier")
        if identifier and str(identifier).lower() != "null":
            lines.append(f"Identifier: {identifier}")
            
        core_attrs = item.get("Core_Attributes", {})
        if isinstance(core_attrs, dict):
            sub_items = [f"{k.replace('_', ' ')}: {v}" for k, v in core_attrs.items() if v and str(v).lower() != "null"]
            if sub_items:
                lines.append(f"Attributes: " + ", ".join(sub_items))
                
        steps = item.get("Procedural_Steps", [])
        if steps and isinstance(steps, list):
            lines.append("Procedural Steps: " + " | ".join(str(s) for s in steps if str(s).lower() != "null"))
            
        ui_signals = item.get("UI_Safety_Signals", [])
        if ui_signals and isinstance(ui_signals, list):
            signal_strs = []
            for sig in ui_signals:
                if isinstance(sig, dict):
                    el = sig.get("Element", "")
                    st = sig.get("State_Condition", "")
                    if el or st:
                        signal_strs.append(f"{el} ({st})")
                else:
                    signal_strs.append(str(sig))
            if signal_strs:
                lines.append(f"UI/Safety Signals: " + ", ".join(signal_strs))
                
        trouble = item.get("Troubleshooting_and_Discrepancies", {})
        if isinstance(trouble, dict):
            rc = trouble.get("Root_Cause_or_Conflicting_Data", [])
            rem = trouble.get("Remedy_or_Footnote_Explanation", [])
            if rc and isinstance(rc, list):
                lines.append("Root Causes / Conflicts: " + ", ".join(str(r) for r in rc if str(r).lower() != "null"))
            if rem and isinstance(rem, list):
                lines.append("Remedies / Explanations: " + ", ".join(str(r) for r in rem if str(r).lower() != "null"))
                
        notes = item.get("Notes_and_Modifications")
        if notes and str(notes).lower() != "null":
            lines.append(f"Notes: {notes}")
            
        chunks.append("\n".join(lines))
        
    return chunks
