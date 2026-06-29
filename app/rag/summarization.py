"""
=============================================================================
 Enterprise Level RAG: Document Summarization (Map-Reduce)
=============================================================================
 Used for broad queries (e.g., "Summarize the entire manual") where standard 
 chunk limits would truncate information.
=============================================================================
"""

import os
import requests
import asyncio
from typing import List, Dict, Any

from app.rag.model_loader import get_ollama_generate_url, OLLAMA_MODEL

OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "300"))

async def _async_ollama_generate(prompt: str, system: str = "") -> str:
    loop = asyncio.get_running_loop()
    def _call():
        payload = {
            "model": OLLAMA_MODEL,
            "system": system,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": 2048,
                "temperature": 0.2,
                "num_ctx": int(os.getenv("OLLAMA_CONTEXT_LENGTH", "8192")),
            }
        }
        try:
            resp = requests.post(get_ollama_generate_url(), json=payload, timeout=OLLAMA_TIMEOUT_SECONDS)
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()
        except Exception as e:
            print(f"[Summarization] Ollama generation failed: {e}")
        return ""
    return await loop.run_in_executor(None, _call)

async def _map_stage(chunk: Dict[str, Any], query: str) -> str:
    text = chunk.get("text", "")
    source = chunk.get("metadata", {}).get("source", "Unknown")
    prompt = (
        f"You are a summarization assistant. Extract the key facts from the following text that are relevant to the query: '{query}'. "
        f"Do not write introductory filler. Include the source file name '{source}' in your summary if relevant.\n\n"
        f"Text:\n{text}"
    )
    return await _async_ollama_generate(prompt, system="Expert summarizer.")

async def map_reduce_summarize(query: str, chunks: List[Dict[str, Any]]) -> str:
    """
    Perform a Map-Reduce summarization over a large number of chunks.
    1. Map: Summarize each chunk independently (or in batches) in parallel.
    2. Reduce: Combine all summaries into a final coherent answer.
    """
    if not chunks:
        return "No information found to summarize."

    print(f"[Summarization] Starting Map stage for {len(chunks)} chunks...")
    
    # Run Map stage concurrently. Batch chunks to reduce calls.
    batch_size = 5
    batched_chunks = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        combined_text = "\n\n".join([f"Source: {c.get('metadata', {}).get('source', 'Unknown')}\nText: {c.get('text', '')}" for c in batch])
        batched_chunks.append({"text": combined_text, "metadata": {"source": "Multiple"}})

    map_tasks = [_map_stage(c, query) for c in batched_chunks]
    map_results = await asyncio.gather(*map_tasks)

    # Filter out empty results
    map_results = [m for m in map_results if m and len(m.strip()) > 10]
    
    if not map_results:
        return "Failed to extract summary from documents."

    print(f"[Summarization] Map stage complete. Starting Reduce stage with {len(map_results)} partial summaries...")
    
    reduce_context = "\n---\n".join(map_results)
    
    prompt = (
        f"You are a master synthesizer. You are given several partial summaries of documents. "
        f"Combine them into a comprehensive, final answer to the query: '{query}'. "
        f"Ensure no important details are lost, remove redundancies, and use clear markdown formatting (headings, bullet points).\n\n"
        f"Partial Summaries:\n{reduce_context}"
    )
    
    final_summary = await _async_ollama_generate(prompt, system="Expert technical writer.")
    return final_summary

def run_map_reduce_summarize_sync(query: str, chunks: List[Dict[str, Any]]) -> str:
    return asyncio.run(map_reduce_summarize(query, chunks))
