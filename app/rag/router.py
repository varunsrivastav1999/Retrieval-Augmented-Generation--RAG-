import json
import requests
from app.rag.model_loader import get_ollama_generate_url, OLLAMA_MODEL

class Router:
    """Logical Router: Routes a user query to Vector, Graph, or SQL paths based on intent."""
    
    def route_query(self, query: str) -> str:
        prompt = f"""
        You are a highly intelligent query router for a RAG system.
        Classify the following user query into exactly ONE of these categories:
        - "graph" : If the query asks for relationships between entities (e.g. "Who works with X?", "How is X related to Y?").
        - "sql" : If the query asks about document metadata, file counts, or file types (e.g. "How many PDF files do I have?", "List all documents").
        - "vector" : If the query asks about the content or information within the documents (e.g. "What is X?", "Explain Y").
        
        User Query: {query}
        
        Return ONLY the single word category ("graph", "sql", or "vector"). No explanations.
        """
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0}
        }
        try:
            response = requests.post(get_ollama_generate_url(), json=payload, timeout=5)
            if response.status_code == 200:
                classification = response.json().get("response", "").strip().lower()
                for valid in ["graph", "sql", "vector"]:
                    if valid in classification:
                        return valid
        except Exception as e:
            print(f"[Router] Routing failed, defaulting to vector: {e}")
            
        return "vector"  # Default fallback

query_router = Router()
