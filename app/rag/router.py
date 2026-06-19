import re
import os
import requests
from app.rag.model_loader import get_ollama_generate_url, OLLAMA_MODEL

class Router:
    """
    Two-tier query router:
    Tier 1 (fast, ~0ms): Keyword patterns — covers 90%+ of queries instantly.
    Tier 2 (slow, ~500ms+): Ollama LLM fallback for ambiguous queries only.

    Routes to: "vector" (content questions), "graph" (relationship questions),
               "sql" (metadata/stats questions).
    """

    TIERS = {"vector", "graph", "sql", "raptor"}

    # --- Tier 1: Keyword patterns (O(1), no LLM needed) ---
    GRAPH_PATTERNS = [
        r"\b(?:relationship|relation|connect|link|associat|network|interact|interconnection|path\s*between)\b",
        r"\b(?:who\s+(?:works?|report|manage|lead|direct))\b",
        r"\b(?:how\s+(?:is|are)\s+\w+\s+(?:related|connected|linked))\b",
    ]
    SQL_PATTERNS = [
        r"\b(?:how\s+many|count|number\s+of|list\s+all|show\s+all)\b",
        r"\b(?:file\s*(?:type|format|count)|document\s*(?:count|list|type))\b",
        r"\b(?:metadata|statistics|summary|overview|total)\b",
    ]
    RAPTOR_PATTERNS = [
        r"\b(?:overall\s+theme|global\s+summary|general\s+overview|high-level\s+summary|entire\s+dataset)\b",
        r"\b(?:what\s+is\s+(?:this|the)\s+corpus\s+about)\b",
    ]
    _GRAPH_RE = re.compile("|".join(GRAPH_PATTERNS), re.IGNORECASE)
    _SQL_RE = re.compile("|".join(SQL_PATTERNS), re.IGNORECASE)
    _RAPTOR_RE = re.compile("|".join(RAPTOR_PATTERNS), re.IGNORECASE)

    def _route_keyword(self, query: str) -> str | None:
        """Instant keyword routing — returns route or None if ambiguous."""
        if self._GRAPH_RE.search(query):
            return "graph"
        if self._SQL_RE.search(query):
            return "sql"
        if self._RAPTOR_RE.search(query):
            return "raptor"
        return None  # ambiguous → fall through to LLM

    def _route_llm(self, query: str) -> str:
        """Fallback: use Ollama for ambiguous queries."""
        prompt = (
            "You are a highly intelligent query router. "
            "Classify into exactly ONE: 'graph' (entity relationships), "
            "'sql' (metadata/file counts), 'raptor' (global summary/themes of all files), or 'vector' (specific document content).\n\n"
            f"Query: {query}\n\nCategory:"
        )
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_ctx": int(os.getenv("OLLAMA_CONTEXT_LENGTH", "32768"))
            },
        }
        try:
            resp = requests.post(get_ollama_generate_url(), json=payload, timeout=15)
            if resp.status_code == 200:
                text = resp.json().get("response", "").strip().lower()
                for valid in self.TIERS:
                    if valid in text:
                        return valid
        except Exception as e:
            print(f"[Router] LLM fallback failed: {e}")
        return "vector"

    def route_query(self, query: str) -> str:
        result = self._route_keyword(query)
        if result is not None:
            print(f"[Router] Keyword → {result.upper()} (0ms)")
            return result
        result = self._route_llm(query)
        print(f"[Router] LLM → {result.upper()}")
        return result


query_router = Router()
