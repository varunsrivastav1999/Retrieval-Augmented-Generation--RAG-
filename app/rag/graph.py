import os
import json
import requests
from typing import List, Dict, Any

from app.rag.model_loader import extract_entities, extract_triplets, get_ollama_generate_url

# Neo4j is optional — app runs fine without it
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "rag_password")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

class GraphDB:
    def __init__(self):
        self.driver = None
        if not NEO4J_AVAILABLE:
            print("[GraphDB] neo4j package not installed — GraphRAG disabled (app runs fine without it)")
            return
        try:
            self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            # Verify connectivity with a lightweight check
            self.driver.verify_connectivity()
            print(f"[GraphDB] ✅ Connected to Neo4j at {NEO4J_URI}")
        except Exception as e:
            self.driver = None
            print(f"[GraphDB] ⚠️ Neo4j unavailable — GraphRAG disabled (app runs fine without it): {e}")

    def close(self):
        if self.driver:
            try:
                self.driver.close()
            except Exception:
                pass

    def populate_from_chunk(self, chunk_id: str, text: str, tenant_id: str):
        """Extracts entities and relationships, and creates them in Neo4j."""
        if not self.driver:
            return
            
        triplets = extract_triplets(text)
        if not triplets:
            return
            
        query = """
        UNWIND $triplets AS trip
        MERGE (n1:Entity {name: trip[0], tenant_id: $tenant_id})
        MERGE (n2:Entity {name: trip[2], tenant_id: $tenant_id})
        MERGE (n1)-[r:RELATED_TO {verb: trip[1]}]->(n2)
        ON CREATE SET r.weight = 1
        ON MATCH SET r.weight = r.weight + 1
        """
        try:
            with self.driver.session() as session:
                session.run(query, triplets=triplets, tenant_id=tenant_id)
        except Exception as e:
            print(f"[GraphDB] Error populating graph: {e}")

    def text_to_cypher(self, query: str) -> str:
        """Uses Ollama to translate natural language into a Cypher query."""
        prompt = f"""
        You are a Neo4j Cypher expert. Convert the following natural language query into a Cypher query.
        The graph schema is:
        - Nodes: Entity (properties: name, tenant_id)
        - Relationships: RELATED_TO (properties: verb, weight)
        
        Examples:
        - "What is related to the compressor?" -> MATCH (n:Entity {{name: "compressor"}})-[r:RELATED_TO]->(m) RETURN n.name, r.verb, m.name LIMIT 25
        - "Show all entities" -> MATCH (n:Entity) RETURN n.name LIMIT 50
        - "What is connected to sensor A?" -> MATCH (n:Entity {{name: "sensor A"}})-[r:RELATED_TO]-(m) RETURN n.name, r.verb, m.name LIMIT 25
        
        ALWAYS add LIMIT 50 to the query to prevent returning too many results.
        
        Natural language query: {query}
        
        Return ONLY the Cypher query. No explanation, no markdown.
        Cypher query:
        """
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0}
        }
        try:
            response = requests.post(get_ollama_generate_url(), json=payload, timeout=10)
            if response.status_code == 200:
                cypher = response.json().get("response", "").strip()
                if cypher.startswith("```cypher"):
                    cypher = cypher[9:-3].strip()
                elif cypher.startswith("```"):
                    cypher = cypher[3:-3].strip()
                # Ensure LIMIT clause exists (safety: prevent full graph dump)
                if "limit" not in cypher.lower():
                    cypher = cypher.rstrip().rstrip(";") + " LIMIT 50"
                return cypher
        except Exception as e:
            print(f"[GraphDB] Text-to-Cypher error: {e}")
        return ""

    def query_graph(self, nl_query: str, tenant_id: str) -> str:
        """Translates NL to Cypher, executes it, and returns the result as context."""
        if not self.driver:
            return ""
            
        cypher = self.text_to_cypher(nl_query)
        if not cypher:
            return ""
            
        print(f"[GraphDB] Executing Cypher: {cypher}")
        try:
            with self.driver.session() as session:
                # Enforce tenant isolation: inject tenant_id filter into generated Cypher
                if "WHERE" in cypher.upper():
                    isolated = cypher.replace("WHERE", f"WHERE n.tenant_id = '{tenant_id}' AND")
                else:
                    isolated = cypher.rstrip().rstrip(";") + f" WHERE n.tenant_id = '{tenant_id}'"
                result = session.run(isolated)
                records = [record.data() for record in result]
                if not records:
                    return ""
                # Limit results for safety
                records = records[:50]
                return f"Graph Knowledge: {json.dumps(records)}"
        except Exception as e:
            print(f"[GraphDB] Query error: {e}")
            return ""

graph_db = GraphDB()
