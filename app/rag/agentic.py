import os
import json
import time
import asyncio
import re
from typing import List, Dict, Any, Optional

import requests
from app.rag.retrieval import perform_hybrid_search, perform_multi_query_search
from app.rag.context import assemble_context, _context_sources
from app.rag.reranker import rerank_results
from app.rag.grounding import compute_grounding_score, verify_answer_grounding, build_strict_grounding_prompt
from app.rag.model_loader import get_ollama_generate_url, OLLAMA_MODEL

OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "1024"))

class AgenticRAGPipeline:
    def __init__(self, db_session, tenant_id: str, original_top_k: int):
        self.db = db_session
        self.tenant_id = tenant_id
        self.original_top_k = original_top_k
        self.ollama_url = get_ollama_generate_url()
        self.model = OLLAMA_MODEL

    async def _async_ollama_generate(self, prompt: str, system: str = "") -> str:
        """Call Ollama locally but asynchronously so we don't block the event loop."""
        # Using aiohttp or run_in_executor
        loop = asyncio.get_event_loop()
        
        def _call():
            payload = {
                "model": self.model,
                "system": system,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": OLLAMA_NUM_PREDICT,
                    "temperature": 0.0,
                    "num_ctx": int(os.getenv("OLLAMA_CONTEXT_LENGTH", "8192")),
                }
            }
            try:
                resp = requests.post(self.ollama_url, json=payload, timeout=OLLAMA_TIMEOUT_SECONDS)
                if resp.status_code == 200:
                    return resp.json().get("response", "").strip()
            except Exception as e:
                print(f"[AgenticRAG] Ollama generation failed: {e}")
            return ""
            
        return await loop.run_in_executor(None, _call)

    async def planner_stage(self, query: str, initial_context: str) -> Dict[str, Any]:
        """Determine if we need scope discovery, answer plan, or empty plan."""
        system_prompt = (
            "You are an enterprise RAG Planner. Analyze the user's query and the provided initial context.\n"
            "If the context completely and directly answers the query, output an 'empty_plan'.\n"
            "If the query requires combining multiple distinct facts, or is complex, break it down into 2-3 specific sub-questions in 'answer_plan'.\n"
            "If the query is ambiguous, output 'scope_discovery' with 2 probing questions.\n"
            "Output valid JSON ONLY in this format:\n"
            "{\n"
            '  "plan_type": "empty_plan" | "answer_plan" | "scope_discovery",\n'
            '  "tasks": ["sub-question 1", "sub-question 2"]\n'
            "}"
        )
        prompt = f"Query: {query}\n\nInitial Context snippet:\n{initial_context[:2000]}"
        
        response = await self._async_ollama_generate(prompt, system=system_prompt)
        
        # Parse JSON safely
        try:
            # Extract JSON block if surrounded by markdown
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            
            plan = json.loads(json_str)
            if "plan_type" not in plan or "tasks" not in plan:
                raise ValueError("Missing required keys")
            return plan
        except Exception as e:
            print(f"[AgenticRAG] Planner failed to output valid JSON. Defaulting to empty_plan. Raw: {response}")
            return {"plan_type": "empty_plan", "tasks": []}

    async def execute_task(self, sub_question: str) -> Dict[str, Any]:
        """A mini-agent that retrieves context for a sub-question and answers it."""
        loop = asyncio.get_event_loop()
        
        def _retrieve():
            chunks = perform_multi_query_search(self.db, [sub_question], self.tenant_id, top_k=self.original_top_k)
            reranked = rerank_results(sub_question, chunks, top_n=self.original_top_k)
            context = assemble_context(sub_question, reranked, db=self.db)
            return context
            
        context = await loop.run_in_executor(None, _retrieve)
        
        context_text = "\n---\n".join([c.get('text', '') for c in context])
        
        prompt = build_strict_grounding_prompt(sub_question, context_text, broad_query=False)
        
        answer = await self._async_ollama_generate(prompt, system="")
        
        return {
            "sub_question": sub_question,
            "answer": answer,
            "context": context
        }

    async def synthesize(self, query: str, task_results: List[Dict[str, Any]], initial_context: List[Dict[str, Any]]) -> str:
        """Combine all sub-answers and initial context into a final cohesive response."""
        initial_text = "\n---\n".join([c.get('text', '') for c in initial_context[:5]])
        context_text = f"Initial Context:\n{initial_text}\n\n"
        
        for i, task in enumerate(task_results):
            context_text += f"Sub-Task {i+1} Output for '{task['sub_question']}':\n{task['answer']}\n\n"
            
        prompt = build_strict_grounding_prompt(query, context_text, broad_query=True)
        
        return await self._async_ollama_generate(prompt, system="")

    async def run(self, query: str) -> Dict[str, Any]:
        """Run the full Agentic Plan-and-Execute pipeline."""
        start_time = time.time()
        
        # Stage 1: Initial Retrieval
        loop = asyncio.get_event_loop()
        def _initial_retrieval():
            # fast_path=False enables exact_catalogue_lookup, pulling tables offline
            chunks = perform_hybrid_search(self.db, query, self.tenant_id, top_k=self.original_top_k, fast_path=False)
            return chunks
        
        initial_context = await loop.run_in_executor(None, _initial_retrieval)
        initial_context_text = "\n".join([c.get('text', '') for c in initial_context])
        
        # Stage 2: Planner
        plan = await self.planner_stage(query, initial_context_text)
        print(f"[AgenticRAG] Plan formulated: {plan['plan_type']} with {len(plan['tasks'])} tasks.")
        
        final_context = list(initial_context)
        
        if plan["plan_type"] == "empty_plan" or not plan["tasks"]:
            # Fall back to standard generate if planner says no sub-tasks needed
            pass
            final_answer = "" # Indicates standard generation handles it later
        else:
            # Stage 3: Parallel Task Execution
            tasks = [self.execute_task(t) for t in plan["tasks"]]
            task_results = await asyncio.gather(*tasks)
            
            # Aggregate context
            for tr in task_results:
                for c in tr["context"]:
                    if c not in final_context:
                        final_context.append(c)
                        
            # Stage 4: Synthesis
            final_answer = await self.synthesize(query, task_results, initial_context)
        
        # Stage 5: Verification
        # Let's verify the initial retrieval grounding if empty plan, or synthesize grounding
        grounding_result = compute_grounding_score(query, final_context)
        
        return {
            "answer": final_answer, # if empty, main.py will generate it
            "context": final_context,
            "sources": _context_sources(final_context),
            "grounding": grounding_result,
            "latency_ms": int((time.time() - start_time) * 1000),
            "plan_type": plan["plan_type"]
        }
