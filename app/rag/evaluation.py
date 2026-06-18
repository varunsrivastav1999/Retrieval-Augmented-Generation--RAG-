import os
from typing import List, Dict, Any
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434").replace("/api/generate", "")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")

def evaluate_rag_response(question: str, answer: str, contexts: List[str]) -> Dict[str, Any]:
    """
    Evaluates a RAG response using RAGAS.
    Note: RAGAS typically expects 'ground_truth' for context_recall and context_precision,
    but without ground truth, we can only safely measure faithfulness and answer_relevancy.
    """
    try:
        # Initialize Langchain Ollama Wrapper
        llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_URL)
        
        # Initialize Embeddings
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

        # Create the RAGAS dataset format
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }
        dataset = Dataset.from_dict(data)

        # Run evaluation (faithfulness checks if answer is derived from context, relevancy checks if it answers the question)
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=llm,
            embeddings=embeddings,
        )
        
        return {
            "faithfulness": result.get("faithfulness", 0.0),
            "answer_relevancy": result.get("answer_relevancy", 0.0)
        }
    except Exception as e:
        print(f"[RAGAS] Evaluation failed: {e}")
        return {
            "faithfulness": None,
            "answer_relevancy": None,
            "error": str(e)
        }
