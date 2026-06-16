import os
import sys

# Ensure app is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.rag.parsers import _parse_docx, parse_file
from app.database import SessionLocal, DocumentChunk

def test_docx_parsing(docx_path):
    print(f"\n--- Testing DOCX Parsing: {docx_path} ---")
    result = parse_file(docx_path)
    if not result.success:
        print(f"FAILED to parse: {result.error}")
        return
    
    table_count = sum(len(page.tables) for page in result.pages)
    print(f"Total Pages Detected: {len(result.pages)}")
    print(f"Total Tables Detected: {table_count}")
    
    # Assertions
    if table_count < 4:
        print("❌ ERROR: Expected at least 4 tables, found fewer.")
    else:
        print("✅ Tables parsed successfully.")

def test_broad_retrieval(tenant_id, query):
    print(f"\n--- Testing Broad Retrieval: '{query}' ---")
    db = SessionLocal()
    from app.rag.retrieval import perform_multi_query_search
    from app.rag.reranker import rerank_results
    from app.rag.context import assemble_context

    chunks = perform_multi_query_search(db, [query], tenant_id, top_k=20)
    reranked = rerank_results(query, chunks, top_n=10)
    context = assemble_context(query, reranked, db=db)
    
    print(f"Chunks retrieved into context window: {len(context)}")
    
    page_nums = set()
    for c in context:
        page_num = c.get('metadata', {}).get('page_num')
        if page_num:
            page_nums.add(page_num)
            
    print(f"Pages represented in context window: {sorted(list(page_nums))}")
    if len(page_nums) >= 3:
        print("✅ Context successfully spans multiple pages (Broad retrieval working).")
    else:
        print("⚠️ Warning: Context does not span many pages. May still be fragmented.")

if __name__ == "__main__":
    print("==========================================")
    print(" RAG Regression Test ")
    print("==========================================")
    
    # You can pass a path to your DOCX file here
    sample_docx = os.getenv("TEST_DOCX_PATH", "sample.docx")
    tenant_id = os.getenv("TENANT_ID", "default")
    
    if os.path.exists(sample_docx):
        test_docx_parsing(sample_docx)
    else:
        print(f"Skipping DOCX test: file '{sample_docx}' not found. Set TEST_DOCX_PATH.")
        
    try:
        test_broad_retrieval(tenant_id, "how to setup production server")
    except Exception as e:
        print(f"Skipping DB Retrieval test: {e} (Is the DB running?)")
