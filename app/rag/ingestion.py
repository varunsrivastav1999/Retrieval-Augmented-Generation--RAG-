import hashlib
import fitz # PyMuPDF
from sentence_transformers import SentenceTransformer
from app.database import SessionLocal, DocumentChunk
from sqlalchemy.exc import IntegrityError

embedder = SentenceTransformer('all-MiniLM-L6-v2') 

def hash_chunk(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

import re

def semantic_chunking(text: str) -> list:
    # Normalize accidental line breaks from PDF extraction
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    
    # Split by actual sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    for s in sentences:
        s = s.strip()
        if not s: continue
            
        if len(current_chunk) + len(s) < 1200:
            current_chunk += s + " "
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = s + " "
            
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
        
    return chunks

def ingest_pdf(pdf_path: str):
    """
    Parses PDF, chunks semantically, deduplicates using hash, embeds, and saves to pgvector.
    """
    db = SessionLocal()
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
            
        chunks = semantic_chunking(text)
        
        for i, chunk in enumerate(chunks):
            chunk_hash = hash_chunk(chunk)
            
            # Check if exists (deduplication)
            existing = db.query(DocumentChunk).filter(DocumentChunk.chunk_hash == chunk_hash).first()
            if existing:
                continue
            
            vector = embedder.encode(chunk).tolist()
            
            doc_chunk = DocumentChunk(
                doc_id=pdf_path,
                chunk_hash=chunk_hash,
                text_content=chunk,
                section=i,
                doc_metadata={"type": "pdf", "page_count": len(doc)},
                embedding=vector
            )
            db.add(doc_chunk)
            try:
                db.commit()
                print(f"[Ingest] Saved chunk {i} for {pdf_path} into pgvector")
            except IntegrityError:
                db.rollback() # Handle race conditions
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
    finally:
        db.close()
