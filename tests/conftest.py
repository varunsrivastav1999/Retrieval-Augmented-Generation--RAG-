import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DATABASE_URL", "postgresql+psycopg2://test:test@127.0.0.1:1/test")
os.environ.setdefault("RAG_OCR_ENABLED", "false")
