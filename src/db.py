# src/db.py
import sqlite3
from pathlib import Path

DB_PATH = Path("retrieval_data.db")

def get_connection():
    return sqlite3.connect(DB_PATH)

def init_db():
    with get_connection() as conn:
        cur = conn.cursor()

        # Table for all documents
        cur.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT,
                text TEXT
            )
        """)

        # Table for retrieval results
        cur.execute("""
            CREATE TABLE IF NOT EXISTS retrieval_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT,
                doc_id INTEGER,
                score REAL,
                FOREIGN KEY(doc_id) REFERENCES documents(id)
            )
        """)

        conn.commit()
