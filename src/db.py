# src/db.py
import sqlite3
import os

# Absolute path to DB
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rag_bias.db"))

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db():
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                text TEXT NOT NULL
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_source ON documents(source);")
        conn.commit()

def clear_documents():
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM documents;")
        conn.commit()
