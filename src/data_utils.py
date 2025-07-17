# src/data_utils.py
import sqlite3
import re
from typing import List, Tuple

DB_PATH = "data/processed/conversations.db"

def fetch_examples(query: str, k: int = 3) -> List[Tuple[str,str]]:
    """
    Return up to k (context, response) pairs whose context
    contains ANY of the top keywords from the query.
    """
    # 1) Extract simple keyword list (alphanumeric, length>3)
    words = re.findall(r"\b\w{4,}\b", query.lower())
    # Dedupe and pick top 5
    keywords = list(dict.fromkeys(words))[:5]
    if not keywords:
        return []

    # 2) Build WHERE clause: context LIKE ? OR context LIKE ? ...
    placeholders = " OR ".join(["context LIKE ?"] * len(keywords))
    sql = f"SELECT context, response FROM conversations WHERE {placeholders} LIMIT ?;"

    # 3) Params: ["%word1%", "%word2%", ..., k]
    params = [f"%{w}%" for w in keywords] + [k]

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return rows
