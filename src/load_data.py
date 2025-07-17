from pathlib import Path
import sqlite3
import pandas as pd

def main():
    # Paths
    repo_root = Path(__file__).parent.parent
    raw_csv = repo_root / 'data' / 'raw' / 'train.csv'
    processed_dir = repo_root / 'data' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Load and clean
    df = pd.read_csv(raw_csv)
    print(f"Loaded {len(df)} raw records")

    # Drop any rows where Context or Response is missing
    before = len(df)
    df = df.dropna(subset=['Context', 'Response'])
    after = len(df)
    print(f"Dropped {before-after} rows with nulls → {after} records remain")

    # Setup SQLite
    db_path = processed_dir / 'conversations.db'
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY,
            context TEXT NOT NULL,
            response TEXT NOT NULL
        );
    """)
    conn.commit()

    # Clear & insert
    conn.execute("DELETE FROM conversations;")
    data_tuples = [(i+1, row.Context, row.Response) for i, row in df.iterrows()]
    conn.executemany(
        "INSERT INTO conversations (id, context, response) VALUES (?, ?, ?)",
        data_tuples
    )
    conn.commit()
    conn.close()
    print(f"Saved {after} conversations to SQLite at {db_path}")

    # Export JSON too
    json_path = processed_dir / 'conversations.json'
    df.to_json(json_path, orient='records', indent=2)
    print(f"Exported JSON to {json_path}")

if __name__ == '__main__':
    main()
