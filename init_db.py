import sqlite3
import json
from pathlib import Path

def init_db(json_path: str, db_path: str):
    """
    Reads the competitor intelligence JSON and populates a relational SQLite database.
    """
    json_file = Path(json_path)
    if not json_file.exists():
        print(f"Error: {json_path} not found. Please run the extraction agent first.")
        return

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.executescript("""
        DROP TABLE IF EXISTS Weaknesses;
        DROP TABLE IF EXISTS Strengths;
        DROP TABLE IF EXISTS Features;
        DROP TABLE IF EXISTS Competitors;
        
        CREATE TABLE Competitors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            price REAL
        );
        
        CREATE TABLE Features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            competitor_id INTEGER NOT NULL,
            feature_text TEXT NOT NULL,
            FOREIGN KEY (competitor_id) REFERENCES Competitors(id)
        );
        
        CREATE TABLE Strengths (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            competitor_id INTEGER NOT NULL,
            strength_text TEXT NOT NULL,
            FOREIGN KEY (competitor_id) REFERENCES Competitors(id)
        );
        
        CREATE TABLE Weaknesses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            competitor_id INTEGER NOT NULL,
            weakness_text TEXT NOT NULL,
            FOREIGN KEY (competitor_id) REFERENCES Competitors(id)
        );
    """)

    for comp in data:
        name = comp.get("name", "Unknown")
        price = comp.get("price") 
        
        try:
            cursor.execute("INSERT INTO Competitors (name, price) VALUES (?, ?)", (name, price))
            comp_id = cursor.lastrowid
        except sqlite3.IntegrityError:
            print(f"Skipping duplicate competitor: {name}")
            continue

        for feature in comp.get("features", []):
            if feature:
                cursor.execute("INSERT INTO Features (competitor_id, feature_text) VALUES (?, ?)", (comp_id, feature))

        for strength in comp.get("strengths", []):
            if strength:
                cursor.execute("INSERT INTO Strengths (competitor_id, strength_text) VALUES (?, ?)", (comp_id, strength))

        for weakness in comp.get("weaknesses", []):
            if weakness:
                cursor.execute("INSERT INTO Weaknesses (competitor_id, weakness_text) VALUES (?, ?)", (comp_id, weakness))

    conn.commit()
    conn.close()
    print(f"Successfully populated database at {db_path} with {len(data)} competitors.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Initialize SQLite database from JSON")
    parser.add_argument("--json", type=str, default="json/competitor_intelligence.json", help="Path to input JSON")
    parser.add_argument("--db", type=str, default="database.db", help="Path to output SQLite database")
    args = parser.parse_args()
    
    init_db(args.json, args.db) 
