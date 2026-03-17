import sqlite3

DB_PATH = "attendance.db"  # database file

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

# Create table if it doesn't exist
c.execute("""
CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    date TEXT NOT NULL,
    time TEXT NOT NULL,
    UNIQUE(name, date)  -- avoid duplicates for same day
)
""")

conn.commit()
conn.close()
print("[INFO] Database initialized successfully.")
