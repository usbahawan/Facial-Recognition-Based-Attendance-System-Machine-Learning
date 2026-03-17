import sqlite3

DB_PATH = "../attendance.db"

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
c.execute("SELECT * FROM attendance ORDER BY date DESC, time DESC")
rows = c.fetchall()
conn.close()

print("ID | Name | Date | Time")
print("-"*30)
for row in rows:
    print(f"{row[0]} | {row[1]} | {row[2]} | {row[3]}")
