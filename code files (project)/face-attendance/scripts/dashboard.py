from flask import Flask, render_template_string
import sqlite3

app = Flask(__name__)
DB_PATH = "attendance.db"

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Attendance Dashboard</title>
    <style>
        body { font-family: Arial; margin: 30px; }
        table { border-collapse: collapse; width: 70%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #4CAF50; color: white; }
        tr:nth-child(even){background-color: #f2f2f2;}
    </style>
</head>
<body>
    <h1>Attendance Dashboard</h1>
    <table>
        <tr><th>ID</th><th>Name</th><th>Date</th><th>Time</th></tr>
        {% for row in rows %}
        <tr>
            <td>{{ row[0] }}</td>
            <td>{{ row[1] }}</td>
            <td>{{ row[2] }}</td>
            <td>{{ row[3] }}</td>
        </tr>
        {% endfor %}
    </table>
</body>
</html>
"""

@app.route('/')
def dashboard():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM attendance ORDER BY date DESC, time DESC")
    rows = c.fetchall()
    conn.close()
    return render_template_string(HTML_TEMPLATE, rows=rows)

if __name__ == '__main__':
    app.run(debug=True)
