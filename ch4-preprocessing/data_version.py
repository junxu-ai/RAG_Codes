import sqlite3
from datetime import datetime

# Connect to SQLite database
conn = sqlite3.connect('knowledge_base.db')
cursor = conn.cursor()

# Create table with versioning
cursor.execute('''
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    content TEXT,
    effective_date DATE
)
''')

# Insert a document with an effective date
def insert_document(content, effective_date):
    cursor.execute('''
    INSERT INTO documents (content, effective_date)
    VALUES (?, ?)
    ''', (content, effective_date))
    conn.commit()

# Retrieve document as of a specific date
def get_document_as_of(date):
    cursor.execute('''
    SELECT content FROM documents
    WHERE effective_date <= ?
    ORDER BY effective_date DESC
    LIMIT 1
    ''', (date,))
    result = cursor.fetchone()
    return result[0] if result else None

# Example usage
insert_document('CEO is Alice', '2023-01-01')
insert_document('CEO is Bob', '2025-06-01')

# Retrieve the CEO as of 2024-12-31
print(get_document_as_of('2024-12-31'))  # Output: 'CEO is Alice'

# Retrieve the CEO as of 2025-08-31
print(get_document_as_of('2025-08-31'))  # Output: 'CEO is Bob'
