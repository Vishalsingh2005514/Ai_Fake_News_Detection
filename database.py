"""
database.py
-----------
Handles all SQLite database operations for the AI Fake News Detector.
Stores user queries, verdicts, confidence scores, and timestamps.
"""

import sqlite3
import datetime

# Path to the local SQLite database file
DB_PATH = "history.db"


def get_connection():
    """Open and return a connection to the database."""
    return sqlite3.connect(DB_PATH)


def init_db():
    """
    Create the history table if it doesn't already exist.
    Called once when the app starts up.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_input TEXT    NOT NULL,
            verdict    TEXT    NOT NULL,
            confidence INTEGER NOT NULL,
            timestamp  TEXT    NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def save_result(user_input, verdict, confidence):
    """
    Save one analysis result to the database.

    Args:
        user_input (str): The news text the user submitted.
        verdict    (str): 'REAL', 'FAKE', or 'UNCERTAIN'.
        confidence (int): Model confidence percentage (0–100).
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Get current date and time as a readable string
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute(
        "INSERT INTO history (user_input, verdict, confidence, timestamp) VALUES (?, ?, ?, ?)",
        (user_input, verdict, confidence, timestamp)
    )

    conn.commit()
    conn.close()


def get_all_history():
    """
    Fetch all saved history records, newest first.

    Returns:
        list of tuples: (user_input, verdict, confidence, timestamp)
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT user_input, verdict, confidence, timestamp FROM history ORDER BY id DESC"
    )
    rows = cursor.fetchall()

    conn.close()
    return rows


def delete_all_history():
    """Permanently delete all records from the history table."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM history")

    conn.commit()
    conn.close()


# Keep old function names working as aliases (backwards compatibility)
insert_history = save_result
fetch_history  = get_all_history
clear_history  = delete_all_history
