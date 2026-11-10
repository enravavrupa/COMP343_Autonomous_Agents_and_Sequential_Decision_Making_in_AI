"""
shelves.py - physical ground truth (what's actually on the shelf right
now), plus per-shelf bandit reliability tracking (shelf_reliability),
independent of digital_database.py's catalog.
""" 



from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Optional

DB_PATH = "shelves.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS shelf_truth (
    book_id TEXT PRIMARY KEY,
    shelf_id TEXT NOT NULL,
    physically_present INTEGER NOT NULL,   -- 1 = on the shelf right now, 0 = not
    last_updated TEXT
); -- adding this to check if shelf needs to be checked
CREATE TABLE IF NOT EXISTS shelf_reliability (
    shelf_id TEXT PRIMARY KEY,
    success_score REAL NOT NULL DEFAULT 0.5, -- random checkin
    attempts INTEGER NOT NULL DEFAULT 0
);
"""


@contextmanager
def get_conn(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str = DB_PATH) -> None:
    with get_conn(db_path) as conn:
        conn.executescript(SCHEMA)


def get_shelf_truth(book_id: str, db_path: str = DB_PATH) -> Optional[Dict[str, Any]]:
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT book_id, shelf_id, physically_present, last_updated "
            "FROM shelf_truth WHERE book_id = ?",
            (book_id,),
        ).fetchone()
        return dict(row) if row else None


def set_shelf_truth(book_id: str, shelf_id: str, physically_present: bool,
                     db_path: str = DB_PATH) -> None:
    """For manually staging a demo scenario, e.g. from a dashboard button
    that simulates 'someone quietly re-shelved this without telling the
    catalog.' Not called by sensor.py itself -- sensor.py only reads."""
    now = datetime.now(timezone.utc).isoformat()
    with get_conn(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO shelf_truth VALUES (?, ?, ?, ?)",
            (book_id, shelf_id, int(physically_present), now),
        )

STEP = 0.1

# adding to get the value of trust we have in this shelf 
def get_shelf_reliability(shelf_id, db_path=DB_PATH) -> dict:
    with get_conn(db_path) as conn:
        row = conn.execute("SELECT * FROM shelf_reliability WHERE shelf_id=?", (shelf_id,)).fetchone()
        if row is None:
            return {"shelf_id": shelf_id, "success_score": 0.5, "attempts": 0}
        return dict(row)

# everytime the shelf and catalog matach, the value increments which means we have more trust in it
# everytime it fails, we decrement the trust/success_rate we have
def update_shelf_reliability(shelf_id, success: bool, db_path=DB_PATH) -> None:
    current = get_shelf_reliability(shelf_id, db_path=db_path)
    new_score = current["success_score"] + STEP if success else current["success_score"] - STEP
    new_score = max(0.0, min(1.0, new_score))
    new_attempts = current["attempts"] + 1
    with get_conn(db_path) as conn:
        conn.execute("INSERT OR REPLACE INTO shelf_reliability VALUES (?,?,?)",
                     (shelf_id, new_score, new_attempts))

if __name__ == "__main__":
    init_db()
    print("shelves.db created.")
    print(get_shelf_truth("book-003"))
