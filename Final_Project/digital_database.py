"""
digital_database.py - the system's official catalog: book records and
robot state. Kept deliberately separate from shelves.py (physical
ground truth) - a mismatch between the two is the informational noise
this project reconciles.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

DB_PATH = "digital_database.db"


SCHEMA = """
CREATE TABLE IF NOT EXISTS books (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    shelf_id TEXT NOT NULL,
    weight_kg REAL NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('available', 'checked_out')),
    last_confirmed_at TEXT
);

CREATE TABLE IF NOT EXISTS robot_state (
    id TEXT PRIMARY KEY,
    battery_pct INTEGER NOT NULL,
    capacity_kg REAL NOT NULL,
    current_shelf_id TEXT
);

CREATE TABLE IF NOT EXISTS scan_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    book_id TEXT NOT NULL,
    shelf_id TEXT NOT NULL,
    found INTEGER NOT NULL,
    confidence REAL NOT NULL,
    scanned_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS transaction_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    book_id TEXT NOT NULL,
    action TEXT NOT NULL,
    db_status TEXT,
    scan_status TEXT,
    resolved_status TEXT,
    timestamp TEXT NOT NULL
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


def get_book_record(book_id: str, db_path: str = DB_PATH) -> Optional[Dict[str, Any]]:
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT id, title, shelf_id, weight_kg, status, last_confirmed_at "
            "FROM books WHERE id = ?",
            (book_id,),
        ).fetchone()
        return dict(row) if row else None


def get_robot_state(robot_id: str, db_path: str = DB_PATH) -> Optional[Dict[str, Any]]:
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT id, battery_pct, capacity_kg, current_shelf_id "
            "FROM robot_state WHERE id = ?",
            (robot_id,),
        ).fetchone()
        return dict(row) if row else None


def update_book_status(book_id: str, new_status: str, db_path: str = DB_PATH) -> None:
    """Only call this AFTER reconciliation has resolved a conflict -- never
    write a status straight from a raw scan or an LLM claim."""
    assert new_status in ("available", "checked_out")
    now = datetime.now(timezone.utc).isoformat()
    with get_conn(db_path) as conn:
        conn.execute(
            "UPDATE books SET status = ?, last_confirmed_at = ? WHERE id = ?",
            (new_status, now, book_id),
        )
