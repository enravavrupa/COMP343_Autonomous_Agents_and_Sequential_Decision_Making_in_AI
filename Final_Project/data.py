# populate_once.py -- run this ONE TIME to seed your .db files, then delete it or keep it around, doesn't matter
import digital_database as digital
import shelves 

digital.init_db()
shelves.init_db()

with digital.get_conn() as conn:
    conn.execute(
        "INSERT OR REPLACE INTO books VALUES (?, ?, ?, ?, ?, ?)",
        ("book-1", "Introduction to Algorithms", "shelf-12B", 1.8, "available", None),
    )
    conn.execute(
        "INSERT OR REPLACE INTO books VALUES (?, ?, ?, ?, ?, ?)",
        ("book-2", "Design Patterns", "shelf-14A", 1.2, "checked_out", None),
    )
    conn.execute(
        "INSERT OR REPLACE INTO books VALUES (?, ?, ?, ?, ?, ?)",
        ("book-3", "The Art of Computer Programming", "shelf-09C", 6.0, "checked_out", None),
    )
    conn.execute(
        "INSERT OR REPLACE INTO robot_state VALUES (?, ?, ?, ?)",
        ("robot-1", 40, 5.0, "shelf-A"),
    )

with shelves.get_conn() as conn:
    conn.execute(
        "INSERT OR REPLACE INTO shelf_truth VALUES (?, ?, ?, ?)",
        ("book-1", "shelf-12B", 1, None),   # agrees: present
    )
    conn.execute(
        "INSERT OR REPLACE INTO shelf_truth VALUES (?, ?, ?, ?)",
        ("book-2", "shelf-14A", 0, None),   # agrees: not present
    )
    conn.execute(
        "INSERT OR REPLACE INTO shelf_truth VALUES (?, ?, ?, ?)",
        ("book-3", "shelf-09C", 1, None),   # disagrees: digital says checked_out, shelf says present
    )

print("Added book-1, book-2, book-3 and robot-1.")