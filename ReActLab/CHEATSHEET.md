# JSON, Pydantic & SQLite — Cheat Sheet

Plain-language explanations + the commands you'll actually reach for.
Examples are drawn from this lab's SriRoute data, and every snippet
below has been run to confirm it works.

---

## 1. JSON

**What it is:** a text format for representing data — objects (like
Python dicts), lists, strings, numbers, booleans, `null`. It's how the
LLM in this lab sends you its "thought/action/action_input" — as a
plain string that *looks* like a Python dict but is actually just text.

**Why it matters here:** the model's output is a string. Before you can
touch `.request_id` or `.vehicle_id` on it, you have to turn that
string into a real Python object. That's `json.loads`.

| Task | Command |
|---|---|
| Python dict/list → JSON string | `json.dumps(obj)` |
| Python dict/list → JSON string, readable | `json.dumps(obj, indent=2)` |
| JSON string → Python dict/list | `json.loads(text)` |

```python
import json

person = {"name": "Meera", "age": 29, "active": True, "tags": ["vip", "south"]}

text = json.dumps(person, indent=2)   # dict -> JSON text
print(text)
# {
#   "name": "Meera",
#   "age": 29,
#   "active": true,
#   "tags": ["vip", "south"]
# }

back = json.loads(text)               # JSON text -> dict again
print(back["name"])                   # Meera
```

**Gotchas:**
- JSON `true`/`false`/`null` ↔ Python `True`/`False`/`None`. Different
  spelling, same idea.
- JSON keys are *always* double-quoted strings. `{'a': 1}` (single
  quotes) is valid Python but **not** valid JSON.
- `json.loads` on a malformed string raises `json.JSONDecodeError` —
  this is why your ReAct loop needs a try/except around parsing the
  model's raw output.

---

## 2. Pydantic

**What it is:** a library that turns "here's a shape I expect this
data to have" into a class, then checks incoming data against that
shape and tells you exactly what's wrong if it doesn't match. Where
JSON just gets you *a* Python object, Pydantic checks it's the *right
kind* of object — right fields, right types, right constraints.

**Why it matters here:** the model can say anything. Pydantic is the
gate between "the model proposed a tool call" and "we actually run
that tool call" — see `sriroute_tools.py`'s `TOOL_ARG_MODELS`.

| Task | Command |
|---|---|
| Define a schema | `class Args(BaseModel): field: type` |
| Required field | `field: str` (no default) |
| Optional field | `field: Optional[str] = None` |
| Restrict to specific values | `field: Literal["bike", "van"]` |
| Add a constraint | `field: float = Field(gt=0)` |
| Validate a dict | `Args(**some_dict)` |
| Validate a JSON string directly | `Args.model_validate_json(json_str)` |
| Convert back to a dict | `instance.model_dump()` |
| Catch a bad call | `except ValidationError as e:` |

```python
from pydantic import BaseModel, Field, ValidationError

class AssignCourierArgs(BaseModel):
    request_id: str
    vehicle_id: str

class RequestSurgeApprovalArgs(BaseModel):
    request_id: str
    extra_cost: float = Field(gt=0)   # must be > 0
    reason: str

# Valid call
ok = AssignCourierArgs(request_id="REQ-901", vehicle_id="VAN-04")
print(ok.model_dump())
# {'request_id': 'REQ-901', 'vehicle_id': 'VAN-04'}

# Missing a required field -> caught, not a crash
try:
    AssignCourierArgs(request_id="REQ-901")
except ValidationError as e:
    print(e.errors()[0]["msg"])   # "Field required"

# Constraint violated -> caught
try:
    RequestSurgeApprovalArgs(request_id="REQ-904", extra_cost=-5, reason="test")
except ValidationError as e:
    print(e.errors()[0]["msg"])   # "Input should be greater than 0"

# Straight from a JSON string (this is the ReAct-loop use case)
raw = '{"request_id": "REQ-901", "vehicle_id": "VAN-04"}'
parsed = AssignCourierArgs.model_validate_json(raw)
```

**Mental model:** `json.loads` answers "is this valid JSON?" Pydantic
answers "is this JSON the shape *this specific tool* needs?" Two
different questions — that's the exit-question distinction in
`LAB_HANDOUT.md`.

---

## 3. SQLite

**What it is:** a full SQL database that lives in a single file (no
server to start, nothing to install beyond Python's built-in `sqlite3`
module). Good for demos, small apps, and exactly this kind of "fake
backend data" use case — see `fake_db.py` in this folder for a full
working example built on the SriRoute data.

**Python's connection pattern**, always the same four steps:

```python
import sqlite3

conn = sqlite3.connect("my.db")   # 1. connect (creates the file if missing)
cur = conn.cursor()               # 2. get a cursor to run commands
cur.execute("...")                # 3. run SQL
conn.commit()                     # 4. save changes (needed after INSERT/UPDATE/DELETE)
conn.close()                      # always close when done
```

| Task | SQL / Python |
|---|---|
| Create a table | `CREATE TABLE t (id TEXT PRIMARY KEY, name TEXT NOT NULL)` |
| Insert one row | `cur.execute("INSERT INTO t VALUES (?, ?)", (id_val, name_val))` |
| Insert many rows | `cur.executemany("INSERT INTO t VALUES (?, ?)", list_of_tuples)` |
| Read rows | `cur.execute("SELECT * FROM t WHERE name = ?", (name,))` then `cur.fetchall()` |
| Update rows | `cur.execute("UPDATE t SET name = ? WHERE id = ?", (new, id_val))` |
| Delete rows | `cur.execute("DELETE FROM t WHERE id = ?", (id_val,))` |
| Join two tables | `SELECT ... FROM a JOIN b ON a.key = b.key` |
| Get columns by name | `conn.row_factory = sqlite3.Row` before making the cursor |

```python
import sqlite3

conn = sqlite3.connect("sriroute_fake.db")
conn.row_factory = sqlite3.Row     # so rows behave like dicts
cur = conn.cursor()

# WHERE clause, with a placeholder (never use an f-string here — SQL injection)
cur.execute("SELECT * FROM delivery_requests WHERE package_type = ?", ("perishable",))
for row in cur.fetchall():
    print(dict(row))

conn.close()
```

**Gotchas:**
- **Always use `?` placeholders**, never f-strings / `.format()` to
  build a query with user or model-supplied values — that's how SQL
  injection happens.
- `conn.commit()` is required after any `INSERT`/`UPDATE`/`DELETE`, or
  your changes vanish when the connection closes. `SELECT` alone
  doesn't need it.
- SQLite has no real boolean type — `fake_db.py` stores `is_priority`
  as `0`/`1` (an `INTEGER`).
- The file itself (e.g. `sriroute_fake.db`) is not human-readable —
  open it with the `sqlite3` Python module, a DB browser GUI, or the
  `sqlite3` CLI (`sqlite3 file.db ".tables"`), never a text editor.

---

## How the three fit together in this lab

1. The LLM replies with **JSON** text (a thought/action/action_input).
2. You `json.loads` it into a Python dict.
3. You validate that dict against a **Pydantic** model for the chosen
   tool — this is the "is this call even sensible to attempt" gate.
4. Only if it passes do you call the tool, which — in the SQLite-backed
   version described below — reads/writes a real database instead of
   an in-memory dict.

---

## 4. The DB-backed variant: how these files connect

This folder now holds a standalone SQLite-backed demo, kept alongside
`LAB_HANDOUT.md` and `sriroute_tools.py` purely as reference for the
ReAct/Pydantic ideas it's built from. The other lab files (the loop you'd
implement and its grading tests) have been removed — this isn't a
submission for that lab, just a self-contained exercise in wiring JSON →
Pydantic → SQL together.

| File | Role |
|---|---|
| `LAB_HANDOUT.md` | Reference only — the original assignment this demo borrows its shape and data from. |
| `sriroute_tools.py` | Reference only — the in-memory version of the four tools. Defines `TOOL_ARG_MODELS`, the Pydantic schemas `sriroute_tools_db.py` reuses. |
| `fake_db.py` | Builds `sriroute_fake.db` (SQLite file): `delivery_requests`, `fleet`, and `assignments` tables. Run directly for a standalone CRUD demo. |
| `sriroute_tools_db.py` | Same four tools, same function signatures as `sriroute_tools.py`, but each one reads/writes `sriroute_fake.db` instead of a dict. Imports `TOOL_ARG_MODELS` from `sriroute_tools.py` rather than redefining it — one schema, two backends. |
| `run_db_demo.py` | Self-contained loop: plays back a fixed list of JSON tool-call steps (standing in for what an LLM would emit), validates each with Pydantic, then calls `sriroute_tools_db.make_tool_registry()`. Run it to watch a script assign `VAN-04` to `REQ-901` — then read the `assignments` table it just wrote to disk. |

**The one real behavior difference:** `sriroute_tools.assign_courier`
returns a success message and then forgets — nothing persists once the
process exits. `sriroute_tools_db.assign_courier` does the same
(deliberately un-validated) assignment, but also `INSERT`s a row into
`assignments`. Re-run `run_db_demo.py` and you'll see that table grow —
that's the practical difference between "state in a Python dict" and
"state in a database."

```python
# Swap the backend in one line -- everything else about the loop
# (parsing, Pydantic validation, the message list) stays identical.
from sriroute_tools import make_tool_registry       # in-memory
from sriroute_tools_db import make_tool_registry    # SQLite-backed
```
