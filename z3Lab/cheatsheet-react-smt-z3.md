# Cheat Sheet: ReAct Architecture + Z3 SMT Basics

Quick reference for the SMT Reflection Loop lab. Not a tutorial — see `lecture1-verifiers-sat-smt.md` for the full derivation of SAT/SMT/DPLL if you need the "why."

---

## 1. ReAct loop architecture

```
context c_t  →  LLM  →  Thought (reasoning, no side effect)
                     →  Action  (one tool call, as JSON)
                          │
                          ▼
                  [ external tool / verifier ]
                          │
                          ▼
                   Observation o_t
                          │
                          ▼
     c_{t+1} = c_t + Thought_t + Action_t + Observation_t   (append, loop)
```

Formally: $c_{t+1} = (c_t, \hat a_t, o_t)$, with $\hat a_t \sim \pi_\theta(\cdot \mid c_t)$ — the model samples an action conditioned on everything so far, the environment (here: Pydantic + Z3) returns a real observation, and that observation is folded back into context for the next sample. Nothing about this loop is special to LLMs; the LLM is just the policy $\pi_\theta$ proposing actions.

**For this lab, one turn = one `AgentStep`:**

```python
class AgentStep(BaseModel):
    thought: str
    action: str                  # "verify_dispatch_smt" | "ask_user" | "finish"
    action_input: Dict[str, Any]
```

**The loop, at the shape level:**

```python
messages = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_request}]
seen_payloads = set()

for turn in range(MAX_TURNS):
    raw = call_llm(messages)
    step = parse_and_validate(raw)          # → AgentStep, or ValidationError

    if step is invalid:
        messages += [assistant(raw), user(SCHEMA_ERROR_MSG)]
        continue                             # <- Syntactic Loop

    if step.action == "finish":
        return step.action_input

    if step.action == "verify_dispatch_smt":
        key = freeze(step.action_input)
        if key in seen_payloads:
            observation = FORCE_NEW_THOUGHT_MSG   # don't even call Z3 again
        else:
            seen_payloads.add(key)
            observation = verify_dispatch_smt(step.action_input)  # <- Logical Loop

    if step.action == "ask_user":
        observation = get_user_reply(step.action_input)

    messages += [assistant(raw), user(json.dumps(observation))]

raise TimeoutError("MAX_TURNS exceeded without finish")
```

### The two-loop shape

| | Trigger | What you feed back | Loop exits when |
|---|---|---|---|
| **Syntactic loop** | `pydantic.ValidationError` on the model's own JSON | the exact validation error | model emits valid JSON matching `AgentStep` |
| **Logical loop** | Z3 returns `UNSATISFIABLE` | `diagnostics` + `feedback` from the verifier | a *different* payload returns `SATISFIABLE`, or the model calls `ask_user`/`finish` |

---

## 2. Practical cheat sheet (parsing, validation, message plumbing)

**Strip markdown fences before `json.loads`:**
```python
import re
def strip_fences(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    return text
```
Also handle prose-wrapped JSON (model says "Sure, here's the call: {...}"): grab the outermost `{...}` span with `text.find("{")` / `text.rfind("}")` before parsing, same as `code-in-class/puzzle_react.py` does.

**Validate against the step schema, and only the step schema — let the *tool* validate its own payload:**
```python
from pydantic import ValidationError

try:
    step = AgentStep.model_validate_json(strip_fences(raw))
except ValidationError as e:
    # feed e.errors() back verbatim — don't paraphrase, the model parses it fine
    ...
```

**Freezing a dict for the repeated-payload set** (dicts aren't hashable):
```python
import json
def freeze(d: dict) -> str:
    return json.dumps(d, sort_keys=True)
```

**Message roles, kept consistent:** `system` once at the top, then alternating `assistant` (the model's raw JSON turn) / `user` (the observation, JSON-encoded) — this mirrors what `puzzle_react.py` does and is the simplest thing that keeps a raw-SDK loop legible.

---

## 3. Z3 / SMT essentials

### Core objects and calls

| Call | What it does |
|---|---|
| `Solver()` | fresh, empty constraint store |
| `Int("x")`, `Real("x")`, `Bool("x")` | declare a typed variable |
| `s.add(e1, e2, ...)` | assert constraints — multiple args (or multiple `.add()` calls) are **AND**ed |
| `s.check()` | returns exactly one of `sat`, `unsat`, `unknown` (`unknown` only for undecidable theories) |
| `s.model()` | a satisfying assignment, only valid after `check()` returned `sat` |
| `m[x]` / `m.eval(x, model_completion=True)` | read one variable out of a model `m` |
| `And(...)`, `Or(...)`, `Not(...)`, `Implies(a, b)` | boolean combinators over Z3 expressions |
| `Distinct(a, b, c, ...)` | all pairwise different — cheap alternative to writing every `!=` by hand |
| `z3.If(cond, then_expr, else_expr)` | conditional *value* (not control flow) inside an arithmetic expression — used in the lab's `z3.Sum([z3.If(...)])` weight totals |
| `z3.Sum([...])` | arithmetic sum over a list of Z3 expressions |
| `Optimize()` | like `Solver()` but adds `.maximize(expr)` / `.minimize(expr)` — for "best," not just "exists" |

### Reading a result

```python
s = Solver()
s.add(x > 0, x < 10, y == x + 1)
result = s.check()
if result == sat:
    m = s.model()
    print(m[x], m[y])
elif result == unsat:
    print("no assignment satisfies all constraints")
```

`sat` means a witness exists — the model is one example, not necessarily unique. `unsat` is a *proof* of impossibility, not "the solver gave up" (that's what `unknown` is for, and it doesn't happen for the linear-arithmetic theories this lab uses).

### Example 1 — minimal feasibility check (Booleans + Ints together)

```python
from z3 import Solver, Int, Bool, sat

s = Solver()
distance, stock, surge = Int('distance'), Int('stock'), Int('surge')
courier_available = Bool('courier_available')

s.add(stock > 0, distance <= 15, courier_available, surge <= 20)   # the rules
s.add(distance == 14, stock == 3, courier_available == True, surge == 12)  # candidate

print(s.check())   # sat -> this candidate satisfies every rule
```

### Example 2 — one-package, two-AGV dispatch (the lab's shape, minimized)

```python
from z3 import Solver, Int, If, Sum, sat

s = Solver()
assign = Int("assign_pkg1")          # which AGV index carries pkg1
s.add(assign >= 0, assign < 2)       # bounds: must pick AGV 0 or AGV 1

capacities = [30, 20]                # AGV-A, AGV-B
weight = 25

for idx, cap in enumerate(capacities):
    load = If(assign == idx, weight, 0)
    s.add(load <= cap)

print(s.check())     # sat: pkg1 fits AGV-A (idx 0), not AGV-B (idx 1)
if s.check() == sat:
    print(s.model()[assign])   # -> 0
```

Change `weight = 25` to `weight = 35` and re-run: `s.check()` returns `unsat` — no AGV in this list has enough capacity, which is exactly the `oversized_packages` branch `verify_dispatch_smt` detects in the full lab.

### Common pitfalls

- **`unsat` is not an exception.** Don't wrap `s.check()` in a `try/except` expecting a crash — it's a normal return value your code branches on.
- **A `Solver()` accumulates state.** Constraints added with `.add()` stay until you call `s.push()`/`s.pop()` or build a fresh `Solver()`. Reusing one solver across unrelated candidate checks (as in `code-in-class/demo.py`'s `verify_candidate`) means building a *new* `Solver()` per call, not reusing a module-level one.
- **`m[x]` can be `None`** if `x` never appears in any asserted constraint — use `m.eval(x, model_completion=True)` when a variable might be unconstrained but you still want a default value out of it.
- **Getting a Python `int` out of a model value:** `m[x]` is a Z3 `IntNumRef`, not a Python int — call `.as_long()` (see `model[pkg_vars[p.id]].as_long()` in `verifier.py`) before using it to index a Python list.
