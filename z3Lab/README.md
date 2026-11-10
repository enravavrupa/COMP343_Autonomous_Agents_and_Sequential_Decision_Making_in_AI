# Lab Exercise: Neuro-Symbolic Reasoning — The SMT Reflection Loop

**In one sentence:** you'll build a small program where an AI turns a plain-English request into a structured plan, a strict checker either approves it or explains exactly why not, and the AI has to keep trying — proving each new attempt actually works — until it succeeds, needs your input, or has to honestly admit it can't be done.

## 0. Files in this folder

| File | What it is | Do you edit it? |
| --- | --- | --- |
| `README.md` | This document — the full spec, tasks, and test cases. | No |
| `verifier.py` | The Z3 SMT verifier (Section 4). Ships with one deliberately broken constraint (Task 0, a warm-up) on top of the expected Task 4 stub — `battery_drain_constraint`. | **The planted Task 0 bug, and the Task 4 stub — see Section 5.** Everything else: no. |
| `agent_starter.py` | Your starting point for Tasks 1–3. Contains the schema, config, test-case constants, and dedup-tracking helper already filled in; the loop logic is `TODO`/`NotImplementedError`. | **Rename to `agent.py` and fill in the TODOs — this is the assignment.** |
| `cheatsheet-react-smt-z3.md` | Quick reference: ReAct loop shape, JSON/plumbing patterns, and Z3 essentials with two runnable examples. | No |
| `env-setup.md` | How to put your `OPENAI_API_KEY` in a local `.env` file and load it in `agent.py`, instead of hardcoding it or exporting it every shell session. | No |
| `.env.example` | Template for your own `.env` (see `env-setup.md`). Contains no real key. | Copy to `.env`, then edit `.env` (never `.env.example`) with your real key. |
| `requirements.txt` | Pinned dependencies (`z3-solver`, `pydantic`, `openai`, `python-dotenv`). | No |

Setup:
```bash
pip install -r requirements.txt
cp .env.example .env               # then edit .env with your real OPENAI_API_KEY -- see env-setup.md
python verifier.py                 # Task 0: fix the planted bug before anything else
cp agent_starter.py agent.py       # then implement the TODOs in agent.py
python agent.py sat
python agent.py unsat
python agent.py drain              # Test Case 3 -- needs Task 4 done first
```

`python verifier.py` on its own runs a quick sanity check against the sat/unsat test-case payloads. **As shipped, its output will not match what Section 4 and Section 6 describe** — that's Task 0 (Section 5), a bug deliberately planted in one of the three base constraints. Find and fix it before you touch `agent.py` at all; once fixed, this command is your quickest way to see the raw SAT/UNSAT shape the verifier hands back. (`battery_drain_constraint` is a separate, harmless no-op until you implement it for Task 4 — it only affects Test Case 3, and is unrelated to Task 0.)

---

## 1. What You're Actually Building

Picture a warehouse floor. Someone types this sentence:

> *"Send Heavy-Box-1 (40kg) and Heavy-Box-2 (25kg) out on AGV-1 and AGV-2."*

("AGV" = **A**utomated **G**uided **V**ehicle — this lab's term for a warehouse robot that carries packages around on its own. `AGV-1` and `AGV-2` are just the names of two such robots, the same way you'd write "Robot 1" and "Robot 2".)

That sentence is ambiguous and might not even be physically possible — maybe one robot's battery is too low, maybe a box is heavier than any robot can carry. A human dispatcher would read it, do some quick mental math, and either confirm a plan or push back with "that won't work, here's why." You're building the software version of that dispatcher, made of two very different parts working together:

1. **An AI that reads the sentence** and turns it into a precise, structured plan (which box goes on which robot). AI is good at understanding messy human language, but bad at *guaranteeing* a plan is actually valid — it can miscount, or overlook a rule.
2. **A strict checker (a Z3 SMT solver)** that takes the AI's plan and either proves "yes, this satisfies every rule" or proves "no, and here is *exactly* which rule breaks and why." The checker is the opposite of the AI: no understanding of English at all, but it never guesses — its yes/no answers are mathematically certain.

### Framing this as a word problem, before any rules

Strip away the AI and the English sentence for a second. Underneath, this is a small constraint problem — the same shape you'd see in an operations-research textbook, except there's **no objective to optimize**, only a yes/no question: *does at least one valid assignment exist?* (No "cheapest," no "fastest" — just feasible or not.)

**What you're given (the data):**
- A list of **packages**, each with a `weight`.
- A list of **robots**, each with a `capacity` (max kg it can carry) and a `battery_pct` (its current charge), plus a `min_battery` threshold it needs to meet to be usable at all.

**What you're deciding (the variables):**
- For every package, exactly one robot it's assigned to.

**The constraints:**

1. **Assignment:** every package is assigned to exactly one robot. None left over, none split across two robots.
2. **Battery — a hard on/off gate, not a sliding scale:** a robot is either *usable* (`battery_pct >= min_battery`) or *not usable at all* (`battery_pct < min_battery`). There is no partial credit. A robot at 24% with a 25% threshold can't carry even a 1kg package — it's out of the running entirely, same as if it didn't exist for this request.
3. **Capacity — only applies to robots that passed the battery gate:** for each *usable* robot, add up the weight of every package assigned to it; that sum can't exceed its capacity.

**So what's the relationship between weight and battery? There isn't one.** They're two independent checks, not two ends of a shared budget:
- Battery only answers: *"is this robot allowed to carry anything at all?"* — yes or no.
- Weight only answers: *"how much can a robot that's already allowed to carry things, carry?"*

A robot's battery level never shrinks, scales, or otherwise touches its stated capacity number. It's a gate, not a resource. (If you were expecting something like "low battery = reduced carrying capacity," that's a reasonable real-world guess — but it's not how this lab's checker works, and getting that distinction right matters for Task 1's prompt.)

Concretely, in `AGV-1 (Capacity 30kg, Battery 90%)`: 30kg feeds constraint 3, 90% feeds constraint 2 (checked against whatever `min_battery` the request states) — and those two numbers never interact with each other. (This is the model behind Test Cases 1 and 2, and Tasks 1–3 below. Task 4 changes exactly this — see "Extending the model" further down.)

### The same thing, written the way a solver sees it

Everything above, restated as a formal constraint problem (the same shape as a linear program — decision variables, data, linear constraints — except with no objective to optimize, since this is a feasibility question, not a "best plan" search):

* **Data:** packages $p \in P$, each with a weight $w_p$. AGVs $a \in A$, each with a capacity $c_a$, a battery level $b_a$, and a minimum battery $m_a$.
* **Decision variable:** for every package $p$, which AGV it's assigned to — call it $\text{assign}(p) \in \{0, ..., |A|-1\}$.
* **Constraints:**
  * *Assignment* — every package gets exactly one AGV: true by construction, since $\text{assign}(p)$ is a single value, not a set.
  * *Battery gate* — for every AGV $a$ where $b_a < m_a$: $\text{assign}(p) \ne a$ for every package $p$ (that AGV is excluded outright).
  * *Capacity* — for every AGV $a$: $\displaystyle\sum_{p \,:\, \text{assign}(p)=a} w_p \;\le\; c_a$.

That's the entire model behind `verifier.py`'s three constraint blocks (Section 4) — three linear rules, nothing hidden.

### Extending the model: battery drains with weight (Task 4)

Real robots don't just have a battery that's either "enough" or "not enough" regardless of load — carrying more weight costs more battery. Task 4 asks you to add exactly that: each AGV gets a `drain_rate` $d_a \ge 0$ — the battery cost, in percentage points, of carrying **one kg**.

In words: *an AGV can only be given an assigned load if, after paying that load's drain cost, its battery is still at or above its own minimum threshold.*

That's a genuine relationship between weight and battery now (unlike the base model above, where there deliberately wasn't one) — and it's still just one more linear inequality, in the same $\sum_{p:\text{assign}(p)=a} w_p$ shape as the capacity constraint. Section 5, Task 4 has the specifics; the exercise is to write that inequality yourself and assert it in `verifier.py`'s one TODO (`battery_drain_constraint`) — the three constraints above are your worked template for exactly how to go from "a rule in English" to "a line Z3 can check."

Your job is the loop that connects them: hand the AI's plan to the checker, and if the checker says no, feed its reason back to the AI and make it propose a **genuinely different** plan — not repeat the same one, not just apologize — and check that new plan again. Repeat until the plan works, until the AI needs to ask you a judgment call it can't make alone (e.g. "drop this box, or wait 20 minutes for the robot to charge?"), or until it has to honestly tell the user the request is impossible.

Robots and packages are just this lab's example. The same shape — *AI proposes, a hard checker verifies, failures get explained and retried instead of ignored* — shows up any time you want an AI's output to be trustworthy rather than merely plausible-sounding.

### Why an UNSAT (rejected) answer is not a bug

Most exercises treat a tool's error as something to catch and route around. Here, when the checker says "no, this doesn't work," that rejection is the interesting part of the exercise: it comes with a precise mathematical reason (which robot is too low on battery, which package is too heavy for every available robot, etc.), and your loop's job is to make the AI actually *use* that reason to come up with something better — not just try the same thing again, and not just give up.

### The technical picture

The same flow, in the vocabulary the rest of this document uses (ReAct loop, Pydantic schema, SAT/UNSAT):

```
[ Unstructured Natural Language Request ]
                   │
                   ▼
       [ LLM Spec Parser (ReAct) ]
                   │
                   ▼
  [ Pydantic Schema Guard (Type Check) ]  ──(Syntactic Fail)──► [ Retry Parsing ]
                   │
            (Syntactic Pass)
                   │
                   ▼
     [ Z3 SMT Verifier (Physics/Logic) ] ◄─────────────────────────┐
                   │                                                │
        ┌──────────┴───────────┐                                   │
   (Logical SAT)          (Logical UNSAT)                          │
        │                       │                                  │
        ▼                       ▼                                  │
[ Execute & Finalize ]   [ Reflection Turn: read diagnostics,       │
                           relax constraints or ask_user ] ─────────┘
                                    │
                                    ▼  (no relaxation possible)
                         [ Explain impossibility to user ]
```

Note the loop-back arrow: a relaxed proposal is not just narrated to the user — it is **re-submitted to Z3** for a fresh SAT/UNSAT verdict. Explaining *why* something is impossible is only half the exercise; the other half is proving that a *relaxed* plan actually works before you present it.

### The division of labor

* **LLM (the spec writer):** Translates messy, ambiguous human language into a strict parameter schema. It does *not* attempt to solve NP-complete mathematical allocations in-context.
* **SMT solver (the ground truth):** Acts as an unyielding referee enforcing physical limits ($A \cdot x \le b$, battery thresholds, weight capacities). It provides a mathematical proof of feasibility ($SAT$) or impossibility ($UNSAT$).

---

## 2. What You'll Be Able to Do Afterwards

By completing this exercise, you will be able to:

1. **Tell two kinds of failure apart:** "the AI's output was formatted wrong" (fix: ask it to redo the formatting) versus "the request itself is impossible" (fix: change the actual plan, not the formatting). In this document these are called *Syntactic Failures* and *Logical Impossibility*.
2. **Build the back-and-forth loop yourself, in plain Python** — raw API calls, your own state tracking, and Pydantic validation, with no agent framework (no LangChain / LlamaIndex) doing the orchestration for you.
3. **Write error messages that are actually useful to the AI**, not just "no" — structuring the checker's output so the AI receives the exact reason a plan failed, and can act on it.
4. **Never take the AI's word for it:** show that a "fixed" plan is only accepted after it's been proven to work by the checker again — a claim of success from the AI alone is not enough.

---

## 3. Architecture: The Two Error Loops (Plus One Negotiation Sub-Loop)

**In plain terms:** there are two different ways a turn can "fail," and your loop must react differently to each:

- **The AI wrote broken JSON** (typo, missing field, wrong type). This has nothing to do with robots or packages — it's just malformed output. Fix: tell it exactly what's wrong with the formatting and let it retry.
- **The AI wrote valid JSON, but the plan itself doesn't work** (a robot's battery is too low, a box is too heavy, etc.). Fix: this is not a formatting problem, so re-prompting for "valid JSON" won't help — you have to show the AI the mathematical reason and make it propose an actually different plan.
- **Sometimes there are two valid ways to fix the plan and only a human should pick** (e.g. "drop this box" vs. "wait 20 minutes for the robot to charge"). Fix: the AI pauses and asks the user, instead of guessing.

The table below names these precisely — the same three ideas, with the exact terms (`ValidationError`, `UNSAT`, `ask_user`) your code needs to check for:

Your agent must handle two distinct feedback loops, and the second one can iterate:

| Loop Type | Cause | Handler | Agent Action |
| --- | --- | --- | --- |
| **Syntactic Loop** | LLM outputs bad JSON, missing keys, or invalid types. | `Pydantic ValidationError` | Intercept error, append schema rule to context, and re-prompt for valid JSON formatting. |
| **Logical Loop** | LLM outputs valid parameters, but $Z3$ proves no physical solution exists. | `SMT UNSAT` Status | Intercept $UNSAT$ reason, append diagnostic context, force LLM into a `Thought` turn to reflect and propose a **relaxed** payload. |
| **Logical Loop → Negotiation Sub-Loop** | A relaxed payload is still not enough, or two different relaxations are both defensible (e.g., drop package A vs. wait for AGV-2 to charge) and only the user can decide. | `ask_user` action | Agent pauses, states the trade-off in plain language, and waits for a user decision before re-verifying. |

The Logical Loop is **not** "one UNSAT, then explain and stop." It is "UNSAT → reflect → propose a concretely different payload → re-verify → repeat until SAT, `ask_user`, or the turn budget runs out."

---

## 4. Provided Scaffolding: $Z3$ Verifier (`verifier.py`)

**In plain terms:** this is the checker — you don't write it, you just call it (`verify_dispatch_smt(payload)`) and read what it returns. Give it a proposed plan (which packages, which robots, how heavy, how charged) and it replies with one of two things:

- `"SATISFIABLE"` — the plan works, here's the exact assignment.
- `"UNSATISFIABLE"` — the plan doesn't work, and *why* it doesn't (a robot is too low on battery and can't be used at all; a package is too heavy for every available robot no matter how you rearrange things; the numbers technically add up but no valid arrangement exists; or, once Task 4 is implemented, a robot's battery gets drained below threshold by the load itself). Your loop's job is to pass that reason back to the AI, not just the word "no."

Once Task 0 is fixed, you won't need to re-derive this Z3 syntax line by line for the rest of the assignment — you're only editing one further marked stub after that (`battery_drain_constraint`, Task 4). What matters for Tasks 1–4 is the shape of what goes in (`packages`, `agvs`) and what comes out (`status`, `assignments` or `diagnostics` + `feedback`). Task 0 itself (Section 5) is the one place this lab asks you to read the listing below closely enough to catch a planted mistake — the block quoted next is the actual, current contents of `verify_dispatch_smt`, bug included.

```python
# verifier.py
import z3
from pydantic import BaseModel, Field
from typing import List, Dict, Any

class PackageSpec(BaseModel):
    id: str
    weight: int = Field(gt=0)

class AGVSpec(BaseModel):
    id: str
    capacity: int = Field(gt=0)
    battery_pct: int = Field(ge=0, le=100)
    min_battery: int = Field(default=20)
    drain_rate: float = Field(default=0.0, ge=0)   # % battery per kg carried; 0 = no drain modeled

class VerificationPayload(BaseModel):
    packages: List[PackageSpec]
    agvs: List[AGVSpec]

def battery_drain_constraint(agv: AGVSpec, assigned_weight) -> "z3.BoolRef":
    """
    TODO (Task 4): return the z3 constraint for "carrying weight also costs
    battery" -- see README Section 1 ("Extending the model") for the
    relationship in words, and Section 5, Task 4 for what's required.
    `assigned_weight` is the same z3.Sum([z3.If(...)]) expression used in
    the capacity check below, already built for you.

    Until implemented, this is a no-op (always true) -- Test Cases 1/2
    don't set drain_rate, so they're unaffected. Test Case 3 depends on it.
    """
    return z3.BoolVal(True)  # TODO: replace with the real inequality (Task 4)

def verify_dispatch_smt(raw_payload: dict) -> dict:
    """
    Evaluates payload against First-Order Logic constraints using Z3.
    Returns structured mathematical feedback for the agent context.
    """
    try:
        payload = VerificationPayload.model_validate(raw_payload)
    except Exception as e:
        return {"status": "SYNTACTIC_ERROR", "message": str(e)}

    solver = z3.Solver()
    num_agvs = len(payload.agvs)
    pkg_vars = {p.id: z3.Int(f"assign_{p.id}") for p in payload.packages}

    # 1. Bounds: Every package mapped to a valid AGV index
    for p_id, var in pkg_vars.items():
        solver.add(var >= 0, var < num_agvs)

    # 2. Battery Theory Check: Low battery AGVs cannot receive assignments
    for idx, agv in enumerate(payload.agvs):
        if agv.battery_pct < agv.min_battery:
            for p_id, var in pkg_vars.items():
                solver.add(var != idx)

    # 3. Weight Theory Check: Sum of package weights <= AGV capacity
    for idx, agv in enumerate(payload.agvs):
        assigned_weight = z3.Sum([
            z3.If(pkg_vars[p.id] == idx, p.weight, 0)
            for p in payload.packages
        ])
        solver.add(assigned_weight >= agv.capacity) # wrong

    # 4. Battery-Drain Theory Check (Task 4 -- fill in battery_drain_constraint above)
    for idx, agv in enumerate(payload.agvs):
        assigned_weight = z3.Sum([
            z3.If(pkg_vars[p.id] == idx, p.weight, 0)
            for p in payload.packages
        ])
        solver.add(battery_drain_constraint(agv, assigned_weight))

    # Solve
    if solver.check() == z3.sat:
        model = solver.model()
        assignments = {
            p.id: payload.agvs[model[pkg_vars[p.id]].as_long()].id
            for p in payload.packages
        }
        return {
            "status": "SATISFIABLE",
            "assignments": assignments,
            "feedback": "Allocation verified mathematically."
        }

    # ---- UNSAT: figure out WHICH of three distinct failures actually applies ----
    active_agvs = [a for a in payload.agvs if a.battery_pct >= a.min_battery]
    disabled_agvs = [a for a in payload.agvs if a.battery_pct < a.min_battery]
    total_weight = sum(p.weight for p in payload.packages)
    total_active_capacity = sum(a.capacity for a in active_agvs)
    max_active_capacity = max((a.capacity for a in active_agvs), default=0)
    oversized_packages = [p.id for p in payload.packages if p.weight > max_active_capacity]

    diagnostics = {
        "total_requested_weight": total_weight,
        "available_active_capacity": total_active_capacity,
        "active_agvs": [a.id for a in active_agvs],
        "disabled_agvs_low_battery": [a.id for a in disabled_agvs],
        "oversized_packages": oversized_packages,
    }

    if not active_agvs:
        feedback = (
            "UNSAT: every AGV is below its minimum battery threshold. "
            "No assignment is possible until at least one AGV is charged."
        )
    elif oversized_packages:
        feedback = (
            f"UNSAT: package(s) {oversized_packages} individually exceed the "
            f"capacity of every active AGV (largest active capacity: "
            f"{max_active_capacity}kg). This cannot be fixed by reassigning "
            f"packages alone — split the package, free up a bigger AGV, or drop it."
        )
    elif total_weight > total_active_capacity:
        feedback = (
            f"UNSAT: total requested weight ({total_weight}kg) exceeds usable "
            f"active capacity ({total_active_capacity}kg). Check disabled "
            f"low-battery AGVs or reduce the package set."
        )
    else:
        feedback = (
            f"UNSAT: no valid assignment exists even though total weight "
            f"({total_weight}kg) fits within total active capacity "
            f"({total_active_capacity}kg). This is a packing/arrangement "
            f"conflict, not a raw capacity shortfall — try a different split "
            f"of which packages travel together."
        )
        if any(a.drain_rate > 0 for a in active_agvs):
            feedback += (
                " Note: one or more active AGVs have a nonzero drain_rate, "
                "meaning the weight they're given also costs them battery "
                "headroom above their own min_battery -- that interaction "
                "may be part of why no split of these packages works, even "
                "though the raw capacity numbers look sufficient."
            )

    return {"status": "UNSATISFIABLE", "diagnostics": diagnostics, "feedback": feedback}
```

> **Why this matters:** the earlier draft of this verifier always blamed UNSAT on "total weight exceeds total capacity," even when that number was mathematically false for the case at hand (e.g., a single 35kg package with two 20kg-capacity AGVs — the *sum* fits, but no *single* AGV can carry it). Feeding the LLM an inaccurate cause produces a confidently wrong reflection. Always double-check that your diagnostic text is entailed by the diagnostic numbers. (This is exactly why the note above only *flags* drain_rate as a possible factor rather than claiming it's the cause outright — with weight and battery now interacting, a precise closed-form diagnosis is a harder claim to make honestly. Naming a plausible cause without overclaiming is a legitimate, honest diagnostic too.)

---

## 5. Student Implementation Tasks

### Task 0: Fix the Planted Constraint Bug (Warm-Up)

**In plain terms:** before you write a line of your own code, read `verify_dispatch_smt` in `verifier.py` top to bottom. As shipped, one of its three base constraints (assignment bounds, the battery gate, or the capacity check) does not actually enforce the rule its own comment right above it describes. Find it, fix it, and prove it's actually fixed — this is your only required edit to `verifier.py` besides Task 4's stub.

1. Run `python verifier.py`. It prints the verifier's raw output for the sat/unsat payloads defined at the bottom of the file. Compare that output against what Section 4 and Section 6 say it should be (the SAT case should return `"status": "SATISFIABLE"` with an assignment; the UNSAT case should return `"status": "UNSATISFIABLE"` with `oversized_packages` naming `Heavy-Box-1`).
2. If something doesn't match, the bug is in one of the three `solver.add(...)` blocks *above* the Task 4 stub — not in `battery_drain_constraint` itself, and not in the diagnostic/feedback code below the `solver.check()` call. Re-derive each constraint from Section 1 ("The same thing, written the way a solver sees it") and compare it, symbol by symbol, against what's actually in the code.
3. **A matching top-level `status` on one test case is not proof you found the real bug.** A wrong constraint can still land on the "right" SATISFIABLE/UNSATISFIABLE verdict for the wrong reason — e.g. because a *different* constraint happens to force the same outcome anyway. Don't stop until every constraint block matches both its own comment and the formal model in Section 1, and both the SAT and UNSAT cases in `python verifier.py`'s output match Section 6's Required Behavior Trace in full (including `assignments` / `diagnostics`, not just `status`).
4. Fix the bug directly in `verifier.py`.
5. **Write a short reflection note** (`task0_reflection.md`, 100–200 words, included in your submission) answering:
   - Which constraint was wrong, and what was the one-symbol (or one-operator) mistake?
   - What did the *symptom* look like before you found the real cause — did the top-level `status` on either test case actually change, or did it stay superficially "correct" for a misleading reason? (See point 3 above.)
   - In your own words: why isn't "the test's pass/fail matches" the same claim as "the code is correct," and where else in this lab — or in software generally — does that distinction matter?

This is intentionally a small, self-contained exercise in reading someone else's Z3 constraints closely enough to catch a wrong operator — a skill Task 4 asks you to use productively a few minutes later, when you write one of these constraints yourself.

### Task 1: Design the ReAct System Prompt & Protocol

**In plain terms:** write the instructions you give the AI — the exact rules it must follow when it talks to the checker, what it's allowed to do, and specifically what it must do when the checker says no.

Construct a system prompt that mandates strict single-JSON output formatting. Your protocol must explicitly support **three** tool states:

1. `verify_dispatch_smt` — calls the SMT verifier with structured arguments.
2. `ask_user` — pauses the loop to ask the human a clarifying question, used when a relaxation requires a judgment call the agent cannot make on its own (e.g., "drop Heavy-Box-1, or wait ~20 minutes for AGV-2 to charge?").
3. `finish` — returns a natural language answer to the user once verified (or once the user has answered an `ask_user` question and a follow-up plan is confirmed SAT).

> **Key Rule 1:** *"If an observation returns `UNSATISFIABLE`, you are forbidden from repeating the exact same `action_input`. You must analyze the diagnostic feedback, enter a new `Thought`, and either (a) propose a concretely different, relaxed payload and call `verify_dispatch_smt` again, or (b) call `ask_user` if the correct relaxation depends on a preference only the user can supply."*
>
> **Key Rule 2:** *"Never call `finish` with a claim of feasibility unless the most recent `verify_dispatch_smt` observation for that exact plan was `SATISFIABLE`. If a plan is impossible, `finish` must state that plainly and reference the diagnostic reason — do not soften an UNSAT into a vague apology."*

### Task 2: Build the Manual Execution Loop (`agent.py`)

**In plain terms:** write the actual Python loop — send the AI your instructions and the user's request, read what it replies, call the checker with what it proposed, feed the result back, and repeat. Also build in the safety rails: a hard limit on how many times this can loop, and a guard that stops the AI from resubmitting the exact same rejected plan.

Write the execution loop using raw API calls (e.g., `openai.OpenAI()`). Implement:

* String sanitization to strip markdown backticks (`` ```json ... ``` ``).
* Schema validation using a master Pydantic step model:
```python
class AgentStep(BaseModel):
    thought: str
    action: str
    action_input: Dict[str, Any]
```
* Context management that feeds observations back into the message history array.
* **A hard turn budget** — e.g. `MAX_TURNS = 8`. If the loop exhausts its budget without reaching `finish`, terminate with an explicit failure message rather than looping silently forever.
* **Repeated-payload detection** — do not rely on the prompt alone to stop the LLM from resubmitting an identical `action_input` to `verify_dispatch_smt`. `agent_starter.py` provides a `SeenPayloads` helper for this (backed by `freeze()`, also provided) — call `.seen(action_input)` before calling `verify_dispatch_smt`, and `.add(action_input)` right after; if the LLM resubmits a payload that already returned `UNSATISFIABLE`, short-circuit with a synthetic observation forcing a new `Thought` instead of calling Z3 again. You're wiring this helper into your loop correctly, not building the tracking logic yourself.

### Task 3: Handle the UNSAT Reflection Turn

**In plain terms:** make sure that when the checker rejects a plan, your loop actually shows the AI *why* (not just "rejected"), and doesn't let the AI stop until it lands on a plan that's been proven to work, needs to ask the user a real judgment call, or has honestly run out of options.

Ensure that when `verify_dispatch_smt` returns `UNSATISFIABLE`, your loop appends the `diagnostics` and `feedback` fields into the message stream so the model can consume the mathematical reason for failure. Then require the loop to continue until one of:

* a **relaxed** `verify_dispatch_smt` call returns `SATISFIABLE` (the plan the model finishes with must be the one Z3 actually verified — not a narrated guess), or
* the model calls `ask_user` and, after a (simulated) user reply, a subsequent verification succeeds, or
* the model determines no relaxation is possible and calls `finish` with an honest impossibility explanation grounded in the diagnostics.

### Task 4: Extend the Model — Battery Drains With Weight

**In plain terms:** so far, an AGV's battery only ever gets compared to its own threshold — the load it's carrying never factors in. Your job is to change that: derive and write the one new rule that says "carrying more weight also costs battery," the same way Section 1 walked through the three existing rules, and encode it in Z3.

Concretely:

1. Read "The same thing, written the way a solver sees it" and "Extending the model" in Section 1 — the first is a worked example of going from an English rule to a Z3 constraint for the three existing checks; the second describes, in words, the new rule you need (an AGV's `drain_rate` — provided as a new field on `AGVSpec`, already in `verifier.py` — costs battery per kg carried).
2. Derive the inequality yourself: given an AGV's `battery_pct`, `min_battery`, and `drain_rate`, and the total weight assigned to it, when is that AGV still allowed to take the load?
3. Implement it in `verifier.py`'s `battery_drain_constraint(agv, assigned_weight)` — the **only** part of `verifier.py` you're expected to edit. It currently returns `z3.BoolVal(True)` (a no-op); replace that with your derived inequality, built the same way as the existing weight-capacity check (`agv.battery_pct`, `agv.min_battery`, `agv.drain_rate`, and the `assigned_weight` expression you're handed).
4. Your system prompt (Task 1) needs to parse a stated drain rate (e.g. *"drains 1.0% battery per kg carried"*) into each AGV's `drain_rate`, the same way it already parses "minimum battery threshold" into `min_battery`.
5. Verify against Test Case 3 (Section 6) — a case specifically designed so that a plausible **first-guess assignment fails only because of this new rule**, forcing the same reflect → propose a different assignment → re-verify pattern as Test Case 2, for a different underlying reason.

> **Why the formula isn't given here:** Section 1 hands you the worked template (assignment, battery gate, capacity, all in formal inequality form) precisely so you have a pattern to follow — but the point of this task is translating a described relationship into that form yourself, once. If you're stuck, re-read "Extending the model" in Section 1 slowly; the inequality is a small, direct restatement of that paragraph.

---

## 6. Test Suite & Evaluation Cases

**In plain terms:** your program gets run three times — once with a request that's actually possible (Test Case 1), once with a request that isn't, as written, due to the base rules from Tasks 1–3 (Test Case 2), and once with a request that isn't, as written, specifically because of the Task 4 battery-drain rule (Test Case 3). Cases 2 and 3 are the important ones: each checks that your loop notices the rejection, figures out a *different* plan that works, double-checks that plan with the checker, and only then tells the user — Case 3 just makes sure that still holds once weight and battery interact.

Your implementation will be evaluated against three specific test prompts.

> **Note on the battery threshold:** all three prompts below specify a threshold **different from the schema default** (`min_battery` defaults to 20%) specifically so that a correct implementation must actually parse "minimum battery threshold" into each AGV's `min_battery` field — an implementation that silently ignores the phrase and relies on the default will produce the wrong `active_agvs` set. Test Case 3 additionally states a `drain_rate` per AGV in plain English ("drains X% battery per kg carried") that must be parsed the same way.

### Test Case 1: Feasible Request ($SAT$)

* **Input Prompt:**
> *"Allocate Item-1 (10kg) and Item-2 (15kg) across AGV-A (Capacity 30kg, Battery 80%) and AGV-B (Capacity 20kg, Battery 50%). Minimum battery threshold is 25%."*

* **Required Behavior Trace:**
1. **Turn 1:** LLM parses entities → calls `verify_dispatch_smt`.
2. **Tool Observation:** Returns `SATISFIABLE` with assignments.
3. **Turn 2:** LLM observes `SATISFIABLE` → calls `finish` presenting the assignment.

### Test Case 2: Over-Constrained Request ($UNSAT$ → Relaxation → $SAT$)

* **Input Prompt:**
> *"Allocate Heavy-Box-1 (40kg) and Heavy-Box-2 (25kg) across AGV-1 (Capacity 30kg, Battery 90%) and AGV-2 (Capacity 20kg, Battery 10%). Minimum battery threshold is 25%."*

* **Required Behavior Trace:**
1. **Turn 1:** LLM parses entities → calls `verify_dispatch_smt`.
2. **Tool Observation:** SMT evaluates AGV-2 as disabled ($10\% < 25\%$) and Heavy-Box-1 (40kg) as individually oversized for the only active AGV (AGV-1, 30kg capacity). Tool returns `UNSATISFIABLE` with the `oversized_packages` diagnostic branch.
3. **Turn 2 (The Reflection Turn):** LLM reads the diagnostics — notes AGV-2 is battery-locked out and Heavy-Box-1 cannot fit *any* active AGV, not just this one. It proposes a **relaxed** plan: dispatch only Heavy-Box-2 (25kg ≤ 30kg) on AGV-1, holding Heavy-Box-1 back. Calls `verify_dispatch_smt` again with this reduced payload.
4. **Tool Observation:** Returns `SATISFIABLE` — Heavy-Box-2 → AGV-1.
5. **Turn 3:** LLM calls `finish`, presenting the verified partial dispatch (Heavy-Box-2 only) and explaining to the user that Heavy-Box-1 cannot move under current conditions — recommending either charging AGV-2 above 25% or waiting for a larger AGV, grounded in the diagnostics from Turn 1.

This trace is the one place where the "negotiation" language in Section 1 is actually load-bearing: a correct submission proves the relaxed plan with a second Z3 call rather than just asserting it.

### Test Case 3: Battery-Drain Constraint ($UNSAT$ → Relaxation → $SAT$, Task 4)

* **Input Prompt:**
> *"Allocate Cargo-1 (22kg) and Cargo-2 (12kg) across AGV-X (Capacity 25kg, Battery 35%, drains 1.0% battery per kg carried) and AGV-Y (Capacity 25kg, Battery 90%, drains 0.3% battery per kg carried). Minimum battery threshold is 20%."*

* **Why this case is interesting:** under the base model (Tasks 1–3, no drain), both AGVs pass the plain battery gate (35% ≥ 20%, 90% ≥ 20%) and both packages individually fit either AGV's 25kg capacity — so a plausible first guess, like assigning packages to AGVs in the order they're listed (Cargo-1 → AGV-X, Cargo-2 → AGV-Y), looks completely fine by the old rules. It fails only once the Task 4 drain rule is applied: AGV-X, at 35% battery, can't absorb Cargo-1's 22kg at a 1.0%/kg drain cost and still clear its 20% threshold.

* **Required Behavior Trace:**
1. **Turn 1:** LLM parses entities (including each AGV's `drain_rate`) → calls `verify_dispatch_smt` with a plausible first assignment (e.g. Cargo-1 → AGV-X, Cargo-2 → AGV-Y).
2. **Tool Observation:** Returns `UNSATISFIABLE` — this specific assignment fails the battery-drain constraint on AGV-X even though it passes the plain battery gate and capacity check.
3. **Turn 2 (The Reflection Turn):** LLM reads the diagnostics and feedback, recognizes the drain-related note, and proposes a **relaxed** plan: swap the assignment (Cargo-1 → AGV-Y, Cargo-2 → AGV-X), keeping both packages dispatched rather than dropping one. Calls `verify_dispatch_smt` again with this genuinely different payload.
4. **Tool Observation:** Returns `SATISFIABLE` — Cargo-1 → AGV-Y, Cargo-2 → AGV-X.
5. **Turn 3:** LLM calls `finish`, presenting the verified swapped assignment.

Unlike Test Case 2, the fix here isn't "drop a package" — it's "reassign, not just relax," which only becomes visible once the battery-drain rule from Task 4 is actually implemented and active.

---

## 7. Grading Rubric (informal weighting)

**In plain terms:** most of your grade (65%) is about what happens *after* a rejection — did you show the AI the real reason, did it come up with something genuinely new, did you re-check that new plan before trusting it, and (new, Task 4) did you correctly encode the one new rule that makes weight and battery interact. The rest is basic robustness (doesn't crash on bad output, doesn't loop forever) and readable code.

| Component | Weight | What's checked |
| --- | --- | --- |
| Task 0 (planted bug fixed + reflection note) | 5% | The fixed constraint in `verifier.py` matches Section 1's formal model, not just a coincidentally-matching top-level status; `task0_reflection.md` names the actual mistake and engages with the symptom-vs-cause point above |
| Syntactic loop (retry on bad JSON) | 10% | `trace_sat.json` / `trace_unsat.json` show no silent crashes on malformed model output (inject one deliberately if your own model never produces one) |
| Logical loop (UNSAT → diagnostics → reflection) | 20% | `Thought` text in the reflection turn references the actual diagnostic fields, not a generic apology |
| Negotiation / relaxation round-trip | 25% | Test Case 2 trace contains a **second** `verify_dispatch_smt` call with a genuinely different payload, and that call returns `SATISFIABLE` before `finish` |
| Battery-drain constraint (Task 4) | 25% | Checked entirely through **behavior**, not by inspecting your Z3 expression directly: Test Case 3 (Section 6) produces the required UNSAT → reflect → reassigned payload → `SATISFIABLE` trace |
| Turn budget & repeated-payload guard usage | 10% | Loop terminates cleanly under `MAX_TURNS`; the provided `SeenPayloads` helper is actually wired in, so resubmitting an identical UNSAT payload is intercepted in code, not left to the prompt alone |
| Code quality / clarity | 5% | `agent.py` is readable, uses the provided `AgentStep` model, no framework dependencies (no LangChain/LlamaIndex) |

---

## 8. Submission Checklist

**In plain terms:** seven files, no more, no less — your finished code, the checker with Task 0's planted bug fixed and its Task 4 stub filled in, your written reflection on that bug, your pinned dependency list, and a saved log of each of the three test runs (one where the plan worked as-is, one rejected-then-fixed by dropping a package, one rejected-then-fixed by reassigning).

Submit a repository containing:

1. `agent.py` — Complete manual ReAct loop and prompt engineering code (start from `agent_starter.py`, with every `TODO`/`NotImplementedError` filled in).
2. `verifier.py` — SMT integration module, with Task 0's planted constraint bug fixed and `battery_drain_constraint` (Task 4) implemented; otherwise unmodified from the provided scaffolding.
3. `task0_reflection.md` — Your Task 0 reflection note (Section 5, Task 0, point 5): which constraint was wrong, what the symptom looked like, and why matching test status alone isn't proof of correctness.
4. `requirements.txt` — pinned versions of `z3-solver`, `pydantic`, and your LLM SDK (Z3's Python API has broken across versions before; pin it).
5. `trace_sat.json` — Saved execution trace log of Test Case 1, as a list of `{"role": ..., "content": ...}` or `{"thought": ..., "action": ..., "action_input": ..., "observation": ...}` turns (either shape is fine as long as it's consistent and complete).
6. `trace_unsat.json` — Saved execution trace log of Test Case 2, showing the initial `UNSAT` diagnostic observation, the reflection `Thought`, the **second** (relaxed) `verify_dispatch_smt` call and its `SATISFIABLE` result, and the final `finish` turn.
7. `trace_drain.json` — Saved execution trace log of Test Case 3, showing the same UNSAT → reflect → re-verify → `finish` shape as Test Case 2, but for the battery-drain rule from Task 4.

> Run your model at a low/zero temperature when generating the submitted traces — nondeterministic traces make grading (and your own debugging) harder to reproduce.
