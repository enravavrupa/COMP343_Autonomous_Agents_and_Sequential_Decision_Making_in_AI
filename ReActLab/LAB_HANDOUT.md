# Lab: The SriRoute Dispatch Agent (ReAct + Structured Tool Calls)

**Format:** in-class, four phases, ~100-110 minutes (same structure as the
other labs this term).

**The driving question for this lab:** Given a natural-language dispatch
request, can your ReAct loop correctly decide whether to assign a courier
immediately, escalate for human approval, or report the request as
infeasible — using only the tool observations, without hardcoding any
scenario-specific logic into the loop itself?

## Setup

Your agent dispatches deliveries for SriRoute. It has four tools:

- `get_delivery_request(request_id)`
- `check_fleet_availability(zone, vehicle_type=None)`
- `assign_courier(request_id, vehicle_id)`
- `request_surge_approval(request_id, extra_cost, reason)`

At each turn, the model outputs a JSON object naming a thought, an
action (a tool name or `"finish"`), and the arguments for that action.
Your loop parses that JSON, **validates the arguments against a Pydantic
model before calling anything**, executes the tool, and feeds the result
back as the next observation — repeating until the model calls `finish`.

Read `sriroute_tools.py` before writing any loop code. It's fully
implemented and you shouldn't need to modify it, but note the comment on
`assign_courier`: it does not check whether the vehicle is a sensible
choice. That's relevant to the exit question.

## Phase 1 (20 min): Instructor opening

Recap: ReAct's thought/action/observation cycle; why unconstrained
free-text tool calls are fragile; how a schema (Pydantic model) turns
"the model said something plausible" into "the model's proposed call is
either valid or it isn't," with a machine-checkable boundary between
the two.

## Phase 2 (50 min): Single-question experiment, two checkpoints

**Checkpoint 1 — get the happy path working.**
Implement the five TODOs in `react_loop_lab.py`. Test against the first
scenario only:
```python
from scripted_llm import happy_path_script
run_react_loop(happy_path_script(), "Fulfill pending order REQ-901.")
```
You're done with this checkpoint when it returns a final answer
mentioning `VAN-04` without raising an exception.

**Checkpoint 2 — handle the two non-happy outcomes.**
Run the full suite:
```
python3 test_correctness.py
```
This checks: (a) constraint-gated branching — REQ-902 is a perishable
item and the only available vehicle in its zone isn't refrigerated, so
the correct behavior is to report the request as infeasible, not assign
the wrong vehicle anyway; and (b) the human-in-the-loop surge-approval
flow, both the approved and denied branches, for REQ-904 (a request
whose window only a costlier priority vehicle can meet). Note that
`request_surge_approval` is *just another tool* from your loop's point
of view — it happens to consult a human internally, but your loop code
doesn't need to know that or treat it specially.

## Phase 3 (10 min): Break-it

`test_correctness.py` also runs a third scenario
(`malformed_then_recover_script`) where the scripted model sends a
`assign_courier` call that's missing the required `vehicle_id` field.
**Before running the test**, read `scripted_llm.malformed_then_recover_script`
and predict: what should happen when your loop receives that malformed
call? Then run it and check whether your loop:
- caught the `pydantic.ValidationError` instead of crashing,
- fed the error back as an observation instead of silently skipping the
  call or calling the tool with missing arguments anyway, and
- allowed the (scripted) model to see that error and self-correct on
  its next turn.

If your loop crashes on this scenario, that's the bug to fix here — not
a sign the scenario is broken.

## Phase 4 (10 min): Exit question

`assign_courier` will "succeed" even if you ask it to assign a vehicle
whose capacity, zone, or refrigeration doesn't actually match the
request — it does no validation of its own. Given that, in one
paragraph: what specifically stops today's agent from assigning the OMR
bike to REQ-902 (the perishable item) anyway, and what's the difference
between *that* safeguard and the Pydantic validation you implemented
today? (They're solving two different problems — name both.)

Answer: After looking at the sriroute_tools.py file, it is the model itself that keeps the OMR bike from being assigned to REQ-902. While the Pydantic validation that we implemented today only checks the types of the objects being passed into them: it only ensures that both of its inputs are strings. The first is checking if it makes sense to assign a perishable item to the OMR bike but the second one just validates if the types of the validation is correct (checking if the logic is sound versus validation of the type).

## Optional: running against a real model

`llm_client.py` wraps a live call to an Anthropic model instead of a
scripted script, using the same `.step(messages)` interface — swap it
in for `happy_path_script()` etc. and watch your loop handle a model
that isn't following a fixed script. Not required, not graded, but
worth 10 minutes if you finish early: a real model's phrasing of its
JSON will vary in ways the scripted tests never exercise.

## Grading

- Checkpoint 1 (happy path) and Checkpoint 2 (constraint-gated +
  surge approval, both branches) pass
- Break-it: loop recovers from the malformed tool call without crashing
- Exit-question paragraph, distinguishing schema validation from
  business-logic validation
- W&B run logged for at least one full transcript (same requirement as
  the other labs this term)
