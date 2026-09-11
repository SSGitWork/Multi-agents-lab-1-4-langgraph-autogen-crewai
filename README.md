# Lab 1.4 – Cross-Framework Comparison

**Module 1 · Section 5 · Lab 4**
Build Autonomous Multi-Agent Systems · Saras AI Institute

---

## Objective

Implement the same single-agent system in **LangGraph**, **AutoGen**, and
**CrewAI**, run all three on an identical coding task, and document observable
differences in behaviour and token usage.

By the end of this lab you will be able to articulate concrete tradeoffs
between the three frameworks, which is the foundation for the framework-selection
decisions you will make throughout the rest of the course.

---

## Prerequisites

- Lab 1.2 complete (you understand the three tools: `read_file`, `write_file`,
  `exec_python`)
- Lab 1.3 complete (you understand the ReACT loop and termination conditions)

---

## What is already provided

| File / directory | What it is |
|---|---|
| `llm_client.py` | Shared OpenAI client routed through Helicone — **do not modify** |
| `tools/code_tools.py` | The three tools from Lab 1.2, ready to import — **do not modify** |
| `tools/token_counter.py` | `UsageTracker` dataclass for recording LLM usage — **do not modify** |
| `task.py` | The shared `TASK_DESCRIPTION` string — **do not modify** |
| `compare.py` | Runner that executes all three frameworks and prints a comparison table |
| `comparison/tradeoffs.md` | Template for your qualitative observations |
| `tests/test_frameworks.py` | Behavioural test suite |

---

## Your Task

### Step 1 — Implement `agent_langgraph.py`

Open the file. You need to implement **one function**: `agent_node()`.

The docstring inside the function lists every step. The key points:

- Use `ChatOpenAI(openai_client=get_client(), model=DEFAULT_MODEL)` then call
  `.bind_tools(ALL_SCHEMAS)` on it.
- After the LLM call, read token usage from
  `response.response_metadata["token_usage"]` and pass it to `tracker.record_step()`.
- Set `done=True` in the returned state dict when the response content contains
  the string `"DONE"`.

Run it in isolation to check it works:

```bash
python agent_langgraph.py
```

---

### Step 2 — Implement `agent_autogen.py`

You need to implement **two functions** plus the `initiate_chat()` call:

**`make_assistant()`**
- Create an `AssistantAgent` and register all three tools for LLM use with
  `agent.register_for_llm(name=..., description=...)(function)`.

**`make_proxy()`**
- Create a `UserProxyAgent` with `human_input_mode="NEVER"` and register the
  same tools for execution with `proxy.register_for_execution(name=...)(function)`.

**`run()`**
- Call `proxy.initiate_chat(recipient=assistant, message=TASK_DESCRIPTION, max_turns=15)`.

Run it:

```bash
python agent_autogen.py
```

---

### Step 3 — Implement `agent_crewai.py`

You need to implement **three functions** plus the `Crew` assembly:

**`make_tools()`**
- Use the `@crewai_tool` decorator to wrap `read_file`, `write_file`, and
  `exec_python`. The docstring on each wrapper becomes the tool description.

**`make_agent(tools)`**
- Create a `CrewAI Agent` with `role`, `goal`, `backstory`, `tools`, and `llm=_make_llm()`.

**`make_task(agent)`**
- Create a `Task` with `description=TASK_DESCRIPTION` and `agent=agent`.

**`run()`**
- Create a `Crew(agents=[agent], tasks=[task])` and call `crew.kickoff()`.

Run it:

```bash
python agent_crewai.py
```

---

### Step 4 — Run the comparison

Once all three implementations work, run:

```bash
python compare.py
```

This prints a table like:

```
===========================================================================
  Framework    Status         Steps     Prompt   Completion      Total
---------------------------------------------------------------------------
  LangGraph    ok                 7      4 231          812      5 043
  AutoGen      ok                 9      6 104        1 021      7 125
  CrewAI       ok                 6      5 882          934      6 816
===========================================================================
```

The results are saved to `comparison/results.json`.

---

### Step 5 — Fill in `comparison/tradeoffs.md`

Open the file and answer every section. Your answers must be specific —
reference actual numbers from your comparison table and concrete things
you experienced while implementing each framework.

You need at least **2 of 6 sections** filled in to pass the tests, but
completing all six will prepare you for the Week 1 deliverable discussion.

---

### Step 6 — Run the test suite

```bash
pytest tests/ -v
```

All tests for a framework are automatically skipped if that framework is
not yet implemented (raises `NotImplementedError`), so you can run the
suite at any point during development.

**Passing criteria:**
- `word_freq.py` exists in the sandbox workspace and is valid Python
- `word_freq.py` contains a `word_frequency` function with type annotations
- `sample.txt` exists with at least 20 words
- `tracker.step_count >= 1` and `< 15` for each framework
- `comparison/tradeoffs.md` has at least 2 sections filled in

---

## Common Errors

| Symptom | Likely cause |
|---|---|
| `EnvironmentError: HELICONE_API_KEY is not set` | Key not injected; check Codespaces secrets |
| LangGraph: `KeyError: token_usage` | Model returned no usage data; check `response.response_metadata` keys |
| AutoGen: agent chat terminates immediately | `TERMINATE` string appearing too early; tighten `is_termination_msg` lambda |
| AutoGen: `AssistantAgent has no attribute register_for_llm` | Wrong AutoGen version; check `pip show pyautogen` |
| CrewAI: tool not called | Docstring missing or too vague; make it a precise one-liner |
| CrewAI: `verbose` floods output | Expected — use `verbose=False` during debugging if needed |
| LangGraph: infinite loop | `should_continue` returning `"agent"` when agent has no tool calls; add an `else: return END` |

---

## Hints

- **LangGraph** gives you the most explicit control: you wire every edge and
  decide exactly what goes in the state. This makes debugging easier.
- **AutoGen** is conversation-first: agents communicate by exchanging messages.
  The framework manages turn-taking; you mostly configure agents.
- **CrewAI** is role-first: you define what an agent *is* (role, goal,
  backstory) rather than what it *does* step by step.

---

## Success Criteria

All three agents must:
1. Produce a valid `word_freq.py` file containing a `word_frequency` function
   with type annotations
2. Produce a `sample.txt` file with at least 20 words
3. Record token usage in their respective `tracker` objects
4. Complete within 15 steps

You must also:
5. Fill in at least 2 sections of `comparison/tradeoffs.md`
6. Pass all non-skipped tests with `pytest tests/ -v`
