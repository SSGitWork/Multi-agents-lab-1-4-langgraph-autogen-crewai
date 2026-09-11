"""
agent_crewai.py  –  Lab 1.4 skeleton: CrewAI implementation.

Your job
--------
Implement the three functions marked TODO so that a single CrewAI Agent
and Task solves TASK_DESCRIPTION using the three tools from
tools/code_tools.py.

Do NOT change:
  - The HeliconeOpenAI class
  - The function signatures of `make_tools`, `make_agent`, `make_task`
  - The `run()` function structure after the TODO comment
  - Any imports from llm_client, tools, or task

Run:
  python agent_crewai.py
"""

from __future__ import annotations

import os
import time

from crewai import Agent, Crew, Task
from crewai.tools import tool as crewai_tool
from langchain_openai import ChatOpenAI

from llm_client import DEFAULT_MODEL
from task import TASK_DESCRIPTION
from tools.code_tools import exec_python, read_file, write_file
from tools.token_counter import UsageTracker
from dotenv import load_dotenv

load_dotenv(override=True)

tracker = UsageTracker(framework="CrewAI")

# ---------------------------------------------------------------------------
# LLM wrapper – CrewAI accepts a LangChain-compatible ChatOpenAI object.
# ---------------------------------------------------------------------------


def _make_llm() -> ChatOpenAI:
    _HELICONE_API_KEY = os.getenv("HELICONE_API_KEY")

    return ChatOpenAI(
        model=DEFAULT_MODEL,
        default_headers={"Helicone-Auth": f"Bearer {_HELICONE_API_KEY}"},
        temperature=0,
    )


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

def make_tools() -> list:
    """Wrap the three code tools as CrewAI tool objects and return them.

    TODO: Implement this function.

    Steps:
    1. Use the @crewai_tool decorator (already imported as `crewai_tool`)
       to create three tool functions:
         - crewai_read_file(path: str) -> str
         - crewai_write_file(path: str, content: str) -> str
         - crewai_exec_python(code: str) -> str
       Each should delegate directly to the matching function imported from
       tools.code_tools.
    2. Return a list containing the three decorated tools.

    Hints:
    - @crewai_tool decorates a plain function; the docstring becomes the
      tool description shown to the LLM.
    - CrewAI infers parameter names and types from the function signature.
    - Example:
        @crewai_tool
        def my_tool(x: str) -> str:
            \"\"\"One-line description.\"\"\"
            return some_function(x)

    Raises:
        NotImplementedError: Remove this line once you implement the function.
    """
    raise NotImplementedError("Implement make_tools()")


def make_agent(tools: list) -> Agent:
    """Create and return a CrewAI Agent for the coding task.

    TODO: Implement this function.

    Steps:
    1. Create an Agent with:
       - role="Coder Agent"
       - goal="Complete the assigned coding task correctly and efficiently."
       - backstory: A short sentence describing the agent as an expert Python
         developer who always writes clean, tested code.
       - tools=tools  (the list returned by make_tools())
       - llm=_make_llm()
       - verbose=True
       - max_iter=15
    2. Return the agent.

    Raises:
        NotImplementedError: Remove this line once you implement the function.
    """
    raise NotImplementedError("Implement make_agent()")


def make_task(agent: Agent) -> Task:
    """Create and return a Task that assigns TASK_DESCRIPTION to the agent.

    TODO: Implement this function.

    Steps:
    1. Create a Task with:
       - description=TASK_DESCRIPTION
       - expected_output: A short string such as
         "The word_freq.py module is written, tested, and produces correct output."
       - agent=agent
    2. Return the task.

    Raises:
        NotImplementedError: Remove this line once you implement the function.
    """
    raise NotImplementedError("Implement make_task()")


# ---------------------------------------------------------------------------
# Token tracking helper
# ---------------------------------------------------------------------------

def _record_usage(crew: Crew) -> None:
    """Extract token usage from a Crew's usage_metrics and record it.

    This function is already implemented for you.
    """
    metrics = getattr(crew, "usage_metrics", None)
    if metrics:
        prompt = getattr(metrics, "prompt_tokens", 0) or 0
        completion = getattr(metrics, "completion_tokens", 0) or 0
        tracker.record_step(prompt_tokens=prompt, completion_tokens=completion)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run() -> str:
    """Assemble and run the crew; return the task output."""
    tools = make_tools()
    agent = make_agent(tools)
    task = make_task(agent)

    # TODO: Create a Crew with:
    #   - agents=[agent]
    #   - tasks=[task]
    #   - verbose=True
    # Call crew.kickoff() and store the result in `output`.
    # Then call _record_usage(crew).
    raise NotImplementedError("Create Crew and call kickoff() here")

    tracker.print_summary()
    return str(output)


if __name__ == "__main__":
    result = run()
    print("\n--- Agent finished ---")
    print(result)
