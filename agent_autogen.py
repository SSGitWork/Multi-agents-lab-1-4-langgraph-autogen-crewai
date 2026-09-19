"""
agent_autogen.py  –  Lab 1.4 skeleton: AutoGen implementation.

Your job
--------
Implement the two functions marked TODO so that an AutoGen
AssistantAgent + UserProxyAgent pair solves TASK_DESCRIPTION using
the three tools from tools/code_tools.py.

Do NOT change:
  - The function signatures of `make_assistant` and `make_proxy`
  - The `run()` function structure after the TODO comment
  - Any imports from llm_client, tools, or task

Run:
  python agent_autogen.py
"""

from __future__ import annotations

import os

import autogen
from autogen import AssistantAgent, UserProxyAgent

from llm_client import DEFAULT_MODEL
from task import TASK_DESCRIPTION
from tools.code_tools import dispatch, exec_python, read_file, write_file
from tools.token_counter import UsageTracker
from dotenv import load_dotenv

load_dotenv(override=True)

tracker = UsageTracker(framework="AutoGen")

# ---------------------------------------------------------------------------
# LLM config – AutoGen uses its own config dict format.
# We point it at the Helicone proxy by overriding the base_url.
# ---------------------------------------------------------------------------

def _llm_config() -> dict:
    """Return an AutoGen llm_config dict for Azure OpenAI or OpenRouter.

    This function is already implemented for you. Study it to understand
    how AutoGen's llm_config dict maps onto the OpenAI client.
    """
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT") or DEFAULT_MODEL

    helicone_base = os.getenv("HELICONE_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    helicone_api_key = os.getenv("HELICONE_API_KEY")

    if azure_endpoint and azure_api_key and azure_api_version and azure_deployment:
        return {
            "config_list": [
                {
                    "model": azure_deployment,
                    "api_key": azure_api_key,
                    "base_url": azure_endpoint,
                    "api_type": "openai",
                }
            ],
            "temperature": 0,
        }

    if helicone_base and openrouter_api_key:
        config: dict = {
            "config_list": [
                {
                    "model": DEFAULT_MODEL,
                    "api_key": openrouter_api_key,
                    "base_url": helicone_base,
                    "api_type": "openai",
                }
            ],
            "temperature": 0,
        }
        if helicone_api_key:
            config["config_list"][0]["default_headers"] = {
                "Helicone-Auth": f"Bearer {helicone_api_key}",
            }
        return config

    raise EnvironmentError(
        "Set either Azure OpenAI variables or OpenRouter/Helicone variables."
    )


# ---------------------------------------------------------------------------
# Tool functions registered with AutoGen
# ---------------------------------------------------------------------------
# AutoGen's function-calling integration requires plain Python callables.
# The three tool functions (read_file, write_file, exec_python) are imported
# directly from tools.code_tools above.

SYSTEM_PROMPT = """You are a Coder Agent. Use the available tools to complete
the user's task step by step. When the task is fully done, output the single
word DONE on its own line and nothing else. Do NOT use code blocks for tool
calls – call the registered functions directly."""


# ---------------------------------------------------------------------------
# Agent factory functions
# ---------------------------------------------------------------------------

def make_assistant() -> AssistantAgent:
    """Create and return an AssistantAgent with the three tools registered.

    TODO: Implement this function.

    Steps:
    1. Create an AssistantAgent with:
       - name="coder_agent"
       - system_message=SYSTEM_PROMPT
       - llm_config=_llm_config()
    2. Register each tool for LLM use with:
         agent.register_for_llm(name="...", description="...")(function)
       Register: read_file, write_file, exec_python.
       Hint: Use the descriptions from tools/code_tools.py (READ_FILE_SCHEMA,
             WRITE_FILE_SCHEMA, EXEC_PYTHON_SCHEMA).
    3. Return the agent.

    Raises:
        NotImplementedError: Remove this line once you implement the function.
    """
    agent = AssistantAgent(
        name="coder_agent",
        system_message=SYSTEM_PROMPT,
        llm_config=_llm_config(),
    )
    agent.register_for_llm(name="read_file", description="Read a file from the sandbox workspace.")(read_file)
    agent.register_for_llm(name="write_file", description="Write content to a file in the sandbox workspace.")(write_file)
    agent.register_for_llm(name="exec_python", description="Execute a Python code snippet in a sandboxed subprocess. Returns stdout + stderr, truncated to 2 000 characters.")(exec_python)
    return agent


def make_proxy() -> UserProxyAgent:
    """Create and return a UserProxyAgent that executes tool calls.

    TODO: Implement this function.

    Steps:
    1. Create a UserProxyAgent with:
       - name="user_proxy"
       - human_input_mode="NEVER"
       - max_consecutive_auto_reply=15
       - is_termination_msg: a lambda that returns True when the message
         content contains "DONE"
       - code_execution_config=False  (we use our own exec_python tool)
    2. Register each tool for execution with:
         proxy.register_for_execution(name="...")(function)
       Register the same three functions as in make_assistant().
    3. Return the proxy.

    Raises:
        NotImplementedError: Remove this line once you implement the function.
    """
    proxy = UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=15,
        is_termination_msg=lambda msg: "DONE" in ((msg or {}).get("content") or "") if isinstance(msg, dict) else "DONE" in str(msg or ""),
        code_execution_config=False,
    )
    proxy.register_for_execution(name="read_file")(read_file)
    proxy.register_for_execution(name="write_file")(write_file)
    proxy.register_for_execution(name="exec_python")(exec_python)
    return proxy


# ---------------------------------------------------------------------------
# Token tracking helper
# ---------------------------------------------------------------------------

def _record_usage(chat_result: autogen.ChatResult) -> None:
    """Extract token usage from a ChatResult and record each exchange.

    This function is already implemented for you.
    AutoGen accumulates usage in chat_result.cost – we split it into
    per-step records using the message count as a proxy for step count.
    """
    usage = getattr(chat_result, "cost", {})
    if not isinstance(usage, dict):
        return

    # usage dict shape may vary by AutoGen version.
    model_usage = usage.get("usage_including_cached_inference", {})
    if isinstance(model_usage, dict):
        for model_data in model_usage.values():
            if not isinstance(model_data, dict):
                continue
            prompt = int(model_data.get("prompt_tokens", 0) or 0)
            completion = int(model_data.get("completion_tokens", 0) or 0)
            if prompt or completion:
                tracker.record_step(prompt_tokens=prompt, completion_tokens=completion)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run() -> str:
    """Initiate the two-agent chat and return the final assistant message."""
    assistant = make_assistant()
    proxy = make_proxy()

    result = proxy.initiate_chat(
        recipient=assistant,
        message=TASK_DESCRIPTION,
        max_turns=15,
    )
    _record_usage(result)

    tracker.print_summary()

    # Return the last message from the assistant
    last = result.chat_history[-1]["content"] if result.chat_history else ""
    return last


if __name__ == "__main__":
    output = run()
    print("\n--- Agent finished ---")
    print(output)
