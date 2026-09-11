"""
llm_client.py  –  Shared OpenAI client routed through the Helicone proxy.

Every framework skeleton imports get_client() from here so that all LLM
calls flow through a single point, giving you unified token-usage logging
in the Helicone dashboard.

Students do NOT modify this file.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

_client: OpenAI | None = None


def get_client() -> OpenAI:
    """Return a module-level OpenAI client that routes through Helicone.

    The client is created once and reused on subsequent calls (singleton).
    Requires the HELICONE_API_KEY environment variable to be set.
    """
    global _client

    if _client is None:
        _HELICONE_BASE = os.getenv("HELICONE_BASE_URL")
        _OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
        _HELICONE_API_KEY = os.getenv("HELICONE_API_KEY")
        if not _HELICONE_API_KEY:
            raise EnvironmentError(
                "HELICONE_API_KEY is not set. "
                "In Codespaces it is injected automatically as an org secret. "
                "Running locally? Copy .env.example to .env and add your key."
            )
        _client = OpenAI(
            # Helicone key doubles as the auth token.
            api_key=_OPENROUTER_API_KEY,
            base_url=_HELICONE_BASE,
            default_headers={
                "Helicone-Auth": f"Bearer {_HELICONE_API_KEY}",
            },
        )
    return _client


# ---------------------------------------------------------------------------
# Convenience: default model names used throughout the lab.
# Change these here if you want to swap models globally.
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "gpt-4.1-mini"
