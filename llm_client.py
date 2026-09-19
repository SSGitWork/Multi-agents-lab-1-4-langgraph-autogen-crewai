"""
llm_client.py  –  Shared OpenAI client for Helicone/OpenRouter or Azure AI Foundry.

Every framework skeleton imports get_client() from here so that all LLM
calls flow through a single point, giving you unified token-usage logging.
"""

import os

from openai import AzureOpenAI, OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

_client: OpenAI | AzureOpenAI | None = None


def get_client() -> OpenAI:
    """Return a module-level OpenAI-compatible client.

    If Azure AI Foundry settings are present, an AzureOpenAI client is
    created. Otherwise, the client routes through Helicone/OpenRouter.
    """
    global _client

    if _client is None:
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

        if azure_endpoint and azure_api_key and azure_api_version and azure_deployment:
            _client = AzureOpenAI(
                api_key=azure_api_key,
                azure_endpoint=azure_endpoint,
                api_version=azure_api_version,
            )
            return _client

        _HELICONE_BASE = os.getenv("HELICONE_BASE_URL")
        _OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
        _HELICONE_API_KEY = os.getenv("HELICONE_API_KEY")
        if not _HELICONE_API_KEY:
            raise EnvironmentError(
                "Neither Azure OpenAI settings nor HELICONE_API_KEY are set. "
                "Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY or OPENAI_API_KEY, "
                "AZURE_OPENAI_API_VERSION, and AZURE_OPENAI_DEPLOYMENT for Azure AI Foundry, "
                "or set HELICONE_API_KEY for the lab default."
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
DEFAULT_MODEL = os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv("OPENAI_MODEL") or "gpt-4.1-mini"
