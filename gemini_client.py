"""
Gemini LLM API client.
Loads GEMINI_API_KEY from .env or from environment (or GOOGLE_API_KEY).
Get a key: https://aistudio.google.com/apikey
"""
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()  # load .env from project root if present

# API key: .env (GEMINI_API_KEY) or env var GEMINI_API_KEY / GOOGLE_API_KEY
_api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
_client = None

DEFAULT_MODEL = "gemini-2.5-flash"


def get_client(api_key: str | None = None):
    """Return a configured Gemini client."""
    global _client
    if _client is None:
        key = api_key or _api_key
        _client = genai.Client(api_key=key) if key else genai.Client()
    return _client


def generate(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    system_instruction: str | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    api_key: str | None = None,
) -> str:
    """
    Call Gemini LLM API and return the generated text.

    Args:
        prompt: User message or full prompt.
        model: Model name (e.g. gemini-2.0-flash, gemini-2.5-flash).
        system_instruction: Optional system prompt.
        temperature: 0–1; higher = more random.
        max_tokens: Max output tokens (optional).
        api_key: Override API key (optional).

    Returns:
        Generated text string.
    """
    client = get_client(api_key=api_key)
    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        system_instruction=system_instruction,
    )
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    return response.text or ""


if __name__ == "__main__":
    # Quick test
    import sys
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Say hello in one sentence."
    print(generate(prompt))
