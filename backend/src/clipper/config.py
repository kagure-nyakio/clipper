import os

from dotenv import load_dotenv

load_dotenv()


def get_openai_api_key() -> str:
    """Return the API key when an operation actually needs OpenAI access."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required to analyze highlights. "
            "Set it in the environment or in a local .env file."
        )
    return api_key
