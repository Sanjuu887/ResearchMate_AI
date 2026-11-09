"""List Gemini models available to the API key stored in the local environment."""

import sys

from google import genai
from google.genai.errors import APIError

from config import get_config


def list_compatible_models(api_key: str) -> list[str]:
    """Return model IDs that advertise generateContent support."""
    client = genai.Client(api_key=api_key)
    model_ids = []
    for model in client.models.list():
        actions = {
            str(action).casefold()
            for action in (getattr(model, "supported_actions", None) or [])
        }
        name = getattr(model, "name", None)
        if name and "generatecontent" in actions:
            model_ids.append(str(name).removeprefix("models/"))
    return model_ids


def main() -> int:
    api_key = get_config().gemini_api_key
    if not api_key:
        print(
            "GEMINI_API_KEY is missing from the environment or local .env file.",
            file=sys.stderr,
        )
        return 1

    try:
        model_ids = list_compatible_models(api_key)
    except APIError as exc:
        print(
            "Google could not list models for this key. Check that the key is "
            "active and its project has Gemini API access.",
            file=sys.stderr,
        )
        print(f"Google error: HTTP {exc.code} {exc.status or ''}".strip(), file=sys.stderr)
        return 1

    if not model_ids:
        print("Google returned no models supporting generateContent.")
        return 1

    print("Gemini models available to this key (model IDs only):")
    for model_id in model_ids:
        print(model_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
