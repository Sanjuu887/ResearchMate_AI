"""Centralized environment and Streamlit secrets configuration."""

from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class AppConfig:
    gemini_api_key: str = field(default="", repr=False)
    gemini_model: str = "gemini-3.8-flash"
    tavily_api_key: str = field(default="", repr=False)
    tavily_search_depth: str = "advanced"
    mcp_host: str = "localhost"
    mcp_port: int = 8000


_active_config: ContextVar[AppConfig | None] = ContextVar(
    "researchmate_config", default=None
)


def load_config(secrets: Mapping[str, object] | None = None) -> AppConfig:
    """Load settings from environment variables, .env, and optional Streamlit secrets."""
    load_dotenv()

    def value(name: str, default: str = "") -> str:
        env_value = os.getenv(name, "").strip()
        if env_value:
            return env_value
        if secrets is not None:
            secret_value = secrets.get(name)
            if secret_value is not None:
                configured_value = str(secret_value).strip()
                if configured_value:
                    return configured_value
        return default

    try:
        mcp_port = int(value("MCP_PORT", "8000"))
    except ValueError as exc:
        raise ValueError("MCP_PORT must be a valid port number.") from exc

    if not 1 <= mcp_port <= 65535:
        raise ValueError("MCP_PORT must be between 1 and 65535.")

    return AppConfig(
        gemini_api_key=value("GEMINI_API_KEY"),
        gemini_model=value("GEMINI_MODEL", "gemini-3.8-flash"),
        tavily_api_key=value("TAVILY_API_KEY"),
        tavily_search_depth=value("TAVILY_SEARCH_DEPTH", "advanced"),
        mcp_host=value("MCP_HOST", "localhost"),
        mcp_port=mcp_port,
    )


def get_config() -> AppConfig:
    """Return the request-scoped config or load local environment configuration."""
    return _active_config.get() or load_config()


@contextmanager
def use_config(config: AppConfig) -> Generator[None, None, None]:
    """Make configuration available to shared service functions for one call."""
    token: Token[AppConfig | None] = _active_config.set(config)
    try:
        yield
    finally:
        _active_config.reset(token)
