import os
from dataclasses import dataclass
from typing import Optional


API_BASE_ENV = "API_BASE_URL"
API_TOKEN_ENV = "API_TOKEN"


@dataclass
class APISettings:
    base_url: str
    token: Optional[str]


class MissingConfigurationError(RuntimeError):
    """Raised when required configuration is missing."""


def get_settings(base_url: Optional[str] = None, token: Optional[str] = None) -> APISettings:
    """Load API settings from arguments or environment variables.

    Args:
        base_url: Optional base URL override.
        token: Optional API token override.

    Returns:
        APISettings populated from the first non-empty value in the priority
        order of explicit override then environment variable.

    Raises:
        MissingConfigurationError: when the base URL is not provided.
    """

    resolved_base = base_url or os.getenv(API_BASE_ENV)
    resolved_token = token or os.getenv(API_TOKEN_ENV)

    if not resolved_base:
        raise MissingConfigurationError(
            f"API base URL is required. Set {API_BASE_ENV} or pass --base-url."
        )

    return APISettings(base_url=resolved_base.rstrip("/"), token=resolved_token)
