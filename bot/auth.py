"""One-time OAuth authentication flow for Schwab API.

Run this module directly to complete the initial browser-based auth:
    python -m bot.auth
"""

from pathlib import Path

import schwab
from bot.config import load_config
from bot.file_security import tighten_file_permissions, validate_sensitive_file


def run_auth_flow():
    """Launch the OAuth browser flow and save the token."""
    cfg = load_config()
    token_path = Path(cfg.schwab.token_path).expanduser()

    print("=" * 60)
    print("Schwab API Authentication")
    print("=" * 60)
    print()
    print("A browser window will open. Log in to your Schwab account")
    print("and authorize the application. After authorization, you'll")
    print("be redirected to a URL — paste that full URL back here.")
    print()

    validate_sensitive_file(
        token_path,
        label="Schwab token file",
        allow_missing=True,
    )
    if token_path.exists():
        tighten_file_permissions(token_path, label="Schwab token file")

    client = schwab.auth.client_from_manual_flow(
        api_key=cfg.schwab.app_key,
        app_secret=cfg.schwab.app_secret,
        callback_url=cfg.schwab.callback_url,
        token_path=str(token_path),
    )

    validate_sensitive_file(
        token_path,
        label="Schwab token file",
        allow_missing=False,
    )
    tighten_file_permissions(token_path, label="Schwab token file")

    print()
    print("Authentication successful! Token saved to:", token_path)
    print("You can now run the trading bot.")
    return client


if __name__ == "__main__":
    run_auth_flow()
