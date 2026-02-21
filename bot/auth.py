"""One-time OAuth authentication flow for Schwab API.

Run this module directly to complete the initial browser-based auth:
    python -m bot.auth
"""

import schwab
from bot.config import load_config


def run_auth_flow():
    """Launch the OAuth browser flow and save the token."""
    cfg = load_config()

    print("=" * 60)
    print("Schwab API Authentication")
    print("=" * 60)
    print()
    print("A browser window will open. Log in to your Schwab account")
    print("and authorize the application. After authorization, you'll")
    print("be redirected to a URL — paste that full URL back here.")
    print()

    client = schwab.auth.client_from_manual_flow(
        api_key=cfg.schwab.app_key,
        app_secret=cfg.schwab.app_secret,
        callback_url=cfg.schwab.callback_url,
        token_path=cfg.schwab.token_path,
    )

    print()
    print("Authentication successful! Token saved to:", cfg.schwab.token_path)
    print("You can now run the trading bot.")
    return client


if __name__ == "__main__":
    run_auth_flow()
