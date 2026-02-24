"""Guided live setup for credentials, OAuth token, and account selection."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Optional

from dotenv import dotenv_values, set_key

from bot.auth import run_auth_flow
from bot.config import load_config
from bot.schwab_client import SchwabClient, _mask_hash

REQUIRED_LIVE_KEYS = ("SCHWAB_APP_KEY", "SCHWAB_APP_SECRET")
PLACEHOLDER_MARKERS = ("your_", "_here", "changeme", "example")


def run_live_setup(
    *,
    config_path: str = "config.yaml",
    auto_auth: bool = True,
    auto_select_account: bool = True,
) -> None:
    """Run guided live setup and fail fast when required state is missing."""
    cfg_path = Path(config_path).expanduser().resolve()
    root = cfg_path.parent
    env_path = root / ".env"
    env_example = root / ".env.example"

    created = _ensure_env_file(env_path, env_example)
    if created:
        print(f"Created {env_path} from {env_example.name}.")

    _collect_required_secrets(env_path)
    cfg = load_config(str(cfg_path))
    client = SchwabClient(cfg.schwab)

    try:
        client.connect()
    except RuntimeError as exc:
        if "Token file not found" in str(exc):
            if auto_auth and sys.stdin.isatty():
                print("No Schwab OAuth token found. Starting browser auth flow...")
                run_auth_flow(str(cfg_path))
                cfg = load_config(str(cfg_path))
                client = SchwabClient(cfg.schwab)
                client.connect()
            else:
                raise RuntimeError(
                    "Schwab token is missing. Run `python3 -m bot.auth`."
                ) from exc
        else:
            raise

    selected_hash = _resolve_or_select_account_hash(
        client=client,
        configured_hash=cfg.schwab.account_hash,
        configured_index=cfg.schwab.account_index,
        auto_select=auto_select_account,
    )
    if not cfg.schwab.account_hash or cfg.schwab.account_hash != selected_hash:
        set_key(
            str(env_path),
            "SCHWAB_ACCOUNT_HASH",
            selected_hash,
            quote_mode="never",
        )
        print(f"Saved SCHWAB_ACCOUNT_HASH={_mask_hash(selected_hash)}")

    # Connectivity probe to verify setup end-to-end.
    quote = client.get_quote("SPY")
    qref = quote.get("quote", quote)
    price = float(qref.get("lastPrice", qref.get("mark", 0.0)))
    if price <= 0:
        raise RuntimeError("Live setup probe failed: invalid SPY quote.")

    print("Live setup complete.")
    print(f"- Account: {_mask_hash(selected_hash)}")
    print(f"- Token: {Path(cfg.schwab.token_path).expanduser()}")
    print(f"- SPY probe: ${price:.2f}")
    print("Next step: python3 main.py --live --preflight-only")


def _ensure_env_file(env_path: Path, template_path: Path) -> bool:
    """Create .env from template when missing."""
    if env_path.exists():
        return False
    if template_path.exists():
        shutil.copyfile(template_path, env_path)
    else:
        env_path.write_text("", encoding="utf-8")
    return True


def _collect_required_secrets(env_path: Path) -> None:
    """Ensure required Schwab credentials are present in .env."""
    env_values = dotenv_values(str(env_path))
    missing = [k for k in REQUIRED_LIVE_KEYS if _is_missing_value(env_values.get(k))]

    if not missing:
        return

    if not sys.stdin.isatty():
        missing_str = ", ".join(missing)
        raise RuntimeError(
            f"Missing required live credentials in {env_path}: {missing_str}"
        )

    print("Missing required live credentials:")
    for key in missing:
        entered = ""
        while not entered:
            entered = input(f"  Enter {key}: ").strip()
        set_key(str(env_path), key, entered, quote_mode="never")


def _is_missing_value(value: Optional[str]) -> bool:
    text = (value or "").strip()
    if not text:
        return True
    lowered = text.lower()
    return any(marker in lowered for marker in PLACEHOLDER_MARKERS)


def _resolve_or_select_account_hash(
    *,
    client: SchwabClient,
    configured_hash: str,
    configured_index: int,
    auto_select: bool,
) -> str:
    """Resolve live account hash with deterministic priority."""
    hashes = client._fetch_account_hashes()  # noqa: SLF001 - setup utility context
    if not hashes:
        raise RuntimeError("No linked Schwab accounts found for this token.")

    if configured_hash:
        if configured_hash not in hashes:
            raise RuntimeError(
                "Configured SCHWAB_ACCOUNT_HASH is not linked to this token."
            )
        return configured_hash

    if configured_index >= 0:
        if configured_index >= len(hashes):
            raise RuntimeError(
                f"SCHWAB_ACCOUNT_INDEX={configured_index} is out of range "
                f"(0..{len(hashes) - 1})."
            )
        return hashes[configured_index]

    if len(hashes) == 1:
        return hashes[0]

    if auto_select and sys.stdin.isatty():
        print("Multiple linked Schwab accounts detected:")
        for i, value in enumerate(hashes):
            print(f"  [{i}] {_mask_hash(value)}")
        while True:
            choice = input("Select account index: ").strip()
            if not choice.isdigit():
                print("Please enter a valid number.")
                continue
            idx = int(choice)
            if 0 <= idx < len(hashes):
                return hashes[idx]
            print("Index out of range.")

    masked = ", ".join(_mask_hash(v) for v in hashes)
    raise RuntimeError(
        "Multiple linked Schwab accounts found. Set SCHWAB_ACCOUNT_HASH "
        f"or SCHWAB_ACCOUNT_INDEX. Options: {masked}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Guided live setup for TradingBot")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    parser.add_argument(
        "--no-auth",
        action="store_true",
        help="Do not launch OAuth flow automatically if token is missing",
    )
    parser.add_argument(
        "--no-account-select",
        action="store_true",
        help="Do not prompt for account selection when multiple accounts are linked",
    )
    args = parser.parse_args()

    run_live_setup(
        config_path=args.config,
        auto_auth=not args.no_auth,
        auto_select_account=not args.no_account_select,
    )


if __name__ == "__main__":
    main()
