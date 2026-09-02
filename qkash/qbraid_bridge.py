"""Small qBraid configuration bridge.

The current implementation runs QAOA locally through Qiskit.  This module keeps
qBraid credentials and target metadata in one place so remote execution can be
added without changing the scoring or benchmarking layers.
"""

from __future__ import annotations

# Imports.
from dataclasses import dataclass
import os
from pathlib import Path


# Variable descriptions.
# PROJECT_ROOT points at the repository root that contains app.py and .env.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ENV_FILE is the only dotenv-style file read for secrets.
ENV_FILE = PROJECT_ROOT / ".env"

# QBRAID_ENV_KEYS are the qBraid/QKash variables accepted by the app.
QBRAID_ENV_KEYS = {
    "QBRAID_API_KEY",
    "QBRAID_PROVIDER",
    "QBRAID_DEVICE_ID",
    "QKASH_USE_QBRAID",
}


@dataclass(frozen=True)
class QbraidConfig:
    """Sanitized qBraid configuration used by the UI and backend factory."""

    api_key_present: bool
    provider: str
    device_id: str
    enabled: bool
    sdk_available: bool
    sdk_version: str
    env_file_loaded: bool
    api_key_source: str


def load_qbraid_config() -> QbraidConfig:
    """Read optional qBraid settings from environment variables."""

    env_file_loaded = load_project_env()
    api_key = os.getenv("QBRAID_API_KEY", "")
    provider = os.getenv("QBRAID_PROVIDER", "qbraid")
    device_id = os.getenv("QBRAID_DEVICE_ID", "qbraid:qbraid:sim:qir-sv")
    enabled = os.getenv("QKASH_USE_QBRAID", "false").strip().lower() in {"1", "true", "yes"}
    sdk_available, sdk_version = _qbraid_sdk_status()
    api_key_source = "environment/.env" if api_key else "not configured"
    return QbraidConfig(
        api_key_present=bool(api_key),
        provider=provider,
        device_id=device_id,
        enabled=enabled,
        sdk_available=sdk_available,
        sdk_version=sdk_version,
        env_file_loaded=env_file_loaded,
        api_key_source=api_key_source,
    )


def qbraid_status() -> dict[str, object]:
    """Return UI-safe qBraid status without exposing secrets."""

    config = load_qbraid_config()
    return {
        "enabled": config.enabled,
        "api_key_present": config.api_key_present,
        "provider": config.provider,
        "device_id": config.device_id,
        "sdk_available": config.sdk_available,
        "sdk_version": config.sdk_version,
        "env_file_loaded": config.env_file_loaded,
        "api_key_source": config.api_key_source,
    }


def get_qbraid_api_key() -> str:
    """Return the qBraid API key for backend execution without logging it."""

    load_project_env()
    return os.getenv("QBRAID_API_KEY", "").strip()


def load_project_env() -> bool:
    """Load qBraid variables from `.env` if present.

    The example file is intentionally ignored so real keys are not encouraged
    in a tracked template.
    """

    if not ENV_FILE.exists():
        return False

    for raw_line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in QBRAID_ENV_KEYS:
            continue
        os.environ[key] = _clean_env_value(value)
    return True


def _qbraid_sdk_status() -> tuple[bool, str]:
    """Check qBraid SDK availability without requiring it at import time."""

    try:
        import qbraid

        return True, str(getattr(qbraid, "__version__", "unknown"))
    except Exception:
        return False, "not installed"


def _clean_env_value(value: str) -> str:
    """Strip quotes and whitespace from a dotenv value."""

    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        return cleaned[1:-1]
    return cleaned
