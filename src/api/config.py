"""API settings. Everything is read from environment variables (or a local .env file).

Nothing here needs a source edit to change: see .env.example in the repo root.
"""
import os
from pathlib import Path

try:  # python-dotenv is optional; the API still runs without it
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


def _csv(value: str) -> list:
    return [item.strip() for item in value.split(",") if item.strip()]


# Folder that contains one sub-folder per trip: <DATA_DIR>/<trip_id>/<trip_id>.track ...
DATA_DIR = Path(os.getenv("PATHSNAP_DATA_DIR", "data/student_test"))

# Optional saved model artifact. Empty/unset means "use the documented baseline".
_artifact = os.getenv("PATHSNAP_MODEL_ARTIFACT", "").strip()
MODEL_ARTIFACT = _artifact or None

HOST = os.getenv("PATHSNAP_API_HOST", "0.0.0.0")
PORT = int(os.getenv("PATHSNAP_API_PORT", "8000"))

# Allowed browser origins for the UI. "*" is fine for a closed two-laptop demo.
CORS_ORIGINS = _csv(os.getenv("PATHSNAP_CORS_ORIGINS", "*"))
