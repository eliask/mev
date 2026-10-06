"""Paths and run constants. Everything required lives inside this repository."""


import os
from pathlib import Path

PKG = Path(__file__).resolve().parent
_SOURCE_ROOT = PKG.parent
# A wheel's contracts stay in the installed package; its generated records
# belong in the working directory, not in potentially read-only site-packages.
ROOT = Path(os.environ["PAA_WORKSPACE_ROOT"]).resolve() if os.environ.get("PAA_WORKSPACE_ROOT") else (
    _SOURCE_ROOT if (_SOURCE_ROOT / "pyproject.toml").is_file() else Path.cwd().resolve()
)
DATA = ROOT / "data"
RAW = DATA / "raw"
EXPORT = DATA / "export"
REPORTS = ROOT / "reports"
DIST = ROOT / "dist" / "browser"
SCHEMA_DIR = PKG / "contracts" / "schemas"
FIXTURE_DIR = PKG / "contracts" / "fixtures"

DB_PATH = DATA / "paa.sqlite"
RUN_ID = os.environ.get("PAA_RUN_ID", "paa-2026-10-06")
CORPUS_CUTOFF = "2026-10-06"
TERM_START = "2023-04-12"
USER_AGENT = "paa/0.1 (finnish political memory; local research)"
EDUSKUNTA_API = "https://avoindata.eduskunta.fi/api/v1/tables"
VOTE_META_YEARS = range(2007, 2027)
BALLOT_YEARS = range(2023, 2027)

# Optional local acceleration. Never required for a clean run.
STATUTE_DB = os.environ.get("PAA_STATUTE_DB") or ""


def ensure_dirs() -> None:
    for path in (DATA, RAW, EXPORT, REPORTS, DIST, RAW / "vaalit", RAW / "yle", RAW / "eduskunta"):
        path.mkdir(parents=True, exist_ok=True)
