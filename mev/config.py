"""Workspace paths for legislative tooling, independent of installation layout."""

import os
from pathlib import Path

# Source checkouts default to their project root; installed wheels use the
# caller's workspace. Shared data and optional integrations are explicit inputs.
_SOURCE_ROOT = Path(__file__).resolve().parent.parent
ROOT = Path(os.environ["MEV_WORKSPACE_ROOT"]).resolve() if os.environ.get("MEV_WORKSPACE_ROOT") else (
    _SOURCE_ROOT if (_SOURCE_ROOT / "pyproject.toml").is_file() else Path.cwd().resolve()
)
BOOK_ROOT = Path(os.environ.get("MEV_SHARED_ROOT", str(ROOT))).resolve()

# LawVM
LAWVM_DIR = Path(os.environ.get("LAWVM_DIR", str(ROOT / "LawVM"))).resolve()
GRAPH_DIR = LAWVM_DIR / ".tmp" / "corpus_graph_full"
CENSUS_DIR = ROOT / ".tmp" / "census_results"

# HE pipeline
HE_DB_DIR = ROOT / ".tmp" / "he_dbs"
HE_INDEX_DB = ROOT / ".tmp" / "he_master_index.db"
ENRICHMENTS_DB = ROOT / ".tmp" / "he_enrichments.db"

# Legislative index (optionally shared through MEV_SHARED_ROOT)
INDEX_DB = BOOK_ROOT / "data" / "legislative_index.sqlite"

# State causal map (deployed DB)
CAUSAL_MAP_DB = ROOT / "data" / "statute_graph" / "state_causal_map.db"

# Statute ZIP (Finlex bulk download)
STATUTE_ZIP = Path(os.environ.get("STATUTE_ZIP", str(ROOT / "data" / "statute.zip"))).resolve()

# AKN government-proposal ZIP (for signatory extraction)
AKN_ZIP_PATH = Path(os.environ.get("AKN_ZIP_PATH", str(ROOT / "data" / "government-proposal.zip"))).resolve()

# Llama server
LLAMA_API_BASE = os.environ.get("LLAMA_API_BASE", "http://localhost:8080")
