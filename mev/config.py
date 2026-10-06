"""Central path constants for the mev package."""
from __future__ import annotations

import os
from pathlib import Path

# Repo roots
ROOT = Path(__file__).resolve().parent.parent.parent  # mekanismirealismi/
BOOK_ROOT = ROOT.parent                               # ~/c/civos/book/

# LawVM
LAWVM_DIR = BOOK_ROOT / "LawVM"
GRAPH_DIR = LAWVM_DIR / ".tmp" / "corpus_graph_full"
CENSUS_DIR = ROOT / ".tmp" / "census_results"

# HE pipeline
HE_DB_DIR = ROOT / ".tmp" / "he_dbs"
HE_INDEX_DB = ROOT / ".tmp" / "he_master_index.db"
ENRICHMENTS_DB = ROOT / ".tmp" / "he_enrichments.db"

# Legislative index (shared, in BOOK_ROOT)
INDEX_DB = BOOK_ROOT / "data" / "legislative_index.sqlite"

# State causal map (deployed DB)
CAUSAL_MAP_DB = ROOT / "data" / "statute_graph" / "state_causal_map.db"

# Statute ZIP (Finlex bulk download)
STATUTE_ZIP = Path(os.environ.get("STATUTE_ZIP", str(Path.home() / "Downloads" / "statute.zip")))

# AKN government-proposal ZIP (for signatory extraction)
AKN_ZIP_PATH = Path(os.environ.get("AKN_ZIP_PATH", str(Path.home() / "Downloads" / "government-proposal.zip")))

# Llama server
LLAMA_API_BASE = os.environ.get("LLAMA_API_BASE", "http://localhost:8080")
