"""SQLite store. The database is a local build artifact, not a source of truth."""


import json
import sqlite3
from pathlib import Path

from paa.config import DB_PATH, ensure_dirs

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS manifest (
    source_id TEXT,
    url TEXT,
    sha256 TEXT,
    bytes INTEGER,
    http_status INTEGER,
    retrieved_at TEXT,
    note TEXT
);
CREATE TABLE IF NOT EXISTS candidacies (
    candidacy_id TEXT PRIMARY KEY,
    election_year INTEGER,
    district_code TEXT,
    district_abbr TEXT,
    party TEXT,
    candidate_number INTEGER,
    first_name TEXT,
    last_name TEXT,
    name_key TEXT,
    age INTEGER,
    occupation TEXT,
    home_municipality TEXT,
    votes INTEGER,
    elected INTEGER,
    valintatieto TEXT,
    actor_id TEXT
);
CREATE TABLE IF NOT EXISTS mp_people (
    person_id TEXT PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    name_key TEXT,
    birth_year INTEGER,
    death_date TEXT,
    ended_date TEXT,
    minister INTEGER,
    json TEXT
);
CREATE TABLE IF NOT EXISTS mp_periods (
    person_id TEXT,
    kind TEXT,
    label TEXT,
    start_date TEXT,
    end_date TEXT,
    precision TEXT
);
CREATE TABLE IF NOT EXISTS actors (
    actor_id TEXT PRIMARY KEY,
    display_name TEXT,
    name_key TEXT,
    birth_year INTEGER,
    person_id TEXT,
    identity_status TEXT
);
CREATE TABLE IF NOT EXISTS vote_events (
    aanestys_id TEXT PRIMARY KEY,
    year INTEGER,
    session_date TEXT,
    number INTEGER,
    title TEXT,
    lisa TEXT,
    kohta TEXT,
    jaa INTEGER,
    ei INTEGER,
    tyhjaa INTEGER,
    poissa INTEGER,
    yhteensa INTEGER,
    url TEXT,
    ptk TEXT,
    matter TEXT,
    mitatoity INTEGER,
    json TEXT
);
CREATE TABLE IF NOT EXISTS ballots (
    aanestys_id TEXT,
    person_number TEXT,
    first_name TEXT,
    last_name TEXT,
    name_key TEXT,
    party TEXT,
    raw_response TEXT,
    PRIMARY KEY (aanestys_id, person_number)
);
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    source_id TEXT,
    url TEXT,
    actor_id TEXT,
    field_label TEXT,
    language TEXT,
    text TEXT,
    stated_earliest TEXT,
    sha256 TEXT,
    http_status INTEGER,
    retrieved_at TEXT
);
CREATE TABLE IF NOT EXISTS statements (
    statement_id TEXT PRIMARY KEY,
    json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS propositions (
    proposition_id TEXT PRIMARY KEY,
    statement_id TEXT,
    json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS findings (
    finding_id TEXT PRIMARY KEY,
    json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS vote_records (
    vote_id TEXT PRIMARY KEY,
    json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS relations (
    relation_id TEXT PRIMARY KEY,
    proposition_id TEXT,
    aanestys_id TEXT,
    actor_id TEXT,
    status TEXT,
    note TEXT
);
CREATE TABLE IF NOT EXISTS official_objects (object_id TEXT PRIMARY KEY, json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS source_coverage (coverage_id TEXT PRIMARY KEY, json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS relation_reviews (review_id TEXT PRIMARY KEY, json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS evidence (evidence_id TEXT PRIMARY KEY, json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS decision_episodes (episode_id TEXT PRIMARY KEY, matter_id TEXT NOT NULL, json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS inquiry_cases (case_id TEXT PRIMARY KEY, json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS group_contexts (person_id TEXT PRIMARY KEY, json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS group_sources (source_id TEXT PRIMARY KEY, json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS evidence_traces (
    trace_id TEXT PRIMARY KEY,
    statement_id TEXT NOT NULL,
    proposition_id TEXT NOT NULL,
    actor_id TEXT,
    state TEXT NOT NULL,
    json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trace_statement ON evidence_traces(statement_id);
CREATE INDEX IF NOT EXISTS idx_cand_name ON candidacies(name_key, election_year);
CREATE INDEX IF NOT EXISTS idx_ballot_name ON ballots(name_key);
CREATE INDEX IF NOT EXISTS idx_ballot_person ON ballots(person_number);
CREATE INDEX IF NOT EXISTS idx_cand_actor ON candidacies(actor_id, election_year);
CREATE INDEX IF NOT EXISTS idx_period_person ON mp_periods(person_id);
CREATE INDEX IF NOT EXISTS idx_vote_year ON vote_events(year);
"""


def connect(path: Path | None = None) -> sqlite3.Connection:
    if path is None:
        ensure_dirs()
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path or DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    if "retrieved_at" not in {row["name"] for row in conn.execute("PRAGMA table_info(documents)")}:
        conn.execute("ALTER TABLE documents ADD COLUMN retrieved_at TEXT")
    return conn


def put_meta(conn: sqlite3.Connection, key: str, value: object) -> None:
    payload = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, payload))


def add_manifest(conn: sqlite3.Connection, **row: object) -> None:
    conn.execute(
        """INSERT INTO manifest(source_id, url, sha256, bytes, http_status, retrieved_at, note)
           VALUES (:source_id, :url, :sha256, :bytes, :http_status, :retrieved_at, :note)""",
        row,
    )
