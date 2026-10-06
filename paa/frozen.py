"""Build the small, offline PAA acceptance corpus.

The frozen corpus is intentionally a source slice, not a hand-authored HTML
specimen.  It contains campaign documents, election/MP identity records, an
official-action slice, a closed coverage certificate, and two relation
reviews.  ``build_frozen`` imports those records into a fresh SQLite database
and then invokes the normal compiler and static-site writer.

The fixture has three useful cold-reader cases drawn from repository-frozen
public source slices:

* elected MP Miko Bergbom's conditional promise to leave his council seats,
  with an official Pirkanmaa regional-council resignation decision (the city
  council component remains an explicit scope residual);
* unelected 2011 candidate Armi Lindell's conditional legislative-initiative
  promise, which has no parliamentary opportunity; and
* Sanna Antikainen's diesel-tax position beside a related official vote whose
  personal relation remains unresolved.

No network, global ``data`` directory, or production acquisition routine is
used here.  This is the reproducibility seam for PAA-00/E18 and for tests
which need to distinguish a real pipeline run from a hand-rendered page.
"""


import hashlib
import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any

from paa.config import FIXTURE_DIR
from paa.identity import name_key
from paa.store import add_manifest, connect

DEFAULT_FIXTURE = FIXTURE_DIR / "frozen_sources.jsonl"
DEFAULT_ROOT = Path("dist/frozen")


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").split('\n'), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:  # pragma: no cover - fixture corruption guard
            raise ValueError(f"invalid frozen fixture JSON at {path}:{line_number}") from exc
        if not isinstance(row, dict) or not isinstance(row.get("kind"), str):
            raise TypeError(f"frozen fixture row {line_number} must have a string kind")
        rows.append(row)
    return rows


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalise_document(row: dict[str, Any]) -> dict[str, Any]:
    item = dict(row)
    item["sha256"] = item.get("sha256") or _digest(str(item.get("text") or ""))
    item["actor_id"] = str(item.get("actor_id") or "")
    item["stated_earliest"] = item.get("stated_earliest") or None
    return item


def _normalise_candidacy(row: dict[str, Any]) -> dict[str, Any]:
    item = dict(row)
    item["name_key"] = item.get("name_key") or name_key(item.get("first_name", ""), item.get("last_name", ""))
    item["district_code"] = str(item.get("district_code") or "")
    item["district_abbr"] = str(item.get("district_abbr") or "")
    item["actor_id"] = None
    item["valintatieto"] = str(item.get("valintatieto") or "")
    return item


def _insert_fixture_row(conn: sqlite3.Connection, kind: str, row: dict[str, Any]) -> None:
    if kind == "manifest":
        # Manifest entries are handled separately so the fixture itself is
        # also represented as an immutable source artifact.
        return
    if kind == "candidacy":
        item = _normalise_candidacy(row)
        conn.execute(
            """INSERT OR REPLACE INTO candidacies(
                candidacy_id, election_year, district_code, district_abbr, party,
                candidate_number, first_name, last_name, name_key, age,
                occupation, home_municipality, votes, elected, valintatieto, actor_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                item.get("candidacy_id"), item.get("election_year"), item.get("district_code"),
                item.get("district_abbr"), item.get("party"), item.get("candidate_number"),
                item.get("first_name"), item.get("last_name"), item.get("name_key"), item.get("age"),
                item.get("occupation"), item.get("home_municipality"), item.get("votes", 0),
                int(bool(item.get("elected"))), item.get("valintatieto"), item.get("actor_id"),
            ),
        )
        return
    if kind == "mp_person":
        item = dict(row)
        item["name_key"] = item.get("name_key") or name_key(item.get("first_name", ""), item.get("last_name", ""))
        conn.execute(
            """INSERT OR REPLACE INTO mp_people(
                person_id, first_name, last_name, name_key, birth_year, death_date,
                ended_date, minister, json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                item.get("person_id"), item.get("first_name"), item.get("last_name"), item.get("name_key"),
                item.get("birth_year"), item.get("death_date"), item.get("ended_date"),
                int(bool(item.get("minister"))),
                item.get("json") if isinstance(item.get("json"), str) else json.dumps(item.get("json") or {}, ensure_ascii=False),
            ),
        )
        return
    if kind == "mp_period":
        item = dict(row)
        conn.execute(
            """INSERT INTO mp_periods(person_id, kind, label, start_date, end_date, precision)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (item.get("person_id"), item.get("kind"), item.get("label"), item.get("start_date"), item.get("end_date"), item.get("precision") or "unknown"),
        )
        return
    if kind == "document":
        item = _normalise_document(row)
        conn.execute(
            """INSERT OR REPLACE INTO documents(
                document_id, source_id, url, actor_id, field_label, language,
                text, stated_earliest, sha256, http_status, retrieved_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                item.get("document_id"), item.get("source_id"), item.get("url"), item.get("actor_id"),
                item.get("field_label"), item.get("language") or "fi", item.get("text") or "",
                item.get("stated_earliest"), item.get("sha256"), int(item.get("http_status") or 200),
                item.get("retrieved_at"),
            ),
        )
        return
    if kind in {"official_object", "source_coverage", "relation_review", "evidence"}:
        table = {
            "official_object": "official_objects",
            "source_coverage": "source_coverage",
            "relation_review": "relation_reviews",
            "evidence": "evidence",
        }[kind]
        key = {
            "official_object": "object_id",
            "source_coverage": "coverage_id",
            "relation_review": "review_id",
            "evidence": "evidence_id",
        }[kind]
        if not row.get(key):
            raise ValueError(f"{kind} fixture row lacks {key}")
        payload = dict(row)
        if kind == "evidence" and payload.get("kind") == "text_span":
            quote = str(payload.get("quote") or "")
            payload["text_sha256"] = _digest(quote)
            payload["span_end"] = max(int(payload.get("span_start") or 0) + len(quote), 1)
        conn.execute(
            f"INSERT OR REPLACE INTO {table}({key}, json) VALUES (?, ?)",
            (row[key], json.dumps(payload, ensure_ascii=False, sort_keys=True)),
        )
        return
    raise ValueError(f"unknown frozen fixture kind: {kind}")


def _clear_seed_tables(conn: sqlite3.Connection) -> None:
    # The database path is explicit and normally fresh. Clearing the known
    # source tables makes a rerun deterministic when callers opt into an
    # overwrite, while leaving schema/migration ownership to store.py.
    for table in (
        "meta", "manifest", "candidacies", "mp_people", "mp_periods", "actors", "documents",
        "vote_events", "ballots", "statements", "propositions", "findings", "relations", "events",
        "vote_records", "official_objects", "source_coverage", "relation_reviews", "evidence",
        "evidence_traces",
    ):
        conn.execute(f"DELETE FROM {table}")


def seed_frozen_database(db_path: Path, fixture_path: Path = DEFAULT_FIXTURE, *, overwrite: bool = False) -> dict[str, int | str]:
    """Import the immutable frozen source slice into ``db_path``.

    The importer performs no semantic matching.  In particular, official
    objects and relation reviews are source records for the compiler; they do
    not get silently promoted to a relation or finding here.
    """

    db_path = Path(db_path)
    fixture_path = Path(fixture_path)
    if db_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite frozen database: {db_path}")
    rows = _read_rows(fixture_path)
    conn = connect(db_path)
    try:
        _clear_seed_tables(conn)
        manifests = [row for row in rows if row["kind"] == "manifest"]
        fixture_digest = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
        add_manifest(
            conn,
            source_id="SRC-FROZEN-FIXTURE",
            url=f"fixture://{fixture_path.name}",
            sha256=fixture_digest,
            bytes=fixture_path.stat().st_size,
            http_status=200,
            retrieved_at=None,
            note="Immutable offline PAA E2E source slice.",
        )
        for manifest in manifests:
            # A fixture checksum identifies this JSONL bundle; it is not a
            # substitute for the hash of the source body represented by a
            # manifest row.  Real frozen slices carry the latter explicitly
            # (and may also carry a hash for the extracted source span).
            raw_sha256 = manifest.get("raw_sha256") or manifest.get("sha256") or fixture_digest
            raw_bytes = manifest.get("raw_bytes") or manifest.get("bytes") or fixture_path.stat().st_size
            note = manifest.get("note") or ""
            if manifest.get("slice_sha256"):
                note = f"{note} source_slice_sha256={manifest['slice_sha256']}".strip()
            if manifest.get("slice_locator"):
                note = f"{note} source_slice_locator={manifest['slice_locator']}".strip()
            add_manifest(
                conn,
                source_id=manifest["source_id"],
                url=manifest.get("url") or "",
                sha256=raw_sha256,
                bytes=raw_bytes,
                http_status=200,
                retrieved_at=manifest.get("retrieved_at"),
                note=note,
            )
        counts: dict[str, int | str] = {"fixture_sha256": fixture_digest}
        for row in rows:
            if row["kind"] == "manifest":
                continue
            _insert_fixture_row(conn, row["kind"], row.get("row") or {})
            counts[row["kind"]] = int(counts.get(row["kind"], 0)) + 1
        conn.commit()
    finally:
        conn.close()
    return counts


def build_frozen(
    root: Path | str = DEFAULT_ROOT,
    *,
    fixture_path: Path | str = DEFAULT_FIXTURE,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run the normal compiler/site build against a clean offline fixture.

    ``root`` is a build root, not the repository root.  This lets a clean
    clone test build into a temporary directory without mutating checkout
    state.  The returned dictionary contains the paths and compiler stats so
    callers can inspect the produced artifacts without relying on global
    configuration constants.
    """

    root = Path(root).resolve()
    fixture_path = Path(fixture_path).resolve()
    data_dir = root / "data"
    db_path = data_dir / "paa.sqlite"
    export_dir = data_dir / "export"
    report_dir = root / "reports"
    output_dir = root / "dist" / "browser"
    if root.exists() and not root.is_dir():
        raise NotADirectoryError(root)
    root.mkdir(parents=True, exist_ok=True)
    if overwrite:
        # These are build outputs below the caller-selected root.  Keep the
        # deletion narrow and explicit; no repository/global path is touched.
        if db_path.exists():
            db_path.unlink()
        for generated in (export_dir, report_dir, output_dir):
            if generated.exists():
                shutil.rmtree(generated)
    seed_stats = seed_frozen_database(db_path, fixture_path, overwrite=overwrite)
    from paa.frozen_actions import import_action_fixtures

    with connect(db_path) as conn:
        action_stats = import_action_fixtures(conn)
    from paa.pipeline import compile_all

    compiler_stats = compile_all(
        db_path=db_path,
        output_dir=output_dir,
        export_dir=export_dir,
        report_dir=report_dir,
        load_local=False,
    )
    benchmark = _write_relation_benchmark(db_path, report_dir)
    return {
        "root": str(root),
        "db_path": str(db_path),
        "output_dir": str(output_dir),
        "export_dir": str(export_dir),
        "report_dir": str(report_dir),
        "fixture": str(fixture_path),
        "seed": seed_stats,
        "action_sources": action_stats,
        "compile": compiler_stats,
        "benchmark": benchmark,
    }


def _write_relation_benchmark(db_path: Path, report_dir: Path) -> dict[str, Any]:
    """Write a transparent retrieval/review sanity report for the fixture.

    This is deliberately not a population-quality claim.  It reports only
    the two declared review examples in ``frozen_sources.jsonl``: one
    admitted same-object relation and one intentionally unresolved broad
    topic relation.  The report makes retrieval and admission behavior
    inspectable without pretending that two examples are a held-out benchmark.
    """

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        reviews = [json.loads(row["json"]) for row in conn.execute("SELECT json FROM relation_reviews")]
        packets = {
            row["proposition_id"]: json.loads(row["json"])
            for row in conn.execute("SELECT proposition_id, json FROM evidence_traces")
        }
    finally:
        conn.close()
    gold = [review for review in reviews if review.get("status") in {"SAME_POLICY_OBJECT", "SAME_MATTER"}]
    top_k: dict[str, float] = {}
    review_reference_injections = 0
    candidate_ids_by_review = {}
    for review in gold:
        packet = packets.get(review.get("proposition_id"), {})
        candidates = []
        for item in packet.get("retrieved_objects", []):
            retrieval = item.get("retrieval") or {}
            if retrieval.get("method") == "review-reference":
                review_reference_injections += 1
                continue
            candidates.append(item)
        candidate_ids_by_review[review.get("review_id")] = [item.get("object_id") for item in candidates]
    for k in (1, 3, 5):
        found = 0
        for review in gold:
            ids = candidate_ids_by_review.get(review.get("review_id"), [])
            if review.get("object_id") in ids[:k]:
                found += 1
        top_k[str(k)] = found / len(gold) if gold else 0.0
    admitted = []
    for packet in packets.values():
        admitted.extend(
            item for item in packet.get("relations", [])
            if item.get("validation_state") == "VALID"
            and item.get("status") in {"SAME_POLICY_OBJECT", "SAME_MATTER"}
        )
    true_admitted = sum(item.get("object_id") in {review.get("object_id") for review in gold} for item in admitted)
    unresolved = sum(review.get("status") == "UNRESOLVED" for review in reviews)
    from paa.initiative_benchmark import build_benchmark

    initiative_benchmark = build_benchmark()
    report = {
        "schema_version": "1.0",
        "evaluation_scope": "frozen_fixture_only_not_held_out",
        "gold_reviewed_relations": len(gold),
        "review_count": len(reviews),
        "top_k_recall": top_k,
        "admitted_relation_count": len(admitted),
        "admitted_relation_precision": true_admitted / len(admitted) if admitted else 0.0,
        "abstention_count": unresolved,
        "abstention_rate": unresolved / len(reviews) if reviews else 0.0,
        "review_reference_injections_excluded_from_recall": review_reference_injections,
        "initiative_benchmark": initiative_benchmark,
        "notes": [
            "Literal retrieval is a candidate generator; recall excludes objects injected by a version-bound review.",
            "The frozen DB examples and the initiative pair set are illustrative, not held-out evidence of general retrieval quality.",
        ],
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "relation_benchmark.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


__all__ = ["DEFAULT_FIXTURE", "build_frozen", "seed_frozen_database"]
