"""Source-only action candidate inventory for the full campaign corpus.

Retrieval narrows to the identified person's acquired actions after the
statement's latest date. This is a documented-action discovery plan, never
a proof of register absence, pledge fulfillment, or complete source coverage.
"""


import argparse
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from paa.llm_overlay import ModelInterpretations
from paa.records import proposition_record
from paa.relations import ObjectRetriever
from paa.semantics import analyze_text

ACTION_SOURCES = {"LEGISLATIVE_INITIATIVE", "WRITTEN_QUESTION", "SPEECH", "RESIGN_ROLE", "VOTE"}
SUBSTANTIVE_TYPES = {"PERSONAL_ACTION_COMMITMENT", "PERSONAL_RESTRAINT_COMMITMENT", "POSITION",
                     "POLICY_DESIDERATUM", "BROAD_OBJECTIVE", "OUTCOME_COMMITMENT", "COLLECTIVE_ACTION_COMMITMENT"}


def inventory(db: Path, run: Path, output: Path, *, k: int = 3) -> dict:
    if k < 1:
        raise ValueError("top_k must be positive")
    conn = sqlite3.connect(f"file:{db.resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    statements = {row["statement_id"]: json.loads(row["json"]) for row in conn.execute("SELECT * FROM statements")}
    actors = {row["actor_id"]: dict(row) for row in conn.execute("SELECT * FROM actors")}
    object_rows = [json.loads(row["json"]) for row in conn.execute("SELECT json FROM official_objects")]
    by_person = defaultdict(list)
    for obj in object_rows:
        if obj["kind"] not in ACTION_SOURCES or obj.get("void"):
            continue
        persons = {str(author["person_id"]) for author in obj.get("authors", []) if author.get("person_id")}
        for person in persons:
            by_person[person].append(obj)
    interpret = ModelInterpretations(run)
    counts = Counter()
    pair_kinds = Counter()
    rows = []
    retrievers = {}
    accounting = []
    for doc_row in conn.execute("SELECT * FROM documents ORDER BY document_id"):
        doc = dict(doc_row)
        counts["documents_total"] += 1
        receipt_path = run / "documents" / (doc["document_id"] + ".json")
        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
        receipt_state = receipt.get("receipt_status", "MISSING") if receipt else "MISSING"
        counts[f"model_receipts_{receipt_state}"] += 1
        statement = statements[doc["document_id"]]
        ids = statement["issuer_actor_ids"]
        actor = actors.get(ids[0]) if len(ids) == 1 else None
        if not actor or not actor.get("person_id") or actor["identity_status"] != "MP_UNIQUE":
            counts["documents_without_verified_official_person"] += 1
            accounting.append({"document_id": doc["document_id"], "state": "NO_VERIFIED_PERSON_ID", "absence_claim": False})
            continue
        if receipt is None:
            counts["documents_awaiting_model_pass"] += 1
            accounting.append({"document_id": doc["document_id"], "state": "AWAITING_MODEL_PASS", "absence_claim": False})
            continue
        if receipt_state != "OK":
            counts["documents_without_successful_model_receipt"] += 1
            accounting.append({"document_id": doc["document_id"], "state": "MODEL_RECEIPT_FAILED",
                               "receipt_status": receipt_state, "absence_claim": False})
            continue
        counts["documents_with_model_and_official_person"] += 1
        original = analyze_text(doc["text"], {"field_label": doc["field_label"], "stated_earliest": doc["stated_earliest"]}).propositions
        applied_before, abstained_before = interpret.applied, interpret.abstained
        props = interpret.propositions(doc, original)
        latest = statement["stated_at"].get("latest")
        if not latest:
            counts["documents_without_statement_upper_date"] += 1
            accounting.append({"document_id": doc["document_id"], "state": "STATEMENT_DATE_UNRESOLVED", "absence_claim": False})
            continue
        person = str(actor["person_id"])
        key = (person, latest)
        if key not in retrievers:
            eligible = [o for o in by_person[person] if o.get("action_date") and latest < o["action_date"] <= "2026-10-06"]
            retrievers[key] = ObjectRetriever(eligible)
        retriever = retrievers[key]
        doc_pairs = 0
        for index, prop in enumerate(props, 1):
            counts["units_of_identified_person"] += 1
            if prop.semantic_type not in SUBSTANTIVE_TYPES:
                counts["units_without_substantive_action_query"] += 1
                continue
            prop_id = f"{doc['document_id']}-p{index}"
            record = proposition_record(prop, statement["statement_id"], statement["evidence"][0]["evidence_id"], ids, prop_id)
            record["run_id"] = interpret.run_id
            hits = retriever.search(" ".join(prop.targets) or prop.text, k=k)
            if not hits:
                counts["units_without_retrieved_candidates"] += 1
            for rank, hit in enumerate(hits, 1):
                obj = retriever.objects[hit["object_id"]]
                pair_id = "corpus-pair:" + hashlib.sha256((prop_id + ":" + obj["object_id"]).encode()).hexdigest()[:24]
                # Raw register rows and alternate source records remain in the
                # database. Preserve the complete selected text, its hashes and
                # evidence here without repeating unrelated parsing payloads.
                public_object = {key: value for key, value in obj.items()
                                 if key not in {"source_row", "source_records", "selection"}}
                rows.append({"pair_id": pair_id, "proposition": record, "statement": statement, "official_object": public_object,
                    "retrieval": {**hit, "rank": rank, "scope": "identified-person acquired actions after latest statement date",
                                  "absence_claim": False}, "actor": actor})
                pair_kinds[obj["kind"]] += 1
                doc_pairs += 1
        accounting.append({"document_id": doc["document_id"], "state": "CANDIDATE_RETRIEVAL_COMPLETED",
                           "pair_count": doc_pairs, "source_validated_units": interpret.applied - applied_before,
                           "model_unit_failures": interpret.abstained - abstained_before, "absence_claim": False})
    conn.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    report = {"scope": "All campaign documents accounted; own-action retrieval only where person, dates and local interpretations exist",
        "counts": dict(counts), "pair_count": len(rows), "pair_kinds": dict(pair_kinds), "top_k": k,
        "official_object_snapshot_sha256": hashlib.sha256(json.dumps(object_rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        "model_run_identity": interpret.manifest["identity"], "absence_claim": False,
        "limitations": ["Top-k retrieval does not establish missing action.",
                        "Acquired source slices and review quality bound every possible conclusion.",
                        "Own-action candidate scope is narrower than general policy-object retrieval."]}
    output.with_suffix(".coverage.json").write_text(json.dumps({**report, "documents": accounting}, ensure_ascii=False, indent=2))
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/paa.sqlite"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args(argv)
    print(json.dumps(inventory(args.db, args.run, args.output, k=args.top_k), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
