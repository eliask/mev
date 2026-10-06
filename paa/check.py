"""Check source reconciliation and canonical evidence-packet invariants."""

import json
import sqlite3
from pathlib import Path

from paa.config import CORPUS_CUTOFF, DB_PATH
from paa.records import evidence_reference_ids


def ballot_discrepancies(conn: sqlite3.Connection) -> list[dict]:
    """Reconcile person rows and all published response buckets."""
    return [dict(row) for row in conn.execute(
        """SELECT v.aanestys_id, v.yhteensa AS published, COUNT(b.person_number) AS stored,
                  v.jaa, v.ei, v.tyhjaa, v.poissa,
                  COALESCE(SUM(b.raw_response='JAA'),0) AS observed_jaa,
                  COALESCE(SUM(b.raw_response='EI'),0) AS observed_ei,
                  COALESCE(SUM(b.raw_response='TYHJA'),0) AS observed_tyhjaa,
                  COALESCE(SUM(b.raw_response='POISSA'),0) AS observed_poissa
           FROM vote_events v LEFT JOIN ballots b USING(aanestys_id)
           WHERE v.year BETWEEN 2023 AND 2026 GROUP BY v.aanestys_id
           HAVING stored != published OR observed_jaa != v.jaa OR observed_ei != v.ei
                  OR observed_tyhjaa != v.tyhjaa OR observed_poissa != v.poissa
           ORDER BY v.aanestys_id""")]


def run(db_path: Path | None = None, full_corpus: bool = True) -> list[str]:
    database = db_path or DB_PATH
    if not database.exists():
        return ["database missing; run paa frozen or acquire sources then compile"]
    conn = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    problems = []
    try:
        if full_corpus:
            for year, elected in conn.execute("SELECT election_year, SUM(elected) FROM candidacies GROUP BY 1"):
                if elected != 200:
                    problems.append(f"{year} elected {elected}, expected 200")
            n = conn.execute("SELECT COUNT(*) FROM candidacies WHERE election_year=2023").fetchone()[0]
            if n != 2424:
                problems.append(f"2023 candidates {n}, expected 2424")
            missing = conn.execute("""SELECT COUNT(*) FROM vote_events v WHERE v.year BETWEEN 2023 AND 2026
                                     AND NOT EXISTS(SELECT 1 FROM ballots b WHERE b.aanestys_id=v.aanestys_id)""").fetchone()[0]
            if missing:
                problems.append(f"{missing} current-term votes have no ballots")
            for mismatch in ballot_discrepancies(conn):
                problems.append(f"vote {mismatch['aanestys_id']}: person rows or response buckets do not match published totals")
        evidence = {r["evidence_id"] for r in conn.execute("SELECT evidence_id FROM evidence")}
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='group_contexts'").fetchone():
            from paa.group_agreement import build_group_sources, validate_group_packets

            group_packets = [json.loads(row["json"]) for row in conn.execute("SELECT json FROM group_contexts")]
            group_sources = [json.loads(row["json"]) for row in conn.execute("SELECT json FROM group_sources")]
            if group_packets or group_sources:
                validation = validate_group_packets(group_packets, group_sources)
                problems.extend("group context: " + error for error in validation["errors"])
            current_sources = {source["source_id"]: source for source in build_group_sources(conn)}
            compiled_sources = {source["source_id"]: source for source in group_sources}
            if current_sources.keys() != compiled_sources.keys():
                problems.append("group context: source-event set changed; recompile")
            else:
                for source_id, source in current_sources.items():
                    if source["source_sha256"] != compiled_sources[source_id].get("source_sha256"):
                        problems.append(f"group context: {source_id} changed; recompile")
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='inquiry_cases'").fetchone():
            from paa.inquiry_cases import InquiryError, validate_case
            from paa.legal_inquiry import LegalInquiryError, validate_legal_inquiry

            for row in conn.execute("SELECT json FROM inquiry_cases"):
                case = json.loads(row["json"])
                try:
                    validate_case(case)
                    if case.get("legal_inquiry"):
                        validate_legal_inquiry(case)
                except (InquiryError, LegalInquiryError) as error:
                    problems.append(f"{case.get('case_id')}: {error}")
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='decision_episodes'").fetchone():
            from paa.decision_episodes import EpisodeError, validate_episode

            source_evidence = {r["evidence_id"]: json.loads(r["json"]) for r in conn.execute("SELECT * FROM evidence")}
            source_objects = {r["object_id"]: json.loads(r["json"]) for r in conn.execute("SELECT * FROM official_objects")}
            for object_id, obj in source_objects.items():
                if not evidence_reference_ids(obj) <= evidence:
                    problems.append(f"{object_id}: official object has dangling evidence IDs")
                if obj.get("action_date") and obj["action_date"] > CORPUS_CUTOFF:
                    problems.append(f"{object_id}: official action after the declared snapshot")
            for row in conn.execute("SELECT json FROM decision_episodes"):
                episode = json.loads(row["json"])
                try:
                    validate_episode(episode, evidence={key: source_evidence[key] for key in episode["evidence_ids"] if key in source_evidence})
                except EpisodeError as error:
                    problems.append(f"{episode['episode_id']}: {error}")
                for source in episode["source_objects"]:
                    current = source_objects.get(source["object_id"])
                    if current is None or any(source.get(key) != current.get(key) for key in ("matter_id", "text", "authors", "action_date", "action_date_basis")):
                        problems.append(f"{episode['episode_id']}: episode source object changed; recompile")
        n_traces = 0
        from paa.opportunity import opportunity_requirements
        from paa.opportunity_codec import OpportunityCodecError, decode_plan, plan_to_wire

        for row in conn.execute("SELECT json FROM evidence_traces"):
            packet = json.loads(row["json"]); n_traces += 1
            key = packet["trace_id"]
            if not packet.get("statement_id") or not packet.get("proposition_id"):
                problems.append(f"{key}: missing origin")
            plan_wire = packet["authority"].get("verification_plan")
            try:
                plan = decode_plan(json.dumps(plan_wire, ensure_ascii=False))
                current_prop = conn.execute("SELECT json FROM propositions WHERE proposition_id=?",
                                            (packet.get("proposition_id"),)).fetchone()
                if current_prop is None or plan_to_wire(opportunity_requirements(json.loads(current_prop["json"]))) != plan_to_wire(plan):
                    problems.append(f"{key}: action requirements changed; recompile")
            except OpportunityCodecError as error:
                problems.append(f"{key}: invalid retained action requirements: {error}")
            local = {ref["evidence_id"] for ref in packet["evidence"]}
            # The current trace is self-contained. Historical assessment
            # fingerprints retain older references in the database ledger.
            current = {key: value for key, value in packet.items() if key not in {"evidence", "correction_history"}}
            refs = evidence_reference_ids(current)
            if not refs <= local or not refs <= evidence:
                problems.append(f"{key}: dangling evidence IDs")
            if packet["assessment"]["state"] == "OBSERVED_ALIGNED_ACTION" and not packet["actions"]:
                problems.append(f"{key}: aligned finding has no action")
            if packet["assessment"]["state"] == "OBSERVED_ALIGNED_ACTION" and (
                packet["target"]["interpretation_state"] != "REVIEWED" or not packet["target"]["normalized_object"]
            ):
                problems.append(f"{key}: aligned finding has no reviewed target")
            for action in packet["actions"]:
                obj = action["object_id"]
                source_obj = source_objects.get(obj, {})
                if action.get("state") == "OBSERVED_ALIGNED_ACTION" and action.get("kind") == "LEGISLATIVE_INITIATIVE" and (
                    action.get("date_basis") != "VIREILLETULO_EVENT"
                    or source_obj.get("date_binding_state") == "SIGNATURE_AFTER_FILING"
                ):
                    problems.append(f"{key}: initiative fulfillment lacks a source-bound filing date")
                valid = [r for r in packet["relations"] if r["object_id"]==obj and r["status"] in {"SAME_POLICY_OBJECT", "SAME_MATTER"} and r.get("validation_state")=="VALID"]
                if not valid:
                    problems.append(f"{key}: action has no verified relation")
                if not action["evidence_ids"] or not set(action["evidence_ids"]) <= local:
                    problems.append(f"{key}: ungrounded action")
                window = packet["authority"]["window"]
                if not window["earliest"] or not (window["earliest"] < action["date"] <= window["latest"]):
                    problems.append(f"{key}: action outside window")
            if packet["assessment"]["causal_claim"]:
                problems.append(f"{key}: unexpected causal admission")
        if not n_traces:
            problems.append("no compiled evidence traces; run paa compile")
    except sqlite3.OperationalError as error:
        problems.append(f"database needs compilation: {error}")
    finally:
        conn.close()
    return problems


def main(db_path: Path | None = None, *, full_corpus: bool = True) -> int:
    problems = run(db_path=db_path, full_corpus=full_corpus)
    for problem in problems:
        print("FAIL", problem)
    if not problems:
        print("ok")
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
