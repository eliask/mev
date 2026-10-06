"""Join rosters, compile statements, and write the static browser."""


import json
import sqlite3
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from paa.acquire_eduskunta import serves_term
from paa.acquire_yle import load_2011_open_answers, relink_local, restore_cached_capture_dates
from paa.config import CORPUS_CUTOFF, EXPORT, FIXTURE_DIR, REPORTS, RUN_ID, TERM_START, ensure_dirs
from paa.identity import Candidacy, Member, cluster_candidacies, match_members, name_key
from paa.records import (
    alternatives_from_title,
    event_record,
    evidence_reference_ids,
    finding_record,
    proposition_record,
    statement_record,
    validate,
    vote_record,
)
from paa.semantics import analyze_text
from paa.site import build_from_db
from paa.store import connect, put_meta
from paa.traces import compile_traces

CLAIMS = {
    "VALUE_OR_SLOGAN": "Teksti on iskulause tai arvolause. Kentän nimi ei tee siitä henkilökohtaista lupausta.",
    "BROAD_OBJECTIVE": "Teksti on laaja tavoite. Siitä ei ole luettu mittaria, määrää tai määräaikaa, jota siinä ei ole.",
    "POSITION": "Teksti on kanta. Kannatus ei ole lupaus tietystä omasta teosta.",
    "PROCESS_COMMITMENT": "Teksti kuvaa tapaa toimia. Se ei ole mekaanisesti tarkistettava lopputulos.",
    "PERSONAL_ACTION_COMMITMENT": "Teksti nimeää oman teon. Toteutumista ei päätellä ilman vastaavaa asiakirjaa.",
    "PERSONAL_RESTRAINT_COMMITMENT": "Teksti sitoutuu pidättäytymään omasta teosta. Koko aikaväliä ei ratkaista yksittäisestä asiakirjasta.",
    "POLICY_DESIDERATUM": "Teksti on passiivinen toive tai vaatimus, ei henkilön yksipuolinen toimintalupaus.",
    "COLLECTIVE_ACTION_COMMITMENT": "Teksti on joukon sitoumus. Sitä ei ole siirretty henkilön omaksi lupaukseksi.",
    "OUTCOME_COMMITMENT": "Teksti sitoutuu lopputulokseen. Toimintarekisteri ei yksin ratkaise toteutumista tai henkilön vaikutusta.",
    "MAINTAIN_COMMITMENT": "Teksti sitoutuu säilyttämään tilan. Ajallinen tila ja henkilön toiminta on tarkistettava erikseen.",
    "PREVENT_COMMITMENT": "Teksti sitoutuu estämään muutoksen. Teot, muutoksen toteutuminen ja toimivalta ovat eri kysymyksiä.",
    "CAUSAL_EFFECT_FORECAST": "Teksti arvioi seurauksen. Ennen–jälkeen-ero ei kumoa eikä vahvista arviota.",
    "REPORTED_SPEECH": "Lainattu virke kuuluu toiselle puhujalle.",
    "AMBIGUOUS": "Tekstin lajia ei ratkaistu.",
}



def _actors(conn: sqlite3.Connection) -> dict:
    rows = [
        Candidacy(
            row["candidacy_id"],
            row["election_year"],
            row["district_code"],
            row["candidate_number"],
            row["first_name"],
            row["last_name"],
            row["age"],
        )
        for row in conn.execute("SELECT * FROM candidacies")
    ]
    members = [
        Member(row["person_id"], row["first_name"], row["last_name"], row["birth_year"])
        for row in conn.execute("SELECT * FROM mp_people")
    ]
    assignment = cluster_candidacies(rows)
    grouped: dict[str, list[Candidacy]] = defaultdict(list)
    by_id = {row.candidacy_id: row for row in rows}
    for candidacy_id, cluster_id in assignment.items():
        grouped[cluster_id].append(by_id[candidacy_id])
    matched = match_members(grouped, members)
    conn.execute("DELETE FROM actors")
    used = set()
    for cluster_id, group in grouped.items():
        actor_id, status = matched[cluster_id]
        person_id = actor_id.removeprefix("mp-") if actor_id.startswith("mp-") else None
        display = f"{group[0].first} {group[0].last}".strip()
        birth = None
        if person_id:
            member = conn.execute("SELECT * FROM mp_people WHERE person_id = ?", (person_id,)).fetchone()
            if member:
                display = f"{member['first_name']} {member['last_name']}".strip()
                birth = member["birth_year"]
        conn.execute(
            """INSERT OR REPLACE INTO actors(actor_id, display_name, name_key, birth_year, person_id, identity_status)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (actor_id, display, name_key(group[0].first, group[0].last), birth, person_id, status),
        )
        used.add(actor_id)
        for row in group:
            conn.execute(
                "UPDATE candidacies SET actor_id = ? WHERE candidacy_id = ?",
                (actor_id, row.candidacy_id),
            )
    current = 0
    for member in conn.execute("SELECT * FROM mp_people"):
        periods = [dict(row) for row in conn.execute("SELECT * FROM mp_periods WHERE person_id = ?", (member["person_id"],))]
        if not serves_term(periods, member["ended_date"]):
            continue
        current += 1
        actor_id = f"mp-{member['person_id']}"
        if actor_id in used:
            continue
        conn.execute(
            """INSERT OR REPLACE INTO actors(actor_id, display_name, name_key, birth_year, person_id, identity_status)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                actor_id,
                f"{member['first_name']} {member['last_name']}".strip(),
                member["name_key"],
                member["birth_year"],
                member["person_id"],
                "MP_RECORD_ONLY",
            ),
        )
    return {"clusters": len(grouped), "current_term_mps": current}


def _compile_documents(conn: sqlite3.Connection, model_interpretations=None) -> dict:
    conn.execute("DELETE FROM statements")
    conn.execute("DELETE FROM propositions")
    conn.execute("DELETE FROM findings")
    statements = 0
    propositions = 0
    policy = 0
    types = Counter()
    # The compiler consumes the declared local document slice rather than a
    # hard-coded production source allow-list.  Acquisition still controls
    # what enters the DB; this is what lets an offline/frozen source packet
    # exercise the same statement -> proposition -> trace path.
    for doc in conn.execute(
        "SELECT * FROM documents WHERE http_status = 200 AND text IS NOT NULL ORDER BY document_id"
    ):
        document = dict(doc)
        if document["source_id"] == "SRC-YLE-2011":
            document["stated_latest"] = "2011-04-17"
        elif document["source_id"].startswith("SRC-YLE-2023"):
            document["stated_latest"] = "2023-04-02"
        hint = document["actor_id"] or ""
        actor_ids: list[str] = []
        basis = "UNRESOLVED"
        if hint:
            row = conn.execute("SELECT actor_id FROM candidacies WHERE candidacy_id = ?", (hint,)).fetchone()
            if row and row["actor_id"]:
                actor_ids = [row["actor_id"]]
                basis = "UNRESOLVED" if row["actor_id"].startswith("amb-") else "EXPLICIT"
        evidence_id = document["document_id"] + "-e"
        statement = statement_record(document, evidence_id, actor_ids, basis)
        validate("statement", statement)
        analysis = analyze_text(document["text"], {"field_label": document["field_label"],
            "stated_earliest": document["stated_earliest"], "stated_latest": document.get("stated_latest")})
        if model_interpretations is not None:
            analysis.propositions = model_interpretations.propositions(document, analysis.propositions)
        conn.execute("INSERT INTO statements(statement_id, json) VALUES (?, ?)", (statement["statement_id"], json.dumps(statement, ensure_ascii=False)))
        statements += 1
        prop_ids = []
        for index, prop in enumerate(analysis.propositions, start=1):
            proposition_id = f"{document['document_id']}-p{index}"
            record = proposition_record(prop, statement["statement_id"], evidence_id, actor_ids, proposition_id)
            if model_interpretations is not None:
                record["run_id"] = model_interpretations.run_id
                record["predicate"]["basis"] = "ANALYST_INTERPRETATION"
                record["predicate"]["note"] = ("Local model interpretation; source-checked but not independently admitted. "
                    "Receipt: " + str(model_interpretations.receipts.get(document["document_id"])))
            validate("proposition", record)
            conn.execute(
                "INSERT INTO propositions(proposition_id, statement_id, json) VALUES (?, ?, ?)",
                (proposition_id, statement["statement_id"], json.dumps(record, ensure_ascii=False)),
            )
            prop_ids.append(proposition_id)
            propositions += 1
            types[prop.semantic_type] += 1
            if prop.semantic_type == "POLICY_DESIDERATUM":
                policy += 1
        primary = next((prop for prop in analysis.propositions if not prop.reported_speech), None)
        if primary is None:
            continue
        temporal = "NO_DETERMINABLE_DEADLINE"
        if primary.deadline and primary.deadline > CORPUS_CUTOFF:
            temporal = "NOT_DUE"
        elif primary.deadline:
            temporal = "DUE"
        # Rule coverage in unit tests is not an estimate of corpus-wide semantic precision.
        admitted = False
        finding = finding_record(
            finding_id=document["document_id"] + "-f",
            finding_type="EPISTEMIC",
            proposition_ids=prop_ids,
            actor_ids=actor_ids,
            as_of=CORPUS_CUTOFF,
            temporal_state=temporal if primary.semantic_type == "PERSONAL_ACTION_COMMITMENT" else "NO_DETERMINABLE_DEADLINE",
            condition_state="UNRESOLVED" if primary.condition else "NOT_APPLICABLE",
            target_state="NOT_APPLICABLE" if primary.semantic_type in {"VALUE_OR_SLOGAN", "POSITION"} else "UNKNOWN",
            action_congruence="NOT_APPLICABLE" if not primary.personal_action_commitment else "INSUFFICIENT_EVIDENCE",
            attribution="ACTOR_CONTROLLED_ACTION" if primary.personal_action_commitment else "UNASSIGNED",
            supported_claim=CLAIMS.get(primary.semantic_type, CLAIMS["AMBIGUOUS"]),
            support_evidence_ids=[evidence_id] if admitted else [],
            limitations=[
                "Luokitus koskee tekstin lajia, ei myöhempää käyttäytymistä.",
                "Keskeneräinen vaalikausi ei ole päättynyt lupaushorisontti.",
            ],
            admission_state="ADMITTED" if admitted else "CANDIDATE",
            admission_route="DETERMINISTIC_VALIDATION" if admitted else "NONE",
            validation_artifact_ids=["tests/test_paa_semantics.py"] if admitted else [],
        )
        validate("finding", finding)
        conn.execute("INSERT INTO findings(finding_id, json) VALUES (?, ?)", (finding["finding_id"], json.dumps(finding, ensure_ascii=False)))
    return {"statements": statements, "propositions": propositions, "policy_desiderata": policy,
            "semantic_types": dict(types)}


def _export(conn: sqlite3.Connection, export_dir: Path | None = None) -> dict:
    destination = export_dir or EXPORT
    destination.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, table in (
        ("statements", "statements"),
        ("propositions", "propositions"),
        ("findings", "findings"),
        ("evidence_traces", "evidence_traces"),
        ("decision_episodes", "decision_episodes"),
        ("inquiry_cases", "inquiry_cases"),
        ("group_contexts", "group_contexts"),
        ("group_sources", "group_sources"),
        ("evidence", "evidence"),
        ("official_objects", "official_objects"),
        ("source_coverage", "source_coverage"),
        ("relation_reviews", "relation_reviews"),
    ):
        path = destination / f"{name}.jsonl"
        temporary = path.with_suffix(".jsonl.tmp")
        count = 0
        with temporary.open("w", encoding="utf-8") as output:
            for row in conn.execute(f"SELECT json FROM {table}"):
                output.write(row["json"] + "\n")
                count += 1
        temporary.replace(path)
        counts[name] = count
    events = []
    votes = []
    for vote in conn.execute("SELECT * FROM vote_events"):
        evidence_id = f"vote-{vote['aanestys_id']}"
        event = event_record(
            f"event-{vote['aanestys_id']}",
            vote["session_date"] or None,
            evidence_id,
            {
                "title": vote["title"],
                "jaa": vote["jaa"],
                "ei": vote["ei"],
                "tyhjaa": vote["tyhjaa"],
                "poissa": vote["poissa"],
                "url": vote["url"],
                "ptk": vote["ptk"],
                "matter": vote["matter"],
                "void": bool(vote["mitatoity"]),
            },
        )
        validate("event", event)
        events.append(event)
        alts = alternatives_from_title(vote["title"], evidence_id)
        if not alts:
            continue
        for ballot in ([] if len(votes) >= 400 else conn.execute(
            "SELECT * FROM ballots WHERE aanestys_id = ? LIMIT 5",
            (vote["aanestys_id"],),
        )):
            actor = conn.execute(
                "SELECT actor_id FROM actors WHERE person_id = ?",
                (ballot["person_number"],),
            ).fetchone()
            if not actor:
                continue
            record = vote_record(
                f"ballot-{vote['aanestys_id']}-{ballot['person_number']}",
                f"event-{vote['aanestys_id']}",
                actor["actor_id"],
                ballot["raw_response"],
                vote["title"],
                alts,
                [evidence_id],
                vote["matter"] or "",
            )
            validate("vote", record)
            votes.append(record)
            if len(votes) >= 400:
                break
    (destination / "events.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in events), encoding="utf-8")
    # Sample only. The full ballots stay in data/paa.sqlite. This file is not the vote ledger.
    sample_note = {
        "schema_version": "sample",
        "note": "At most 400 decoded ballots. Not a complete export.",
        "records": len(votes),
    }
    (destination / "votes_sample.jsonl").write_text(
        json.dumps(sample_note, ensure_ascii=False) + "\n"
        + "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in votes),
        encoding="utf-8",
    )
    old_votes = destination / "votes.jsonl"
    if old_votes.exists():
        old_votes.unlink()
    counts["events"] = len(events)
    counts["votes_sample"] = len(votes)
    return counts


def _coverage(conn: sqlite3.Connection) -> dict:
    from paa.check import ballot_discrepancies

    def count(sql: str, params: tuple = ()) -> int:
        return conn.execute(sql, params).fetchone()[0]

    by_year = {
        str(row["election_year"]): {"candidates": row["n"], "elected": row["elected"]}
        for row in conn.execute(
            "SELECT election_year, COUNT(*) n, SUM(elected) elected FROM candidacies GROUP BY election_year"
        )
    }
    vote_years = {
        str(row["year"]): row["n"]
        for row in conn.execute("SELECT year, COUNT(*) n FROM vote_events GROUP BY year ORDER BY year")
    }
    return {
        "candidacies_by_year": by_year,
        "vote_events_by_year": vote_years,
        "ballots": count("SELECT COUNT(*) FROM ballots"),
        "members": count("SELECT COUNT(*) FROM mp_people"),
        "actors": count("SELECT COUNT(*) FROM actors"),
        "ambiguous_identities": count("SELECT COUNT(*) FROM actors WHERE identity_status = 'AMBIGUOUS'"),
        "documents": count("SELECT COUNT(*) FROM documents"),
        "official_objects": count("SELECT COUNT(*) FROM official_objects"),
        "reviewed_relations": count("SELECT COUNT(*) FROM relation_reviews"),
        "ballot_reconciliation": ballot_discrepancies(conn),
    }


def compile_database(conn: sqlite3.Connection, llm_run: Path | None = None) -> dict:
    """One compile path for frozen and live inputs, without network acquisition."""
    actor_stats = _actors(conn)
    if llm_run is not None:
        from paa.llm_overlay import ModelInterpretations

        interpretations = ModelInterpretations(llm_run)
    else:
        interpretations = None
    text_stats = _compile_documents(conn, interpretations)
    episode_stats = _compile_episodes(conn)
    inquiry_stats = _compile_inquiries(conn)
    group_stats = _compile_groups(conn)
    trace_stats = compile_traces(conn)
    stats = {**actor_stats, **text_stats, **trace_stats, **episode_stats, **inquiry_stats, **group_stats}
    if interpretations is not None:
        stats["local_model"] = {"run_id": interpretations.run_id, "applied_units": interpretations.applied,
                                "abstained_units": interpretations.abstained, "admission_state": "PROPOSED"}
        put_meta(conn, "local_model_run", json.dumps(interpretations.manifest, ensure_ascii=False))
    put_meta(conn, "compiled_at", datetime.now(UTC).isoformat())
    conn.commit()
    return stats


def _compile_groups(conn: sqlite3.Connection) -> dict:
    from paa.group_agreement import build_group_context

    packets, sources, report = build_group_context(conn)
    conn.execute("DELETE FROM group_contexts")
    conn.execute("DELETE FROM group_sources")
    conn.executemany("INSERT INTO group_contexts VALUES (?, ?)",
                     [(p["person_id"], json.dumps(p, ensure_ascii=False)) for p in packets])
    conn.executemany("INSERT INTO group_sources VALUES (?, ?)",
                     [(s["source_id"], json.dumps(s, ensure_ascii=False)) for s in sources])
    put_meta(conn, "group_context_report", report)
    return {"group_contexts": len(packets), "group_source_events": len(sources),
            "group_source_exclusions": report["excluded_source_count"]}


def _compile_inquiries(conn: sqlite3.Connection) -> dict:
    from paa.inquiry_cases import load_reviewed_cases

    bundles = (
        ("mev_cases_source_slices.jsonl", "inquiry_case_reviews.json"),
        ("mev_transfer_source_slices.jsonl", "inquiry_transfer_case_reviews.json"),
        ("inquiry_official_source_versions.jsonl", "inquiry_official_source_reviews.json"),
        ("therapy_completion_source_versions.jsonl", "therapy_completion_case_reviews.json"),
    )
    cases = [packet for sources, reviews in bundles
             for packet in load_reviewed_cases(FIXTURE_DIR / sources, FIXTURE_DIR / reviews)]
    if len({packet["case_id"] for packet in cases}) != len(cases):
        raise ValueError("conflicting reviewed inquiry identities")
    conn.execute("DELETE FROM inquiry_cases")
    for packet in cases:
        conn.execute("INSERT INTO inquiry_cases VALUES (?, ?)", (packet["case_id"], json.dumps(packet, ensure_ascii=False)))
    return {"inquiry_cases": len(cases), "inquiry_review_basis": "AI_SOURCE_READING; not independent human adjudication"}


def _compile_episodes(conn: sqlite3.Connection) -> dict:
    """Join only formal source links; keep rejected source episodes visible."""
    from paa.decision_episodes import EpisodeError, build_written_question_episode

    objects = {r["object_id"]: json.loads(r["json"]) for r in conn.execute("SELECT * FROM official_objects")}
    evidence = {r["evidence_id"]: json.loads(r["json"]) for r in conn.execute("SELECT * FROM evidence")}
    by_locator = defaultdict(set)
    for key, ref in evidence.items():
        if ref.get("record_locator"):
            by_locator[ref["record_locator"]].add(key)
    coverage = [json.loads(r["json"]) for r in conn.execute("SELECT json FROM source_coverage")]
    coverage = [record for record in coverage if record.get("kind") == "WRITTEN_QUESTION_REGISTER"]
    actor_map = {str(row["person_id"]): {"actor_id": row["actor_id"], "identity_basis": "SOURCE_PERSON_ID"}
                 for row in conn.execute("SELECT actor_id, person_id FROM actors WHERE person_id IS NOT NULL")}
    conn.execute("DELETE FROM decision_episodes")
    states, rejected = Counter(), []
    for obj in objects.values():
        if obj["kind"] != "WRITTEN_QUESTION":
            continue
        answer = objects.get(obj.get("answer_object_id"))
        source_refs = evidence_reference_ids([obj, answer])
        for source in [obj, answer]:
            for record in (source or {}).get("source_records", []):
                source_refs.update(by_locator[record.get("record_locator")])
        episode_evidence = {key: evidence[key] for key in source_refs if key in evidence}
        try:
            episode = build_written_question_episode(obj, answer, evidence=episode_evidence, coverage=coverage,
                                                      actor_map=actor_map)
        except EpisodeError as error:
            rejected.append({"object_id": obj["object_id"], "error": str(error)})
            continue
        conn.execute("INSERT INTO decision_episodes VALUES (?, ?, ?)",
                     (episode["episode_id"], episode["matter_id"], json.dumps(episode, ensure_ascii=False)))
        states[episode["episode_state"]] += 1
    put_meta(conn, "episode_source_rejections", json.dumps(rejected, ensure_ascii=False))
    return {"decision_episodes": sum(states.values()), "episode_states": dict(states),
            "episode_source_rejections": rejected}


def compile_all(db_path: Path | None = None, output_dir: Path | None = None,
                export_dir: Path | None = None, report_dir: Path | None = None,
                load_local: bool = True, llm_run: Path | None = None) -> dict:
    if db_path is None:
        ensure_dirs()
    conn = connect(db_path)
    try:
        if load_local:
            relink_local(conn)
            load_2011_open_answers(conn)
            restore_cached_capture_dates(conn)
        stats = compile_database(conn, llm_run=llm_run)
        stats.update(_export(conn, export_dir))
        coverage = _coverage(conn)
        _write_reports(stats, coverage, report_dir)
    finally:
        conn.close()
    build_from_db(db_path=db_path, output_dir=output_dir)
    print("compiled", stats)
    return stats


def _write_reports(stats: dict, coverage: dict, report_dir: Path | None = None) -> None:
    destination = report_dir or REPORTS
    destination.mkdir(parents=True, exist_ok=True)
    report = {"run_id": RUN_ID, "cutoff": CORPUS_CUTOFF, "term_start": TERM_START,
              "status": "PARTIAL", "stats": stats, "coverage": coverage,
              "admission": "Version-bound source review; lexical retrieval is candidate-only.",
              "limitations": ["Initiative ingestion and reviewed relations cover declared slices, not every promise.",
                              "No causal effects, honesty scores or institutional-outcome guarantees."]}
    (destination / "environment.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (destination / "acceptance.md").write_text(
        "# PARTIAL\n\nCanonical evidence packets now join source statements, propositions, candidate objects, "
        "version-bound relation reviews, role windows, documented actions and bounded findings. "
        "Corpus size is not acceptance. Consult environment.json, relation_benchmark.json and the execution report "
        "for declared source slices and measured validation.\n", encoding="utf-8")
