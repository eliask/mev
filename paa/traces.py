"""Canonical, source-grounded statement / relation / authority / action packets.

Retrieval is deliberately allowed to be noisy. Only version-bound reviews can
admit a substantive relation; the browser performs no semantic inference.
"""


import json
import sqlite3
from collections import Counter
from datetime import UTC, datetime

from paa.attribution import attach_attribution
from paa.config import CORPUS_CUTOFF
from paa.opportunity import opportunity_requirements
from paa.opportunity_codec import plan_to_wire
from paa.opportunity_records import ActionRequirements
from paa.records import evidence_reference_ids, finding_record, validate
from paa.relations import ObjectRetriever, fingerprint, verified_review


def _put_evidence(conn: sqlite3.Connection, evidence: dict, record: dict) -> str:
    key = record["evidence_id"]
    evidence[key] = record
    conn.execute("INSERT OR REPLACE INTO evidence VALUES (?, ?)", (key, json.dumps(record, ensure_ascii=False)))
    return key


def _source_evidence(key: str, url: str, quote: str, locator: str, **extra) -> dict:
    """Make every generated source reference self-describing.

    The original implementation emitted a convenient ``sha256`` field but
    omitted the evidence-contract fields used by statement spans.  That made
    role and candidacy references hard to audit and allowed a packet to mix
    incompatible evidence shapes.  Structured records still have no text
    span, but they carry the same stable identifiers, quote/hash and source
    locator as all other packet evidence.
    """

    return {
        "evidence_id": key,
        "document_version_id": locator,
        "kind": "structured_field",
        "text_sha256": fingerprint(quote) if quote else None,
        "span_start": None,
        "span_end": None,
        "quote": quote,
        "normalization_version": "nfc-1",
        "record_locator": locator,
        "field_path": None,
        "segment_start_seconds": None,
        "segment_end_seconds": None,
        "context_evidence_ids": [],
        "url": url,
        # Keep this alias for older exports while consumers migrate to the
        # schema-shaped ``text_sha256`` field.
        "sha256": fingerprint(quote) if quote else None,
        **extra,
    }


def _vote_objects(conn: sqlite3.Connection) -> list[dict]:
    return [{"object_id": f"vote-{r['aanestys_id']}", "kind": "VOTE", "matter_id": r["matter"],
             "title": r["kohta"] or r["title"] or "", "text": (r["kohta"] or "") + " " + (r["title"] or ""),
             "date": r["session_date"], "url": r["url"], "source_id": "SRC-EDUSKUNTA-VOTES",
             "authors": [], "evidence_ids": [f"vote-{r['aanestys_id']}-e"],
             "disposition": {"state": "UNRESOLVED", "evidence_ids": []}, "void": bool(r["mitatoity"])}
            for r in conn.execute("SELECT * FROM vote_events")]


_OBJECT_KINDS = {"INITIATIVE_AUTHORED": "LEGISLATIVE_INITIATIVE", "RESIGN_ROLE": "RESIGN_ROLE",
                 "QUESTION_FILED": "WRITTEN_QUESTION", "VOTE_CAST": "VOTE", "SPEECH_DELIVERED": "SPEECH"}
_OBJECT_CAPABILITIES = {"LEGISLATIVE_INITIATIVE": "MP_INITIATE_BILL", "VOTE": "PARLIAMENTARY_VOTE",
                        "WRITTEN_QUESTION": "FILE_PARLIAMENTARY_QUESTION", "SPEECH": "SPEAK_IN_PARLIAMENT",
                        "RESIGN_ROLE": "HOLD_ELECTED_ROLE"}
_ACTION_DATE_BASES = {"LEGISLATIVE_INITIATIVE": {"SIGNATURE_DATE", "VIREILLETULO_EVENT"},
                      "RESIGN_ROLE": {"INSTITUTIONAL_DECISION_DATE"},
                      "WRITTEN_QUESTION": {"SUBMISSION_DATE"}, "VOTE": {"SESSION_DATE"},
                      "SPEECH": {"SPEECH_DATE", "SESSION_DATE"}}


def _authority(conn: sqlite3.Connection, actor: dict, statement: dict, requirements: ActionRequirements | None, kind: str | None,
               deadline: str | None, evidence: dict) -> dict:
    # A source's earliest possible statement date does not prove that an
    # action happened afterwards. Only a known upper bound closes that gap.
    start = statement["stated_at"].get("latest")
    end = min(deadline or CORPUS_CUTOFF, CORPUS_CUTOFF)
    roles = []
    for row in conn.execute("SELECT * FROM mp_periods WHERE person_id = ?", (actor.get("person_id"),)):
        if row["kind"] not in {"VaaliPiiri", "Edustajatoimi"}:
            continue
        if not row["start_date"] or row["precision"] != "day":
            continue
        if row["start_date"] <= end and (row["end_date"] or CORPUS_CUTOFF) >= (start or "0001-01-01"):
            role = dict(row)
            key = f"role-{actor['actor_id']}-{row['kind']}-{row['start_date']}"
            _put_evidence(conn, evidence, _source_evidence(
                key, "https://avoindata.eduskunta.fi/api/v1/tables/MemberOfParliament/rows?columnName=personId&columnValue=" + str(actor["person_id"]),
                json.dumps(role, ensure_ascii=False, sort_keys=True), f"personId={actor['person_id']};{row['kind']}",
                provenance_note="Normalized member-register record; date precision must be day."))
            role["evidence_ids"] = [key]
            roles.append(role)
    year_text = str(statement["stated_at"].get("earliest") or "")[:4]
    year = int(year_text) if year_text.isdigit() else 0
    candidacies = [dict(row) for row in conn.execute(
        "SELECT * FROM candidacies WHERE actor_id = ? AND election_year = ?", (actor["actor_id"], year))]
    seat_condition = requirements is not None and requirements.condition_requires_election
    condition_state = "NOT_APPLICABLE"
    refs = [key for role in roles for key in role["evidence_ids"]]
    if seat_condition and candidacies:
        condition_state = "SATISFIED" if any(r["elected"] for r in candidacies) else "NOT_SATISFIED"
        for row in candidacies:
            key = f"candidacy-{row['candidacy_id']}-e"
            _put_evidence(conn, evidence, _source_evidence(
                key, f"https://tulospalvelu.vaalit.fi/EKV{year}/", json.dumps(row, ensure_ascii=False, sort_keys=True),
                row["candidacy_id"], provenance_note="Official roster result, normalized snapshot."))
            refs.append(key)
    elif seat_condition:
        condition_state = "UNRESOLVED"
    state = "UNRESOLVED"
    # Holding a role and satisfying a campaign condition are distinct axes.
    # A replacement MP can have authority despite losing their own election.
    if kind in {"LEGISLATIVE_INITIATIVE", "WRITTEN_QUESTION", "VOTE", "SPEECH"} and roles:
        state = "OBSERVABLE_OPPORTUNITY"
    elif kind in {"LEGISLATIVE_INITIATIVE", "WRITTEN_QUESTION", "VOTE", "SPEECH"} and condition_state == "NOT_SATISFIED":
        state = "NO_OBSERVABLE_OPPORTUNITY"
    if not start:
        state = "UNRESOLVED"
    return {"required_capability": kind, "role": "MEMBER_OF_PARLIAMENT" if kind in {"LEGISLATIVE_INITIATIVE", "WRITTEN_QUESTION", "VOTE"} else "ACTION_SPECIFIC",
            "roles": roles, "window": {"earliest": start, "latest": end}, "condition_state": condition_state,
            "opportunity_state": state, "evidence_ids": refs,
            "limitations": ["Member authority does not prove control of institutional outcomes."]}


def compile_traces(conn: sqlite3.Connection) -> dict:
    previous = {r["trace_id"]: json.loads(r["json"]) for r in conn.execute("SELECT trace_id, json FROM evidence_traces")}
    conn.execute("DELETE FROM evidence_traces")
    conn.execute("DELETE FROM relations")
    evidence = {r["evidence_id"]: json.loads(r["json"]) for r in conn.execute("SELECT * FROM evidence")}
    # Prefer a source-normalized official object to a metadata shell with the
    # same ID. Otherwise merely loading the vote roster can stale a review of
    # the unchanged decisive motion.
    objects = _vote_objects(conn) + [json.loads(r["json"]) for r in conn.execute("SELECT json FROM official_objects")]
    retriever = ObjectRetriever(objects)
    reviews: dict[str, list[dict]] = {}
    for row in conn.execute("SELECT json FROM relation_reviews"):
        review = json.loads(row["json"])
        reviews.setdefault(review["proposition_id"], []).append(review)
    statements = {r["statement_id"]: json.loads(r["json"]) for r in conn.execute("SELECT * FROM statements")}
    docs = {r["document_id"]: dict(r) for r in conn.execute("SELECT * FROM documents")}
    actors = {r["actor_id"]: dict(r) for r in conn.execute("SELECT * FROM actors")}
    coverage = [json.loads(r["json"]) for r in conn.execute("SELECT json FROM source_coverage")]
    counts: Counter = Counter()
    for row in conn.execute("SELECT json FROM propositions ORDER BY proposition_id"):
        prop = json.loads(row["json"])
        statement = statements[prop["statement_ids"][0]]
        actor_ids = prop.get("subject_actor_ids") or []
        actor_id = actor_ids[0] if len(actor_ids) == 1 else None
        actor = actors.get(actor_id) or {"actor_id": actor_id, "person_id": None}
        doc = docs[statement["statement_id"]]
        # Campaign-span evidence is part of the statement record, but older
        # acquisition paths do not materialize it in the evidence table.
        # Materialize it before validating relation reviews so a review that
        # cites the statement span is not rejected merely for storage order.
        for statement_evidence in statement.get("evidence", []):
            if statement_evidence.get("evidence_id") not in evidence:
                _put_evidence(conn, evidence, statement_evidence)
        text = prop.get("source_text") or prop.get("original_text") or statement["original_text"]
        plan = opportunity_requirements(prop)
        plan_wire = plan_to_wire(plan)
        requirements = plan if type(plan) is ActionRequirements else None
        kind = _OBJECT_KINDS.get(requirements.action_kind.value) if requirements is not None else None
        if kind and requirements.required_capability.value != _OBJECT_CAPABILITIES[kind]:
            kind = None
        target = prop["target"].get("value")
        searchable = prop["semantic_type"] in {"PERSONAL_ACTION_COMMITMENT", "PERSONAL_RESTRAINT_COMMITMENT", "POSITION", "POLICY_DESIDERATUM", "COLLECTIVE_ACTION_COMMITMENT", "BROAD_OBJECTIVE", "OUTCOME_COMMITMENT", "MAINTAIN_COMMITMENT", "PREVENT_COMMITMENT"}
        retrieved = retriever.search(text) if searchable else []
        relations = []
        for review in reviews.get(prop["proposition_id"], []):
            obj = retriever.objects.get(review["object_id"])
            valid = obj is not None and verified_review(review, statement, prop, obj, evidence)
            relations.append({**review, "status": review["status"] if valid else "UNRESOLVED", "validation_state": "VALID" if valid else "STALE_OR_UNGROUNDED"})
            if obj and review["object_id"] not in {item["object_id"] for item in retrieved}:
                retrieved.append({"object_id": obj["object_id"], "status": "CANDIDATE", "method": "review-reference", "matched_tokens": [], "score": None})
        reviewed = {r["object_id"] for r in relations}
        for item in retrieved:
            if item["object_id"] not in reviewed:
                relations.append({"relation_id": f"rel-{prop['proposition_id']}-{item['object_id']}", "object_id": item["object_id"],
                                  "status": "CANDIDATE", "rationale": "Sanat auttavat hakua. Kohteen samuutta ei ole varmennettu.",
                                  "evidence_ids": [statement["evidence"][0]["evidence_id"], *retriever.objects[item["object_id"]].get("evidence_ids", [])], "validation_state": "UNREVIEWED"})
        authority = _authority(conn, actor, statement, requirements, kind, prop["deadline"].get("value"), evidence)
        authority["required_capability"] = requirements.required_capability.value if requirements is not None else None
        authority["required_role"] = requirements.required_role if requirements is not None else None
        authority["verification_plan"] = plan_wire
        if requirements is not None and requirements.condition and not requirements.condition_requires_election:
            authority["condition_state"] = "UNRESOLVED"
        actions = []
        for relation in relations:
            if relation["status"] not in {"SAME_MATTER", "SAME_POLICY_OBJECT"} or relation.get("validation_state") != "VALID":
                continue
            obj = retriever.objects[relation["object_id"]]
            when = obj.get("action_date")
            if not when and obj.get("kind") != "LEGISLATIVE_INITIATIVE":
                when = obj.get("date")
            window = authority["window"]
            object_kind = obj.get("kind")
            narrow_action = kind is not None and object_kind == kind
            related_record = requirements is None and prop["semantic_type"] in {"POSITION", "POLICY_DESIDERATUM", "BROAD_OBJECTIVE", "OUTCOME_COMMITMENT", "MAINTAIN_COMMITMENT", "PREVENT_COMMITMENT"} and object_kind in {"LEGISLATIVE_INITIATIVE", "WRITTEN_QUESTION", "SPEECH"}
            if not (narrow_action or related_record) or obj.get("void") or not when:
                continue
            if obj.get("action_date_basis") not in _ACTION_DATE_BASES.get(object_kind, set()):
                continue
            # A signature proves signing, not that the initiative was filed
            # within a promised interval. A later signature also leaves the
            # policy text's binding to the earlier filing event unresolved.
            if narrow_action and object_kind == "LEGISLATIVE_INITIATIVE" and (
                obj.get("action_date_basis") != "VIREILLETULO_EVENT"
                or obj.get("date_binding_state") == "SIGNATURE_AFTER_FILING"
            ):
                continue
            if not window["earliest"] or not (window["earliest"] < when <= window["latest"]):
                continue
            supported_alignment = relation.get("action_alignment") == "ALIGNED" if narrow_action else relation.get("action_alignment") in {"ALIGNED", "RELATED"}
            if not supported_alignment or authority["condition_state"] not in {"SATISFIED", "NOT_APPLICABLE"}:
                continue
            for author in obj.get("authors", []):
                same_actor = actor.get("person_id") and str(author.get("person_id")) == str(actor["person_id"])
                reviewed_actor = author.get("actor_id") == actor_id and author.get("identity_basis") == "INDEPENDENT_CASE_REVIEW"
                allowed_roles = {"AUTHOR", "ACTOR"} if narrow_action or object_kind == "SPEECH" else {"AUTHOR", "COSIGNER"}
                if not (same_actor or reviewed_actor) or author.get("role") not in allowed_roles:
                    continue
                action_refs = author.get("evidence_ids") or []
                date_refs = obj.get("action_date_source_evidence_ids") or obj.get("evidence_ids") or []
                if not action_refs or not date_refs or not all(key in evidence for key in action_refs + date_refs):
                    continue
                if object_kind != "RESIGN_ROLE" and not any(role["start_date"] <= when <= (role["end_date"] or CORPUS_CUTOFF) for role in authority["roles"]):
                    continue
                actions.append({"object_id": obj["object_id"], "kind": object_kind, "date": when, "actor_id": actor_id,
                                "date_basis": obj["action_date_basis"], "role": author["role"], "state": "OBSERVED_ALIGNED_ACTION" if narrow_action else "DOCUMENTED_RELATED_ACTION", "evidence_ids": sorted(set(action_refs + date_refs + relation["evidence_ids"]))})
                if kind == "RESIGN_ROLE":
                    role = (obj.get("action") or {}).get("role")
                    if not role:
                        actions.pop()
                        continue
                    authority["roles"] = [{"kind": "SOURCE_ATTESTED_ROLE", "label": role,
                                           "start_date": obj.get("request_date"), "end_date": when,
                                           "precision": "bounded_source_observations", "evidence_ids": action_refs,
                                           "limitations": ["Only the request and decision observations establish this role; a complete tenure is not inferred."]}]
                    authority["opportunity_state"] = "OBSERVABLE_OPPORTUNITY"
                    authority["evidence_ids"] = sorted(set(authority["evidence_ids"] + action_refs))
        state = "NOT_TESTABLE"
        non_action_claims = {
            "VALUE_OR_SLOGAN": "Arvo tai tunnuslause ei yksilöi tästä rekisteristä ratkaistavaa omaa tekoa.",
            "BROAD_OBJECTIVE": "Laaja tavoite ei yksilöi mitattavaa omaa tekoa tai lopputulosta tässä tarkistuspolussa.",
            "PROCESS_COMMITMENT": "Työtapasitoumusta ei ratkaista yksittäisestä aloite- tai äänestysmerkinnästä.",
            "REPORTED_SPEECH": "Siteerattua puhetta ei ole luettu tämän henkilön omaksi toimintalupaukseksi.",
            "CAUSAL_EFFECT_FORECAST": "Vaikutusarvio tarvitsee erillisen syy-seurausnäytön; toimintarekisteri ei ratkaise sitä.",
            "OBSERVED_STATE_FORECAST": "Tilannearviota ei ratkaista henkilön toimintarekisteristä.",
        }
        claim = non_action_claims.get(prop["semantic_type"], "Teksti ei nimeä tässä rekisterissä tarkistettavaa omaa tekoa.")
        limitations = ["Havaittu teko ei osoita tavoitteen toteutumista, vaikutusta tai koko lupauksen täyttymistä."]
        if searchable:
            state = "INSUFFICIENT_EVIDENCE"
            claim = "Lausuman toimintakohteen yhteyttä myöhempään asiakirjaan ei ole varmennettu. Hakutulos ei osoita samaa asiaa."
        if prop["semantic_type"] == "AMBIGUOUS":
            state = "INSUFFICIENT_EVIDENCE"
            claim = "Tekstin tulkinta ei riitä valitsemaan tarkistettavaa omaa tekoa. Tämä ei osoita, ettei teksti sisältäisi sitoumusta."
        elif prop["semantic_type"] == "PERSONAL_ACTION_COMMITMENT" and kind is None:
            state = "INSUFFICIENT_EVIDENCE"
            claim = "Lausumassa nimetyn teon toimivaltaa tai tarkistettavaa rekisteriä ei ole varmennettu."
        elif prop["semantic_type"] == "PERSONAL_RESTRAINT_COMMITMENT" and kind is None:
            state = "INSUFFICIENT_EVIDENCE"
            claim = "Pidättäytymissitoumusta ei ole yhdistetty varmennettuun päätökseen tässä aineistossa. Tämä ei osoita sitoumuksen noudattamista tai rikkomista."
        if kind and authority["opportunity_state"] == "NO_OBSERVABLE_OPPORTUNITY":
            state = "NO_OBSERVABLE_OPPORTUNITY"
            claim = "Lauseen vaaliehto ei täyttynyt eikä rekisterissä ole vastaavaa kansanedustajajaksoa tässä aikaikkunassa. Teon puuttumisesta ei päätellä laiminlyöntiä."
        if any(a["state"] == "OBSERVED_ALIGNED_ACTION" for a in actions):
            state = "OBSERVED_ALIGNED_ACTION"
            claim = "Myöhempi julkinen asiakirja osoittaa lausetta vastaavan oman teon ilmoitetussa aikaikkunassa."
            reviewed_claims = [r["bounded_claim"] for r in relations if r.get("bounded_claim") and r.get("validation_state") == "VALID" and r["object_id"] in {a["object_id"] for a in actions}]
            if reviewed_claims:
                claim = " ".join(dict.fromkeys(reviewed_claims))
        elif actions:
            state = "NOT_TESTABLE"
            claim = "Myöhempi lähde osoittaa henkilön aloitetoimintaa samasta tarkistetusta kohteesta. Tämä lause ei lupaa juuri tämän aloitteen tekemistä eikä yhteys osoita vaalitavoitteen toteutumista."
            if prop["semantic_type"] in {"OUTCOME_COMMITMENT", "MAINTAIN_COMMITMENT", "PREVENT_COMMITMENT"}:
                claim = "Myöhempi lähde osoittaa kirjattua toimintaa samasta tarkistetusta kohteesta. Sitoumus lopputulokseen säilyy; yksittäinen teko ei ratkaise toteutumista, säilymistä tai estämistä."
            elif any(action["kind"] != "LEGISLATIVE_INITIATIVE" for action in actions):
                claim = "Myöhempi lähde osoittaa kirjattua toimintaa samasta tarkistetusta kohteesta. Yhteys ei osoita tavoitteen toteutumista tai koko sitoumuksen täyttymistä."
        if statement["attribution_basis"] == "UNRESOLVED" and state in {"OBSERVED_ALIGNED_ACTION", "NO_OBSERVABLE_OPPORTUNITY"}:
            state = "INSUFFICIENT_EVIDENCE"
            claim = "Henkilön tunnistus ei ole riittävän varma toimintapäätelmää varten."
            actions = []
        retrieved_matters = {retriever.objects[item["object_id"]].get("matter_id") for item in retrieved}
        selected_coverage = [c for c in coverage if (
            (kind is not None and c.get("kind") == kind) or
            bool(retrieved_matters & set((c.get("query") or {}).get("matter_ids", [])))
        ) and (not c.get("person_id") or str(c["person_id"]) == str(actor.get("person_id")))]
        if kind and not actions and state == "INSUFFICIENT_EVIDENCE":
            claim = "Vastaavaa omaa tekoa ei ole varmennettu haetussa lähdeaineistossa. Tämä ei osoita, ettei tekoa tehty."
        if not selected_coverage and kind:
            limitations.append("Toimintarekisterin täydellistä kattavuutta ei ole osoitettu.")
        source_ref = statement["evidence"][0]["evidence_id"]
        _put_evidence(conn, evidence, {**statement["evidence"][0], "url": doc["url"], "source_id": doc["source_id"],
                                     "source_text_sha256": doc["sha256"], "record_locator": statement["statement_id"]})
        packet_objects = []
        for item in retrieved:
            obj = retriever.objects[item["object_id"]]
            # Preserve normalized contents and immutable source references.
            # Raw XML and duplicate evidence bodies live in the source store,
            # rather than being copied into every candidate trace.
            packet_objects.append({**{key: value for key, value in obj.items()
                                      if key not in {"source_row", "evidence"}}, "retrieval": item})
            for key in obj.get("evidence_ids", []):
                if key not in evidence:
                    _put_evidence(conn, evidence, _source_evidence(key, obj.get("url", ""), obj.get("text", ""), obj["object_id"], source_id=obj.get("source_id")))
        refs = {source_ref, *authority["evidence_ids"]}
        for obj in packet_objects:
            refs.update(obj.get("evidence_ids", []))
        for relation in relations:
            refs.update(relation.get("evidence_ids", []))
        for action in actions:
            refs.update(action["evidence_ids"])
        refs.update(evidence_reference_ids([packet_objects, relations, selected_coverage, prop]))
        packet = {"schema_version": "1.0", "trace_id": "trace-" + prop["proposition_id"], "statement_id": statement["statement_id"],
                  "proposition_id": prop["proposition_id"], "actor_id": actor_id, "as_of": CORPUS_CUTOFF,
                  "statement": {"text": statement["original_text"], "url": doc["url"], "evidence_ids": [source_ref], "stated_at": statement["stated_at"],
                                "issuer_actor_ids": statement["issuer_actor_ids"], "attribution_basis": statement["attribution_basis"]},
                  "proposition": {"text": text, "span": prop.get("source_span"), "semantic_type": prop["semantic_type"], "evidence_ids": prop["evidence_ids"],
                                  "validation_state": prop.get("validation_state", "PROPOSED"), "run_id": prop.get("run_id"),
                                  "interpretation_provenance": prop.get("predicate")},
                  "target": {"normalized_object": next((r["normalized_target"] for r in relations if r.get("normalized_target") and r["status"] in {"SAME_POLICY_OBJECT", "SAME_MATTER"}), target), "interpretation_state": "REVIEWED" if any(r["status"] in {"SAME_POLICY_OBJECT", "SAME_MATTER"} for r in relations) else "UNRESOLVED", "evidence_ids": sorted({source_ref, *(key for r in relations if r["status"] in {"SAME_POLICY_OBJECT", "SAME_MATTER"} for key in r["evidence_ids"])}), "review_ids": [r["review_id"] for r in relations if r["status"] in {"SAME_POLICY_OBJECT", "SAME_MATTER"}]},
                  "retrieved_objects": packet_objects, "relations": relations, "authority": authority, "actions": actions,
                  "disposition": [{"object_id": o["object_id"], **o.get("disposition", {"state": "UNRESOLVED", "evidence_ids": []})} for o in packet_objects],
                  "coverage": selected_coverage, "assessment": {"state": state, "claim": claim, "limitations": limitations,
                  "evidence_ids": sorted(refs), "causal_claim": False}, "evidence": [evidence[key] for key in sorted(refs) if key in evidence],
                  "correction_history": []}
        packet = attach_attribution(packet, proposition=prop)
        old = previous.get(packet["trace_id"])
        if old:
            packet["correction_history"] = list(old.get("correction_history", []))
            keys = ("statement", "proposition", "relations", "authority", "actions", "assessment",
                    "commitment_carrier", "decision_episodes", "attribution_envelopes")
            if any(old.get(key) != packet[key] for key in keys):
                packet["correction_history"].append({
                    "changed_at": datetime.now(UTC).isoformat(),
                    "previous_packet_sha256": fingerprint(json.dumps(old, ensure_ascii=False, sort_keys=True)),
                    "previous_assessment": old["assessment"],
                    "reason": "Recompiled after source or interpretation changed; prior finding retained here.",
                })
        validate("evidence_trace", packet)
        conn.execute("INSERT INTO evidence_traces VALUES (?, ?, ?, ?, ?, ?)",
                     (packet["trace_id"], packet["statement_id"], packet["proposition_id"], actor_id, state, json.dumps(packet, ensure_ascii=False)))
        counts[state] += 1
        admitted = state in {"OBSERVED_ALIGNED_ACTION", "NO_OBSERVABLE_OPPORTUNITY"}
        deadline = prop["deadline"].get("value")
        finding = finding_record(
            finding_id=packet["trace_id"] + "-finding", finding_type="ACTION_CONGRUENCE",
            proposition_ids=[prop["proposition_id"]], actor_ids=actor_ids, plan_id=packet["trace_id"], as_of=CORPUS_CUTOFF,
            temporal_state="NOT_DUE" if deadline and deadline > CORPUS_CUTOFF else "DUE" if deadline else "NO_DETERMINABLE_DEADLINE",
            condition_state=authority["condition_state"], target_state="UNKNOWN",
            action_congruence="ALIGNED" if state == "OBSERVED_ALIGNED_ACTION" else "NO_OBSERVABLE_OPPORTUNITY" if state == "NO_OBSERVABLE_OPPORTUNITY" else "NOT_APPLICABLE" if state == "NOT_TESTABLE" else "INSUFFICIENT_EVIDENCE",
            attribution="ACTOR_CONTROLLED_ACTION" if state == "OBSERVED_ALIGNED_ACTION" else "UNASSIGNED",
            supported_claim=claim, support_evidence_ids=sorted(refs), coverage_ids=[c["coverage_id"] for c in selected_coverage],
            limitations=limitations, admission_state="ADMITTED" if admitted else "CANDIDATE",
            admission_route="INDEPENDENT_CASE_REVIEW" if state == "OBSERVED_ALIGNED_ACTION" else "DETERMINISTIC_VALIDATION" if admitted else "NONE",
            validation_artifact_ids=[r.get("review_id") or r.get("relation_id") for r in relations if r.get("validation_state") == "VALID"] + ["tests/test_paa_trace_e2e.py"] if admitted else [],
        )
        validate("finding", finding)
        conn.execute("INSERT OR REPLACE INTO findings VALUES (?, ?)", (finding["finding_id"], json.dumps(finding, ensure_ascii=False)))
    return {"traces": sum(counts.values()), "trace_states": dict(counts), "official_objects": len(objects)}
