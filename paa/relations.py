"""Generic official-object retrieval and version-bound relation admission."""


import hashlib
import heapq
import math
import re
from collections import Counter, defaultdict


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_STOP = {"että", "jotta", "tämä", "jotka", "mikä", "missä", "sekä", "kanssa", "ilman", "suomen", "suomi", "eduskuntaan", "lupaan", "teen", "vuoden", "puolella", "pääsen", "haluan", "kaikki", "myös", "tulee", "pitää", "tämän", "näistä", "eduskunnan",
         "siihen", "siitä", "tässä", "tähän", "jossa", "jonka", "jotain", "joiden", "sekään", "tärkeänä", "päästä", "vaikuttamaan", "vaikuttaa", "vaikuttaminen", "edistää", "edistämään", "kirjattava", "syntymään", "syntyvään", "pidän", "omasta", "mielestäni"}


def retrieval_tokens(text: str) -> set[str]:
    """Words and long-word prefixes rank candidates, never object identity.

    Prefixes recover some Finnish inflections. They intentionally tolerate
    false positives and are not a policy ontology or an admission rule.
    """
    words = {word for word in re.findall(r"[a-zåäö]{4,}", text.casefold()) if word not in _STOP}
    return words | {word[:6] + "*" for word in words if len(word) > 8}


class ObjectRetriever:
    def __init__(self, objects: list[dict]):
        self.objects = {item["object_id"]: item for item in objects}
        self.index: dict[str, set[str]] = defaultdict(set)
        for item in objects:
            for token in retrieval_tokens(item.get("title", "") + " " + item.get("text", "")):
                self.index[token].add(item["object_id"])

    def search(self, text: str, k: int = 5) -> list[dict]:
        scores: Counter = Counter()
        tokens = sorted(retrieval_tokens(text))
        for token in tokens:
            posting = self.index.get(token, ())
            if not posting:
                continue
            weight = math.log(1 + len(self.objects) / len(posting))
            for object_id in posting:
                scores[object_id] += weight
        ranked = heapq.nsmallest(k, scores, key=lambda key: (-scores[key], key))
        return [{"object_id": key, "score": round(scores[key], 4),
                 "matched_tokens": [token for token in tokens if key in self.index.get(token, ())],
                 "method": "token-prefix-idf-v3", "status": "CANDIDATE"} for key in ranked]


def review_shape_valid(review: dict) -> bool:
    """An explicit review has typed identifiers, source quotes and references."""
    fields = ("review_id", "proposition_id", "object_id", "matter_id", "statement_sha256",
              "object_sha256", "statement_quote", "object_quote", "reviewer", "rationale", "status")
    if any(not isinstance(review.get(key), str) or not review[key].strip() for key in fields):
        return False
    if review["status"] not in {"SAME_POLICY_OBJECT", "SAME_MATTER", "REJECTED", "UNRESOLVED"}:
        return False
    target = review.get("normalized_target")
    if target is not None and (not isinstance(target, str) or not target.strip()):
        return False
    if review["status"] in {"SAME_POLICY_OBJECT", "SAME_MATTER"} and target is None:
        return False
    refs = review.get("evidence_ids")
    if not isinstance(refs, list) or not refs or any(not isinstance(ref, str) or not ref.strip() for ref in refs):
        return False
    return all(review.get(key) is None or isinstance(review[key], str)
               for key in ("bounded_claim", "action_alignment", "target_scope"))


_UNADMITTED_REVIEW_STATES = frozenset({"PROPOSED", "CANDIDATE", "CONTESTED", "WITHHELD", "RETRACTED"})


def _source_review_admitted(review: dict) -> bool:
    """Return whether a relation record crossed a source-review gate.

    Model proposals deliberately carry the same quotes and hashes as a human
    review so they can be inspected and replayed.  Those anchors are necessary
    but not sufficient for relation admission: without this gate a model's
    ``ALIGNED`` proposal could become an ``OBSERVED_ALIGNED_ACTION`` trace.
    Legacy/source-reviewed fixtures have no workflow-state fields and remain
    admissible after the ordinary shape, hash, quote, and evidence checks.
    """

    admission_state = review.get("admission_state")
    if admission_state in _UNADMITTED_REVIEW_STATES:
        return False
    if isinstance(admission_state, str) and admission_state.upper().startswith("MODEL_"):
        return False
    review_state = review.get("review_state")
    if review_state in _UNADMITTED_REVIEW_STATES:
        return False
    validation_state = review.get("validation_state")
    if validation_state in {"PROPOSED", "UNREVIEWED", "STALE_OR_UNGROUNDED"}:
        return False
    admission_route = review.get("admission_route")
    if isinstance(admission_route, str) and admission_route.upper().startswith("MODEL_"):
        return False
    review_method = review.get("review_method")
    if isinstance(review_method, str) and review_method.upper().startswith("LOCAL_LLM"):
        return False
    reviewer = review.get("reviewer")
    return not (isinstance(reviewer, str) and reviewer.casefold().startswith("local-llm-"))


def verified_review(review: dict, statement: dict, proposition: dict, obj: dict, evidence: dict) -> bool:
    """Reject stale or ungrounded reviews. A retrieval score is never a review."""
    if not review_shape_valid(review):
        return False
    if not _source_review_admitted(review):
        return False
    if review.get("proposition_id") != proposition["proposition_id"] or review.get("object_id") != obj["object_id"]:
        return False
    if not review.get("matter_id") or review["matter_id"] != obj.get("matter_id"):
        return False
    if review.get("statement_sha256") != fingerprint(statement["original_text"]):
        return False
    if review.get("object_sha256") != fingerprint(obj.get("text", "")):
        return False
    if not review.get("rationale") or not review.get("reviewer"):
        return False
    quote = review.get("statement_quote", "")
    object_quote = review.get("object_quote", "")
    if not quote or quote not in statement["original_text"] or not object_quote or object_quote not in obj.get("text", ""):
        return False
    proposition_text = proposition.get("source_text") or proposition.get("original_text")
    if proposition_text and quote not in proposition_text:
        return False
    refs = review.get("evidence_ids") or []
    source_refs = {item["evidence_id"] for item in statement.get("evidence", [])}
    object_refs = set(obj.get("evidence_ids") or [])
    return bool(source_refs & set(refs)) and bool(object_refs & set(refs)) and all(ref in evidence for ref in refs)
