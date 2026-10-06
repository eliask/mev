"""Prompt iteration for campaign-sentence types.

The model sees numbered lines and returns `N CODE`. Deadlines and negation
stay on the regex reader. Model labels are not admitted.
"""


import json
from pathlib import Path

from paa.config import REPORTS, ensure_dirs
from paa.llm_client import PROMPT_A, PROMPT_B, PROMPT_C, complete, expand, parse_codes
from paa.semantics import analyze_text

FIXTURES = Path(__file__).resolve().parent / "contracts" / "fixtures"
ANCHORS = (
    ("R01", "Isänmaa sydämessä.", "VALUE_OR_SLOGAN"),
    ("R02", "Laittaa Suomen talous ja koulutus kuntoon.", "BROAD_OBJECTIVE"),
    ("R03", "Lupaan pohjata päätökseni parhaaseen mahdolliseen tietoon.", "PROCESS_COMMITMENT"),
    ("R04", "Polttoaineen hinta on saatava alaspäin ja dieselvero on poistettava.", "POLICY_DESIDERATUM"),
)


def _items() -> list[dict]:
    rows = []
    for line in (FIXTURES / "semantic_cases.jsonl").read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        row = json.loads(line)
        text = (row.get("input") or {}).get("text") or ""
        if not text:
            continue
        expected = row["expected"].get("statement_type")
        rows.append({"id": row["fixture_id"], "text": text, "expected": expected})
    for item_id, text, expected in ANCHORS:
        rows.append({"id": item_id, "text": text, "expected": expected})
    return rows


def _user_block(rows: list[dict]) -> str:
    return "\n".join(f"[{index}] {row['text']}" for index, row in enumerate(rows, start=1))


def _score(name: str, system: str, rows: list[dict]) -> dict:
    user = _user_block(rows)
    budget = 40 + len(rows) * 6
    try:
        raw, usage = complete(system, user, budget)
    except RuntimeError:
        raw, usage = complete(system, user, budget * 2)
    codes = parse_codes(raw)
    compared = []
    for index, row in enumerate(rows, start=1):
        code = codes.get(index)
        label = expand(code) if code else None
        det = analyze_text(row["text"])
        det_type = det.propositions[0].semantic_type if det.propositions else None
        compared.append(
            {
                "id": row["id"],
                "code": code,
                "label": label,
                "expected": row["expected"],
                "deterministic": det_type,
                "match_expected": None if row["expected"] is None else label == row["expected"],
                "match_deterministic": label == det_type,
            }
        )
    expected_rows = [item for item in compared if item["expected"] is not None]
    hits = sum(item["match_expected"] for item in expected_rows)
    return {
        "prompt": name,
        "items": len(rows),
        "parsed_lines": len(codes),
        "compliance": len(codes) / len(rows),
        "expected_hits": hits,
        "expected_n": len(expected_rows),
        "deterministic_hits": sum(item["match_deterministic"] for item in compared),
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "rows": compared,
        "raw": raw,
    }


def main() -> None:
    ensure_dirs()
    rows = _items()
    reports = [_score("A", PROMPT_A, rows), _score("B", PROMPT_B, rows), _score("C", PROMPT_C, rows)]
    best = max(reports, key=lambda item: (item["expected_hits"], item["deterministic_hits"], item["compliance"]))
    summary = {
        "format": "N CODE",
        "codes_expanded_at_storage": True,
        "grammar": False,
        "admitted": False,
        "prompts": [
            {
                "prompt": item["prompt"],
                "compliance": round(item["compliance"], 3),
                "expected": f"{item['expected_hits']}/{item['expected_n']}",
                "deterministic": f"{item['deterministic_hits']}/{item['items']}",
                "completion_tokens": item["completion_tokens"],
            }
            for item in reports
        ],
        "best": best["prompt"],
        "misses": [
            {"id": row["id"], "got": row["label"], "expected": row["expected"], "deterministic": row["deterministic"]}
            for row in best["rows"]
            if row["match_expected"] is False or (row["expected"] and row["label"] is None)
        ],
    }
    path = REPORTS / "llm_line_format.json"
    path.write_text(json.dumps({"summary": summary, "runs": reports}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
