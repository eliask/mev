"""Apply source-checked, candidate-only local interpretations to a build."""


import hashlib
import json
from pathlib import Path

from paa.semantics import Proposition


class ModelInterpretations:
    def __init__(self, run_dir: Path):
        self.root = run_dir
        self.manifest = json.loads((run_dir / "manifest.json").read_text())
        self.run_id = "local-model:" + hashlib.sha256(json.dumps(self.manifest["identity"], sort_keys=True).encode()).hexdigest()[:20]
        self.applied = 0
        self.abstained = 0
        self.receipts = {}

    def propositions(self, document: dict, canonical: list[Proposition]) -> list[Proposition]:
        path = self.root / "documents" / (document["document_id"] + ".json")
        if not path.exists():
            raise ValueError(f"Model run has no receipt for {document['document_id']}; finish the corpus run before compiling")
        result = json.loads(path.read_text())
        self.receipts[document["document_id"]] = result.get("request_id")
        source_hash = hashlib.sha256(document["text"].encode()).hexdigest()
        if result.get("source_text_sha256") != source_hash:
            raise ValueError(f"Model source version changed: {document['document_id']}")
        if result.get("receipt_status") == "OK":
            from paa.llm_run import revalidate_document

            spec = {"document": document, "units": [
                {"unit_id": f"{document['document_id']}-p{index}", "text": prop.text,
                 "start": prop.source_start, "end": prop.source_end}
                for index, prop in enumerate(canonical, 1)]}
            result = revalidate_document(result, spec, self.manifest["identity"]["prompt"])
        units = {u["unit_id"]: u for u in result.get("units", [])}
        output = []
        for index, original in enumerate(canonical, 1):
            unit_id = f"{document['document_id']}-p{index}"
            unit = units.get(unit_id)
            row = unit.get("proposition") if unit and unit.get("status") == "VALID" and result.get("receipt_status") == "OK" else None
            if row:
                if (row["source_quote"] != original.text or row["source_start"] != original.source_start
                        or row["source_end"] != original.source_end):
                    raise ValueError(f"Model segmentation changed: {unit_id}")
                prop = Proposition(
                    text=original.text, semantic_type=row["semantic_type"], testability="UNRESOLVED",
                    personal_action_commitment=row["semantic_type"] in {"PERSONAL_ACTION_COMMITMENT", "PERSONAL_RESTRAINT_COMMITMENT"},
                    issuer_scope=row["issuer_scope"], deadline=row["deadline_normalized"],
                    deadline_basis=row["deadline_basis"], negation=row["negation"],
                    condition=row["condition_quote"], targets=[row["target_quote"]] if row["target_quote"] else [],
                    reported_speech=row["semantic_type"] == "REPORTED_SPEECH",
                    action_kind=row["action_kind"], required_capability=row["required_capability"],
                    observable_action=row["observable_action"],
                    source_start=original.source_start, source_end=original.source_end,
                    validation_state="PROPOSED",
                )
                self.applied += 1
            else:
                prop = Proposition(text=original.text, semantic_type="AMBIGUOUS", testability="UNRESOLVED",
                    source_start=original.source_start, source_end=original.source_end,
                    missing_specification=["local_model_invalid_or_abstained"], validation_state="PROPOSED")
                self.abstained += 1
            output.append(prop)
        return output
