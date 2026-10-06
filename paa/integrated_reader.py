"""Portable offline builder for a question-driven reader artifact.

Manifest paths are resolved relative to the manifest file directory (or an
explicit ``--inputs`` root). No private experiment directories are hardcoded.
Callers stage relative source/candidate/mapping/review locators in their bundle.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from paa.bootstrap_site import build_bootstrap_artifact
from paa.question_journey import build_journey_from_paths

SCHEMA_VERSION = "paa.integrated_reader_manifest.v1"


class IntegratedReaderError(ValueError):
    """Raised when a portable integrated-reader manifest cannot be built."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(root: Path, value: object, label: str) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        raise IntegratedReaderError(f"missing {label}: {path}")
    return path


def _resolve_dir(root: Path, value: object, label: str) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    if not path.is_dir():
        raise IntegratedReaderError(f"missing {label}: {path}")
    return path


def compose_journeys_from_manifest_entries(
    *,
    root: Path,
    entries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Compose journey packets from portable journey_builds entries."""

    journeys: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise IntegratedReaderError(f"journey_builds[{index}] must be an object")
        build_id = str(entry.get("id") or entry.get("task_id") or f"journey_{index}")
        if entry.get("route_packet"):
            route_packet = _resolve(root, entry["route_packet"], f"{build_id}.route_packet")
        elif entry.get("route_dir") and entry.get("task_id"):
            route_dir = _resolve_dir(root, entry["route_dir"], f"{build_id}.route_dir")
            matches = [
                path
                for path in sorted(route_dir.glob("*.json"))
                if path.name != "manifest.json"
                and _load(path).get("task_id") == entry["task_id"]
            ]
            if len(matches) != 1:
                raise IntegratedReaderError(
                    f"{build_id}: could not resolve unique route packet for task_id="
                    f"{entry['task_id']!r} in {route_dir}"
                )
            route_packet = matches[0]
        else:
            raise IntegratedReaderError(
                f"{build_id} requires route_packet or route_dir+task_id"
            )
        mapping = _resolve(root, entry["mapping"], f"{build_id}.mapping")
        answer = (
            _resolve(root, entry["answer"], f"{build_id}.answer")
            if entry.get("answer")
            else None
        )
        source_review = (
            _resolve(root, entry["source_answer_review"], f"{build_id}.source_answer_review")
            if entry.get("source_answer_review")
            else None
        )
        projection_review = (
            _resolve(root, entry["projection_review"], f"{build_id}.projection_review")
            if entry.get("projection_review")
            else None
        )
        # Negative / non-v1 receipts must not be passed as projection_review.
        if projection_review is not None:
            receipt = _load(projection_review)
            if receipt.get("schema_version") != "paa.journey_projection_review.v1":
                raise IntegratedReaderError(
                    f"{build_id}: projection_review must be paa.journey_projection_review.v1; "
                    f"got {receipt.get('schema_version')!r}"
                )
        journeys.append(
            build_journey_from_paths(
                route_packet_path=route_packet,
                mapping_path=mapping,
                answer_path=answer,
                source_answer_review_path=source_review,
                projection_review_path=projection_review,
            )
        )
    return journeys


def build_integrated_reader(
    *,
    manifest_path: Path,
    output_dir: Path,
    inputs_root: Path | None = None,
) -> dict[str, Any]:
    """Build one linked reader artifact from a portable manifest.

    Supported manifest keys (relative to inputs root unless absolute):

    - ``matters``: list of matter packet JSON paths
    - ``question_packets``, ``source_banks``, ``route_dirs``, ``reviews``
    - ``reviewed_candidates``: ``{task_id: path}``
    - ``journey_builds``: mapping/route/answer/review locators (composed into site)
    - optional Oulaskangas/editorial paths and ``title`` / ``review_date``
    """

    manifest_path = Path(manifest_path).resolve()
    root = Path(inputs_root).resolve() if inputs_root else manifest_path.parent
    manifest = _load(manifest_path)
    if not isinstance(manifest, Mapping):
        raise IntegratedReaderError("manifest must be a JSON object")
    known = {SCHEMA_VERSION, "paa.bs02_portable_inputs.v1"}
    has_carrier = any(key in manifest for key in ("journey_builds", "matters", "retained_matters"))
    if manifest.get("schema_version") not in known and not has_carrier:
        raise IntegratedReaderError(
            f"unsupported integrated-reader schema {manifest.get('schema_version')!r}"
        )
    matter_rels = list(manifest.get("matters") or manifest.get("retained_matters") or [])
    matters = [_resolve(root, rel, f"matter {rel}") for rel in matter_rels]
    questions = [
        _resolve(root, rel, f"question {rel}")
        for rel in manifest.get("question_packets") or []
    ]
    source_banks = [
        _resolve(root, rel, f"source-bank {rel}")
        for rel in manifest.get("source_banks") or []
    ]
    route_dirs = [
        _resolve_dir(root, rel, f"route-dir {rel}")
        for rel in manifest.get("route_dirs") or []
    ]
    reviews = [_resolve(root, rel, f"review {rel}") for rel in manifest.get("reviews") or []]
    reviewed_candidates: dict[str, Path] = {}
    declared = manifest.get("reviewed_candidates") or {}
    if isinstance(declared, Mapping):
        for task_id, rel in declared.items():
            reviewed_candidates[str(task_id)] = _resolve(
                root, rel, f"reviewed_candidates[{task_id}]"
            )
    # Route-build question/source banks for review hash search.
    for entry in manifest.get("route_builds") or []:
        if isinstance(entry, Mapping):
            for key in ("questions", "sources"):
                if entry.get(key):
                    path = root / str(entry[key])
                    if path.is_file() and path not in source_banks:
                        source_banks.append(path)
    journeys = compose_journeys_from_manifest_entries(
        root=root,
        entries=list(manifest.get("journey_builds") or []),
    )
    output_dir = Path(output_dir)
    site_manifest = build_bootstrap_artifact(
        output_dir=output_dir,
        matter_packets=matters,
        question_packets=questions,
        oulaskangas_group=[
            _resolve(root, rel, rel) for rel in manifest.get("oulaskangas_group") or []
        ],
        oulaskangas_cost=[
            _resolve(root, rel, rel) for rel in manifest.get("oulaskangas_cost") or []
        ],
        oulaskangas_monitoring=[
            _resolve(root, rel, rel) for rel in manifest.get("oulaskangas_monitoring") or []
        ],
        editorial_packet_path=(
            _resolve(root, manifest["editorial_packet"], "editorial_packet")
            if manifest.get("editorial_packet")
            else None
        ),
        editorial_overview_path=(
            _resolve(root, manifest["editorial_overview"], "editorial_overview")
            if manifest.get("editorial_overview")
            else None
        ),
        source_bank_paths=source_banks,
        route_dirs=route_dirs,
        review_paths=reviews,
        reviewed_candidate_paths=reviewed_candidates,
        journey_packets=journeys,
        title=str(manifest.get("title") or "Poliittinen muisti"),
        review_date=str(manifest["review_date"]) if manifest.get("review_date") else None,
    )
    receipt = {
        "schema_version": "paa.integrated_reader_build.v1",
        "manifest_sha256": _sha(manifest_path),
        "inputs_root": str(root),
        "site": {
            "matter_count": site_manifest.get("matter_count"),
            "question_count": site_manifest.get("question_count"),
            "route_count": site_manifest.get("route_count"),
            "journey_count": site_manifest.get("journey_count"),
            "projection_reviewed_journey_count": site_manifest.get(
                "projection_reviewed_journey_count"
            ),
            "file_count": site_manifest.get("file_count"),
            "lookup": site_manifest.get("lookup"),
        },
        "journeys": [
            {
                "task_id": item.get("task_id"),
                "projection_status": item.get("projection_status"),
                "projection_reviewed": item.get("projection_reviewed"),
                "path": item.get("path"),
            }
            for item in site_manifest.get("journeys") or []
        ],
        "db_writes": False,
        "inference": False,
    }
    (output_dir / "INTEGRATED_READER_RECEIPT.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


__all__ = [
    "SCHEMA_VERSION",
    "IntegratedReaderError",
    "build_integrated_reader",
    "compose_journeys_from_manifest_entries",
]
