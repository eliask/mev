#!/usr/bin/env python3
"""Offline dossier preview composition from a portable inputs directory.

The inputs directory is ordinary bundled records (JSON banks, retained matter
snapshots, optional route_builds / route dirs, review receipts). No ``.tmp``
path is hardcoded; pass ``--inputs`` explicitly. Does not call providers, write
the live database, deploy, commit, or push.

Manifest contract (relative paths under ``--inputs``):

- ``retained_matters``, ``witness_bank``, ``comparison``
- optional ``kkv_bank``, ``committee_bank`` + ``committee_matter_ids``
- optional ``route_builds``: list of
  ``{id, questions, sources, answer, task_id?, as_of?}`` → generated route dirs
- optional ``route_dirs``: prebuilt route directories (in addition to builds)
- ``question_packets``, ``source_banks``, ``reviews``
- optional ``reviewed_candidates``: ``{task_id: relative_candidate_path}``
  (otherwise taken from each review's ``candidate_file`` / ``candidate_path``)
- optional Oulaskangas / editorial paths
- optional ``journey_builds``: list of
  ``{id?, route_packet|route_dir+task_id, mapping, answer?, source_answer_review?,
  projection_review?}`` → ``journeys/`` in the same site (UNREVIEWED unless
  projection_review). ``route_dir`` may name ``generated_routes/<id>`` from this
  build's ``route_builds`` output, or a directory under ``--inputs``.
"""

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(path: Path, label: str) -> Path:
    if not path.exists():
        raise SystemExit(f"missing {label}: {path}")
    return path


def _safe_id(value: str) -> str:
    return re.sub(r"[^\w.-]+", "_", value) or "route"


def _generate_route_dirs(
    *,
    inputs_dir: Path,
    output_dir: Path,
    route_builds: list[Any],
) -> list[Path]:
    """Build selected route directories from bundled candidate/questions/sources."""

    from paa.route_packet import build_route_site

    generated: list[Path] = []
    for index, entry in enumerate(route_builds):
        if not isinstance(entry, dict):
            raise SystemExit(f"route_builds[{index}] must be an object")
        build_id = str(entry.get("id") or f"route_{index}")
        questions = _require(inputs_dir / str(entry["questions"]), f"route_builds[{build_id}].questions")
        sources = _require(inputs_dir / str(entry["sources"]), f"route_builds[{build_id}].sources")
        answer = _require(inputs_dir / str(entry["answer"]), f"route_builds[{build_id}].answer")
        task_id = entry.get("task_id")
        if not task_id:
            answer_doc = _load(answer)
            task_id = answer_doc.get("task_id")
        if not task_id:
            raise SystemExit(f"route_builds[{build_id}] missing task_id")
        as_of = entry.get("as_of")
        out = output_dir / "generated_routes" / _safe_id(str(build_id))
        if out.exists() and any(out.iterdir()):
            raise SystemExit(f"generated route output not empty: {out}")
        build_route_site(
            questions=questions,
            sources=sources,
            output_dir=out,
            task_id=str(task_id),
            as_of=str(as_of) if as_of else None,
            answer=answer,
            overwrite=False,
        )
        generated.append(out)
    return generated


def _reviewed_candidate_map(
    *,
    inputs_dir: Path,
    manifest: Mapping[str, Any],
    reviews: list[Path],
) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    declared = manifest.get("reviewed_candidates") or {}
    if isinstance(declared, dict):
        for task_id, rel in declared.items():
            mapping[str(task_id)] = _require(
                inputs_dir / str(rel), f"reviewed_candidates[{task_id}]"
            )
    for review_path in reviews:
        receipt = _load(review_path)
        task_id = str(receipt.get("task_id") or "")
        if not task_id or task_id in mapping:
            continue
        for key in ("candidate_file", "candidate_path", "candidate"):
            rel = receipt.get(key)
            if not rel:
                continue
            path = Path(str(rel))
            if not path.is_file():
                path = (review_path.parent / str(rel)).resolve()
            if path.is_file():
                mapping[task_id] = path
                break
    return mapping


def build_from_inputs(*, inputs_dir: Path, output_dir: Path) -> dict[str, Any]:
    from paa.ballot_snapshot_enrichment import enrich_matter_files
    from paa.bootstrap_site import build_bootstrap_artifact
    from paa.matter_ledger import project_retained_matter_packet_for_current_emit
    from paa.matter_sourcebank_adapters import (
        adapt_committee_role_sourcebank,
        adapt_kkv_response_sourcebank,
        merge_matter_packets,
    )

    inputs_dir = Path(inputs_dir).resolve()
    output_dir = Path(output_dir).resolve()
    manifest_path = _require(inputs_dir / "inputs_manifest.json", "inputs_manifest.json")
    manifest = _load(manifest_path)

    matters_in = [
        _require(inputs_dir / rel, f"matter {rel}")
        for rel in manifest.get("retained_matters") or []
    ]
    if len(matters_in) != 9:
        raise SystemExit(f"expected 9 retained matters, got {len(matters_in)}")

    witness = _require(
        inputs_dir / str(manifest["witness_bank"]),
        "witness_bank",
    )
    comparison = _require(
        inputs_dir / str(manifest["comparison"]),
        "comparison",
    )
    enriched_dir = output_dir / "matters_enriched"
    enrich_manifest = enrich_matter_files(
        matter_paths=matters_in,
        witness_bank_path=witness,
        output_dir=enriched_dir,
        comparison_path=comparison,
    )

    immutable_dir = output_dir / "matters_immutable9"
    immutable_dir.mkdir(parents=True, exist_ok=True)
    projected_paths: list[Path] = []
    status_counts: dict[str, int] = {}
    for path in sorted(enriched_dir.glob("*.json")):
        if path.name in {
            "manifest.json",
            "SUMMARY.json",
            "portable_ballot_witness_bank.json",
        }:
            continue
        raw = path.read_bytes()
        packet = json.loads(raw.decode("utf-8"))
        projected = project_retained_matter_packet_for_current_emit(
            packet, retained_bytes_sha256=hashlib.sha256(raw).hexdigest()
        )
        out_path = immutable_dir / path.name
        out_path.write_text(
            json.dumps(projected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        projected_paths.append(out_path)
        for gap in projected.get("missingness") or []:
            code = str((gap or {}).get("code") or "MISSINGNESS")
            status_counts[code] = status_counts.get(code, 0) + 1
        for cov in projected.get("coverage") or []:
            state = str((cov or {}).get("state") or "COVERAGE")
            status_counts[f"coverage:{state}"] = status_counts.get(f"coverage:{state}", 0) + 1

    # Optional KKV + committee merges into matching matter packets.
    kkv_rel = manifest.get("kkv_bank")
    if kkv_rel:
        kkv_packet = adapt_kkv_response_sourcebank(_load(inputs_dir / str(kkv_rel)))
        kkv_matter = str((kkv_packet.get("matter") or {}).get("matter_id") or "")
        for path in projected_paths:
            packet = _load(path)
            if str((packet.get("matter") or {}).get("matter_id") or "") == kkv_matter:
                merged = merge_matter_packets(packet, kkv_packet)
                path.write_text(
                    json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                break
        else:
            kkv_out = immutable_dir / "KK_1_2023_vp.sourcebank.json"
            kkv_out.write_text(
                json.dumps(kkv_packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            projected_paths.append(kkv_out)

    committee_rel = manifest.get("committee_bank")
    if committee_rel:
        bank = _load(inputs_dir / str(committee_rel))
        for matter_id in manifest.get("committee_matter_ids") or []:
            adapted = adapt_committee_role_sourcebank(bank, matter_id=matter_id)
            for path in projected_paths:
                packet = _load(path)
                if str((packet.get("matter") or {}).get("matter_id") or "") == matter_id:
                    merged = merge_matter_packets(packet, adapted)
                    path.write_text(
                        json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    break

    generated_routes = _generate_route_dirs(
        inputs_dir=inputs_dir,
        output_dir=output_dir,
        route_builds=list(manifest.get("route_builds") or []),
    )
    route_dirs = [
        _require(inputs_dir / rel, f"route-dir {rel}")
        for rel in manifest.get("route_dirs") or []
    ] + generated_routes
    reviews = [
        _require(inputs_dir / rel, f"review {rel}")
        for rel in manifest.get("reviews") or []
    ]
    questions = [
        _require(inputs_dir / rel, f"question {rel}")
        for rel in manifest.get("question_packets") or []
    ]
    source_banks = [
        _require(inputs_dir / rel, f"source-bank {rel}")
        for rel in manifest.get("source_banks") or []
    ]
    # Route/question source banks used for review SHA binding.
    for entry in manifest.get("route_builds") or []:
        if isinstance(entry, dict):
            for key in ("questions", "sources"):
                if entry.get(key):
                    path = inputs_dir / str(entry[key])
                    if path.is_file() and path not in source_banks:
                        source_banks.append(path)
    reviewed_candidates = _reviewed_candidate_map(
        inputs_dir=inputs_dir, manifest=manifest, reviews=reviews
    )
    journeys = []
    if manifest.get("journey_builds"):
        from paa.integrated_reader import compose_journeys_from_manifest_entries

        # Journey route_dir may name this build's generated_routes/ under --output;
        # mapping/answer/review stay relative to the portable inputs root.
        generated_by_rel = {
            str(path.relative_to(output_dir)).replace("\\", "/"): path
            for path in generated_routes
        }
        generated_by_task: dict[str, Path] = {}
        for path in generated_routes:
            for packet_path in sorted(path.glob("*.json")):
                if packet_path.name == "manifest.json":
                    continue
                task = str(_load(packet_path).get("task_id") or "")
                if task and task not in generated_by_task:
                    generated_by_task[task] = path
        journey_entries: list[dict[str, Any]] = []
        for index, entry in enumerate(manifest.get("journey_builds") or []):
            if not isinstance(entry, dict):
                raise SystemExit(f"journey_builds[{index}] must be an object")
            item = dict(entry)
            if not item.get("route_packet"):
                resolved: Path | None = None
                if item.get("route_dir"):
                    rel = str(item["route_dir"]).replace("\\", "/").lstrip("./")
                    if rel in generated_by_rel:
                        resolved = generated_by_rel[rel]
                    elif (inputs_dir / rel).is_dir():
                        resolved = inputs_dir / rel
                    elif (output_dir / rel).is_dir():
                        resolved = output_dir / rel
                if resolved is None and item.get("task_id"):
                    resolved = generated_by_task.get(str(item["task_id"]))
                if resolved is not None:
                    item["route_dir"] = str(resolved)
            journey_entries.append(item)
        journeys = compose_journeys_from_manifest_entries(
            root=inputs_dir,
            entries=journey_entries,
        )
        # Optional ordinary reader titles (explicit data; does not admit projection).
        overrides_rel = manifest.get("reader_title_overrides")
        if overrides_rel:
            overlay = _load(_require(inputs_dir / str(overrides_rel), "reader_title_overrides"))
            titles = overlay.get("titles") if isinstance(overlay, dict) else None
            if isinstance(titles, dict):
                for journey in journeys:
                    task_id = str(journey.get("task_id") or "")
                    entry = titles.get(task_id)
                    if not isinstance(entry, dict):
                        continue
                    title = str(entry.get("editorial_title") or "").strip()
                    if not title:
                        continue
                    journey["editorial_title"] = title
                    if entry.get("prefer_editorial_title") is True or str(
                        entry.get("source") or ""
                    ).startswith("editorial_successor"):
                        journey["prefer_editorial_title"] = True
    site_dir = output_dir / "site"
    site_manifest = build_bootstrap_artifact(
        output_dir=site_dir,
        matter_packets=projected_paths,
        question_packets=questions,
        oulaskangas_group=[
            _require(inputs_dir / rel, rel)
            for rel in manifest.get("oulaskangas_group") or []
        ],
        oulaskangas_cost=[
            _require(inputs_dir / rel, rel) for rel in manifest.get("oulaskangas_cost") or []
        ],
        oulaskangas_monitoring=[
            _require(inputs_dir / rel, rel)
            for rel in manifest.get("oulaskangas_monitoring") or []
        ],
        editorial_packet_path=(
            inputs_dir / str(manifest["editorial_packet"])
            if manifest.get("editorial_packet")
            else None
        ),
        editorial_overview_path=(
            inputs_dir / str(manifest["editorial_overview"])
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
        "schema_version": "paa.bs02_preview_offline.v3",
        "inputs_manifest_sha256": _sha(manifest_path),
        "enrichment": enrich_manifest,
        "immutable_matter_count": len(projected_paths),
        "generated_route_dirs": [str(path.relative_to(output_dir)) for path in generated_routes],
        "source_gap_bound_status_counts": status_counts,
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
            "reviewed_route_count": sum(
                1 for row in site_manifest.get("routes") or [] if row.get("ai_source_review")
            ),
            "reviewed_question_count": sum(
                1 for row in site_manifest.get("questions") or [] if row.get("ai_source_review")
            ),
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
    (output_dir / "OFFLINE_BUILD_RECEIPT.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        type=Path,
        required=True,
        help="Portable inputs directory containing inputs_manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Unique empty output directory for immutable matters + site",
    )
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    receipt = build_from_inputs(inputs_dir=Path(args.inputs), output_dir=output)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
