#!/usr/bin/env python3
"""Offline question-journey composition from route packets and explicit mappings.

No ``.tmp`` path is hardcoded; pass ``--manifest`` or individual builds. Does not
call providers, write the live database, deploy, commit, or push. Projection
mappings remain UNREVIEWED unless a projection review path is supplied.

Manifest JSON schema (object):
  journey_builds: list of {
    id?,
    route_packet,
    mapping,
    answer?,
    source_answer_review?,
    projection_review?
  }
Paths are relative to the manifest file's directory unless absolute.
"""

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(path: Path, label: str) -> Path:
    if not path.is_file():
        raise SystemExit(f"missing {label}: {path}")
    return path


def _resolve(base: Path, value: object, label: str) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = base / path
    return _require(path, label)


def build_journeys_from_manifest(*, manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    from paa.question_journey import build_journey_from_paths, write_journey_site

    manifest_path = Path(manifest_path).resolve()
    base = manifest_path.parent
    manifest = _load(manifest_path)
    if not isinstance(manifest, Mapping):
        raise SystemExit("manifest must be a JSON object")
    builds = list(manifest.get("journey_builds") or [])
    if not builds:
        raise SystemExit("manifest.journey_builds is empty")
    journeys = []
    build_receipts = []
    for index, entry in enumerate(builds):
        if not isinstance(entry, Mapping):
            raise SystemExit(f"journey_builds[{index}] must be an object")
        build_id = str(entry.get("id") or entry.get("task_id") or f"journey_{index}")
        route_packet = _resolve(base, entry["route_packet"], f"{build_id}.route_packet")
        mapping = _resolve(base, entry["mapping"], f"{build_id}.mapping")
        answer = (
            _resolve(base, entry["answer"], f"{build_id}.answer")
            if entry.get("answer")
            else None
        )
        source_review = (
            _resolve(base, entry["source_answer_review"], f"{build_id}.source_answer_review")
            if entry.get("source_answer_review")
            else None
        )
        projection_review = (
            _resolve(base, entry["projection_review"], f"{build_id}.projection_review")
            if entry.get("projection_review")
            else None
        )
        packet = build_journey_from_paths(
            route_packet_path=route_packet,
            mapping_path=mapping,
            answer_path=answer,
            source_answer_review_path=source_review,
            projection_review_path=projection_review,
        )
        journeys.append(packet)
        build_receipts.append(
            {
                "id": build_id,
                "task_id": packet.get("task_id"),
                "projection_status": packet.get("projection_status"),
                "admission": packet.get("admission"),
                "mapping_sha256": _sha(mapping),
                "route_packet_sha256": _sha(route_packet),
                "answer_sha256": _sha(answer) if answer else None,
            }
        )
    site_dir = output_dir / "site"
    site_manifest = write_journey_site(
        journeys,
        site_dir,
        title=str(manifest.get("title") or "Kysymysmatkat"),
        overwrite=False,
    )
    # Retain composed packets beside the site for review handoff.
    packets_dir = output_dir / "packets"
    packets_dir.mkdir(parents=True, exist_ok=True)
    for packet in journeys:
        name = str(packet.get("task_id") or "journey").replace("/", "_") + ".json"
        (packets_dir / name).write_text(
            json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    receipt = {
        "schema_version": "paa.journey_offline.v1",
        "manifest_sha256": _sha(manifest_path),
        "builds": build_receipts,
        "site": site_manifest,
        "unreviewed_projection_count": sum(
            1 for row in build_receipts if row.get("projection_status") == "UNREVIEWED"
        ),
        "db_writes": False,
        "inference": False,
    }
    (output_dir / "JOURNEY_OFFLINE_RECEIPT.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Journey build manifest JSON (journey_builds list)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Unique empty output directory for packets + site",
    )
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    receipt = build_journeys_from_manifest(manifest_path=Path(args.manifest), output_dir=output)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
