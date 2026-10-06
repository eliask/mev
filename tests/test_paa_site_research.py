"""Research exports cannot be presented as admitted canonical conclusions."""

import json

import pytest

from paa.research_site import RESEARCH_STATUS, SCHEMA_VERSION
from paa.site import _write_research_packets


def test_research_export_rejects_admission_and_keeps_exact_packet(tmp_path):
    packet = {
        "schema_version": SCHEMA_VERSION,
        "research_status": RESEARCH_STATUS,
        "not_model_admission": True,
        "label": "Lähteiden vertailu",
        "case_count": 1,
        "episodes": [{"episode_id": "test", "modes": {
            "baseline": {"admission_state": "ADMITTED", "answer": "A false promotion"},
        }}],
    }
    path = tmp_path / "packet.json"
    path.write_text(json.dumps(packet), encoding="utf-8")
    with pytest.raises(ValueError, match="non-proposal"):
        _write_research_packets([path], tmp_path / "rejected")
    assert not list((tmp_path / "rejected").glob("*.html"))

    packet["episodes"][0]["modes"]["baseline"]["admission_state"] = "PROPOSED / NOT_ADMITTED"
    path.write_text(json.dumps(packet), encoding="utf-8")
    output = tmp_path / "research"
    _write_research_packets([path], output)
    assert next(output.glob("*.json")).read_bytes() == path.read_bytes()
    assert "eivät hyväksyttyjä päätelmiä" in (output / "index.html").read_text()
    page = next(item for item in output.glob("*.html") if item.name != "index.html")
    assert "Lataa lähdepaketti" in page.read_text()
