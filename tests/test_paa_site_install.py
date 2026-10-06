from pathlib import Path

import pytest

from paa.frozen import build_frozen
from paa.site import build_from_db


def test_site_rebuild_restores_previous_output_when_install_fails(tmp_path, monkeypatch):
    root = tmp_path / "frozen"
    build_frozen(root=root)
    destination = root / "dist" / "browser"
    prior_index = (destination / "index.html").read_bytes()
    prior_source = next((destination / "objects").glob("*.json"))
    prior_source_bytes = prior_source.read_bytes()
    rename = Path.rename

    def fail_staging_install(path, target):
        if path.name.startswith("paa-site-"):
            raise OSError("injected site install failure")
        return rename(path, target)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "rename", fail_staging_install)
        with pytest.raises(OSError, match="injected site install failure"):
            build_from_db(root / "data" / "paa.sqlite", destination)
    assert (destination / "index.html").read_bytes() == prior_index
    assert prior_source.read_bytes() == prior_source_bytes
    assert not list(destination.parent.glob("paa-site-*"))
    assert not list(destination.parent.glob("paa-previous-site-*"))

    build_from_db(root / "data" / "paa.sqlite", destination)
    assert (destination / "index.html").read_bytes() == prior_index
    assert prior_source.read_bytes() == prior_source_bytes
