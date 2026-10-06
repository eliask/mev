"""Release relocation must not send data into an installation's parent tree."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[1] / "mev/config.py"
PATH_NAMES = (
    "MEV_WORKSPACE_ROOT", "MEV_SHARED_ROOT", "LAWVM_DIR", "STATUTE_ZIP", "AKN_ZIP_PATH",
)


def _paths(config: Path, cwd: Path, overrides: dict[str, str] | None = None) -> dict:
    environment = {key: value for key, value in os.environ.items() if key not in PATH_NAMES}
    environment.update(overrides or {})
    result = subprocess.run(
        [sys.executable, "-c", """
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location('relocated_config', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(json.dumps({name: str(getattr(module, name)) for name in (
    'ROOT', 'BOOK_ROOT', 'LAWVM_DIR', 'HE_DB_DIR', 'INDEX_DB', 'STATUTE_ZIP', 'AKN_ZIP_PATH'
)}))
""", str(config)],
        cwd=cwd, env=environment, check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)


def test_source_checkout_owns_its_workspace_even_from_another_directory(tmp_path):
    project = tmp_path / "project"
    package = project / "mev"
    package.mkdir(parents=True)
    shutil.copyfile(CONFIG, package / "config.py")
    (project / "pyproject.toml").write_text('[project]\nname="example"\n')
    caller = tmp_path / "caller"
    caller.mkdir()
    paths = _paths(package / "config.py", caller)
    assert paths["ROOT"] == str(project)
    assert all(Path(value).is_relative_to(project) for value in paths.values())


def test_installed_package_owns_callers_workspace_not_site_packages(tmp_path):
    package = tmp_path / "site-packages/mev"
    package.mkdir(parents=True)
    shutil.copyfile(CONFIG, package / "config.py")
    caller = tmp_path / "caller"
    caller.mkdir()
    paths = _paths(package / "config.py", caller)
    assert paths["ROOT"] == str(caller)
    assert all(Path(value).is_relative_to(caller) for value in paths.values())


def test_explicit_workspace_archives_and_shared_index_are_respected(tmp_path):
    workspace = tmp_path / "workspace"
    shared = tmp_path / "shared"
    lawvm = tmp_path / "legal-replay"
    statute = tmp_path / "archives/statute.zip"
    proposals = tmp_path / "archives/government-proposal.zip"
    paths = _paths(CONFIG, tmp_path, dict(zip(PATH_NAMES, map(str, (
        workspace, shared, lawvm, statute, proposals,
    )), strict=True)))
    assert paths["ROOT"] == str(workspace)
    assert paths["HE_DB_DIR"] == str(workspace / ".tmp/he_dbs")
    assert paths["INDEX_DB"] == str(shared / "data/legislative_index.sqlite")
    assert paths["LAWVM_DIR"] == str(lawvm)
    assert paths["STATUTE_ZIP"] == str(statute)
    assert paths["AKN_ZIP_PATH"] == str(proposals)


def test_local_finlex_reads_the_configured_proposal_archive(tmp_path):
    import zipfile

    archive = tmp_path / "proposals.zip"
    xml = b'<act><title>Source archive control</title></act>'
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("akn/fi/doc/government-proposal/2024/12/fin@/main.xml", xml)
    environment = {key: value for key, value in os.environ.items() if key not in PATH_NAMES}
    environment.update(MEV_WORKSPACE_ROOT=str(tmp_path), AKN_ZIP_PATH=str(archive))
    result = subprocess.run(
        [sys.executable, "-c", """
from mev.pipeline.legislative_index import LocalFinlex
from mev.config import ROOT
from mev.llm import CACHE_DIR, _cache
assert CACHE_DIR == ROOT / 'data/mev_llm_cache'
_cache.close()
source = LocalFinlex()
try:
    assert source.get_he(2024, 12) == b'<act><title>Source archive control</title></act>'
finally:
    source.close()
"""],
        env=environment, check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
