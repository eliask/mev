#!/usr/bin/env python3
"""Check actual release payloads and replay PAA outside a source checkout.

The supplied interpreter must have locked dependencies and the supplied wheel
installed. This check uses isolated Python mode and a fresh caller-owned folder.
It neither contacts sources nor reads private working inputs.
"""

import argparse
import hashlib
import json
import re
import subprocess
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

ALLOWED_ROOTS = {
    "mev", "paa", "tests", "scripts", "docs", ".github", ".gitignore",
    ".python-version", "AGENTS.md", "README.md", "LICENSE", "DATA_NOTICE.md",
    "pyproject.toml", "uv.lock", "PKG-INFO",
}
FORBIDDEN_PARTS = {".tmp", ".git", ".venv", "__pycache__", "llm_cache", ".cache"}
REQUIRED = {
    "pyproject.toml", "uv.lock", "README.md", "AGENTS.md", "LICENSE", "DATA_NOTICE.md",
    "docs/PURPOSE.md", "docs/SPEC.md", "docs/INVESTIGATION_PROCESS.md",
    "docs/UI.md", "docs/PYTHON_PROFILE.md", "docs/RUNNING.md",
    "paa/cli.py", "paa/evidence_site.py", "paa/research_import.py",
    "paa/contracts/fixtures/frozen_sources.jsonl",
    ".github/workflows/quality.yml",
}


def check_payloads(sdist: Path, wheel: Path) -> dict:
    source: dict[str, bytes] = {}
    with tarfile.open(sdist) as archive:
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or len(path.parts) < 2:
                raise ValueError("unsafe source archive member")
            relative = PurePosixPath(*path.parts[1:])
            if relative.parts[0] not in ALLOWED_ROOTS:
                raise ValueError(f"unexpected release root: {relative.parts[0]}")
            if any(part in FORBIDDEN_PARTS or part.startswith('.env') for part in relative.parts):
                raise ValueError(f"private release member: {relative}")
            if member.isdir():
                continue
            if not member.isfile() or str(relative) in source:
                raise ValueError(f"non-file or duplicate release member: {relative}")
            source[str(relative)] = archive.extractfile(member).read()
    if missing := REQUIRED - source.keys():
        raise ValueError(f"missing maintained release inputs: {sorted(missing)}")
    with zipfile.ZipFile(wheel) as archive:
        wheel_payload = {}
        for name in archive.namelist():
            path = PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts or any(
                part in FORBIDDEN_PARTS or part.startswith('.env') for part in path.parts
            ):
                raise ValueError("unsafe or private wheel member")
            if path.parts[0] in {"paa", "mev"}:
                if name in wheel_payload:
                    raise ValueError("duplicate wheel member")
                wheel_payload[name] = archive.read(name)
    runtime = {name: body for name, body in source.items() if name.split('/')[0] in {'paa', 'mev'}}
    if wheel_payload != runtime:
        raise ValueError("wheel runtime code/assets differ from source release")
    for name, body in source.items():
        if name not in {'README.md', 'AGENTS.md', 'DATA_NOTICE.md'} and not name.startswith('docs/'):
            continue
        for target in re.findall(r'\]\(([^)]+)\)', body.decode('utf-8')):
            target = target.split('#')[0]
            if not target or re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', target):
                continue
            parts = []
            for part in (PurePosixPath(name).parent / target).parts:
                if part == '..':
                    if not parts:
                        raise ValueError(f"documentation escapes release: {name}")
                    parts.pop()
                elif part != '.':
                    parts.append(part)
            if '/'.join(parts) not in source:
                raise ValueError(f"broken packaged documentation link: {name} -> {target}")
    return {"source_files": len(source), "runtime_files": len(runtime), "private_members": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdist', type=Path, required=True)
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    args = parser.parse_args()
    report = check_payloads(args.sdist.resolve(), args.wheel.resolve())
    # Keep the venv executable's path: resolving its symlink selects the base
    # interpreter and loses the wheel's installation environment.
    python = args.python.absolute()
    root = args.work_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)

    def run(*arguments: str) -> str:
        result = subprocess.run(
            [str(python), '-I', *arguments], cwd=root,
            check=True, capture_output=True, text=True,
        )
        return result.stdout

    run('-c', """
import json, sys
from pathlib import Path
from paa.config import ROOT, PKG
from mev.config import ROOT as MEV_ROOT
assert ROOT == Path.cwd() and MEV_ROOT == Path.cwd()
assert PKG.is_relative_to(Path(sys.prefix))
print(json.dumps({'installed_package': True, 'caller_owned_workspace': True}))
""")
    run('-m', 'paa.cli', 'frozen', '--root', 'frozen')
    run('-m', 'paa.cli', 'check', '--db', 'frozen/data/paa.sqlite', '--slice')
    run('-m', 'paa.cli', 'release', '--output', 'publication', '--source-revision', 'installed-wheel')
    assert (root / 'frozen/dist/browser/index.html').is_file()
    assert (root / 'publication/index.html').is_file()
    report.update(
        installed_wheel_replay=True,
        frozen_check=True,
        selected_publication=True,
        sdist_sha256=hashlib.sha256(args.sdist.read_bytes()).hexdigest(),
        wheel_sha256=hashlib.sha256(args.wheel.read_bytes()).hexdigest(),
    )
    (root / 'PACKAGE_CHECK.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
