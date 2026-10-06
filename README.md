# Poliittinen muisti (PAA)

PAA connects public statements and parliamentary records to answer **what people
wanted, what they did, and what the institution decided**. Sources, dates and
exact alternatives remain inspectable. Recorded participation is distinguished
from personal influence, implementation and effects.

The application is a research workbench with selected source-reviewed
investigations. General automated analytical reliability remains unproven.
Model interpretations are candidates; AI source review is not independent human
validation. The reader supplies the political values and judgments.

## Run locally

Install CPython 3.14 and uv, then build the bundled offline example:

```bash
uv sync --locked
uv run --locked paa frozen --root dist/frozen
uv run --locked paa check --db dist/frozen/data/paa.sqlite --slice
uv run --locked python -m http.server 8765 --bind 127.0.0.1 --directory dist/frozen/dist/browser
```

Open **http://127.0.0.1:8765**. The frozen path builds SQLite, compiles evidence
and writes the static browser from bundled inputs. Its small fixture population
is a demonstration, not a complete candidate register. After dependencies are
installed, it needs no network, model server, credentials or sibling repository.
Use a fresh build root; `--overwrite` replaces that selected build's outputs.

For a selected static publication with downloadable evidence:

```bash
uv run --locked paa release --output dist/public-beta --source-revision working-tree
uv run --locked python -m http.server 8765 --bind 127.0.0.1 --directory dist/public-beta
```

This builds the bundled historical examples and `dist/public-beta.zip`. Serve
its files with any static host, including a VPS. For a chosen public destination,
rebuild with `--base-url https://your-host.example/paa/`. Building is local and
does not deploy. Separately prepared investigations require their own retained
inputs; a fresh checkout does not include the complete local research corpus.

See [running and research](docs/RUNNING.md) for acquisition, person/matter queries,
model configuration and portable reader inputs. `uv run --locked paa --help`
lists commands and each command's `--help` describes its options.

## Development and source releases

```bash
uv run --locked ruff check paa tests mev/llm.py
uv run --locked ty check
uv run --locked pytest -q
uv build --out-dir dist/packages
```

The source archive includes maintained code, contracts, fixtures, tests, scripts,
these instructions and CI. The wheel includes application packages and their
runtime assets. Build outputs, live databases, credentials, private research
notes and local caches are excluded. Fixtures are retained because they make the
offline build and regression checks self-contained; code licensing does not
relicense their third-party source material.

`paa/` contains the maintained application; `mev/` contains legislative tooling.
`paa/contracts/` holds evidence schemas and source-shaped fixtures. `ty` checks
only its declared scope in `pyproject.toml`. Browser harnesses in `scripts/`
require Playwright and a browser; they are separate from the Python test suite.
CI also checks the actual archive contents and replays the installed wheel from
a fresh directory using `scripts/paa_package_check.py`.

The maintained documentation has distinct responsibilities:

- [Purpose](docs/PURPOSE.md): useful questions and the project goal.
- [Product contract](docs/SPEC.md): evidence, admission and acceptance.
- [Investigation process](docs/INVESTIGATION_PROCESS.md): research and stopping.
- [Interface contract](docs/UI.md): the minimum useful reader surface.
- [Python profile](docs/PYTHON_PROFILE.md): implementation boundaries.
- [Contribution instructions](AGENTS.md): repository workflow.

Owned code and implementation documentation are [MIT-licensed](LICENSE).
Source data retain separate terms and attribution; see [DATA_NOTICE.md](DATA_NOTICE.md).
