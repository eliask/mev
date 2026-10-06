# Poliittinen muisti (PAA)

PAA investigates Finnish political action and public decisions using inspectable
source records. It asks what an actor said and did, where a policy changed, and
what happened to a particular objection or proposed repair. The voter supplies
the value judgment.

The current application is a public-record research workbench with bounded,
source-reviewed investigations. **General automated analytical reliability remains
unproven.** Reviewed cases include AI source readings, which do not constitute
independent human validation. Model outputs are candidates; a successful request
or exact quotation does not admit a finding. See the [product contract](docs/SPEC.md)
for requirements and the distinction between implemented capability and the
acceptance target.

## Build the offline example

CPython 3.14 and uv are required.

```bash
uv sync --locked
uv run --locked pytest -q
uv run --locked paa frozen --root dist/frozen
uv run --locked paa check --db dist/frozen/data/paa.sqlite --slice
uv run --locked python -m http.server 8765 --bind 127.0.0.1 --directory dist/frozen/dist/browser
```

Open `http://127.0.0.1:8765`. **Tutkittavat päätökset** contains the question-first
investigations. Person pages provide statements, actions and source links.
The frozen build uses bundled source records to create SQLite, compile evidence
traces and write the static browser. It needs no network after dependency
installation and no model server or sibling repository. Use a new `--root` for
another build; `--overwrite` deliberately replaces that selected build's outputs.

## Sources and live acquisition

Adapters support official candidate rosters, named Yle campaign material,
parliamentary members, votes and ballots, legislative initiatives, written
questions and government-answer records, and parliamentary speeches. Each
acquired slice records its own period, provenance, completeness and failures.
These adapters do not imply that a fresh checkout includes the complete datasets.

```bash
uv run paa acquire
uv run paa ballots --year 2023
uv run paa initiatives --year 2023 --partition-identifiers
uv run paa questions --year 2023
uv run paa speeches --year 2023
uv run paa compile
uv run paa check
uv run paa site
```

Use `uv run paa --help` and a command's `--help` for supported options. Live data
is stored under `data/`; generated pages under `dist/browser/`. Network sources
can be unavailable or incomplete. Metadata-only answers do not establish a
substantive response. First signatory does not prove drafting; enactment does
not prove implementation or effects. The browser exposes those limits.

The Yle 2011 material is attributed to **Yle Uutisten vaalikone 2011**, under
**CC-BY-NC-SA 3.0**, as recorded by its [data publication](https://yle.fi/aihe/a/20-162059).
Other source records retain their own attribution and rights metadata. No code
license is currently declared by this repository.

## Model-assisted research

The local client uses `LLAMA_API_BASE` (default `http://127.0.0.1:8080`). Source
extraction and inquiry tools retain request settings, source versions, outputs,
failures and truncation. Compilation keeps model interpretations proposed.

```bash
uv run paa llm extract --output data/llm_runs/my-run \
  --compact --prompt extract_batch_multi_v2 --batch-size 16 --concurrency 2
uv run paa compile --llm-run data/llm_runs/my-run
```

The generic source-comparison driver is `python -m paa.structure_probe --help`.
Remote inference requires explicit provider selection and `OPENCODE_KEY_FILE`
pointing to a literal credential file. It has no automatic model fallback.
Only authorized public source material may be sent to a remote provider.
Indexed source-span selections let code construct exact quotations; they do not
prove relevance, speaker attribution or the truth of an interpretation.

## Development

[AGENTS.md](AGENTS.md) contains the contribution rules; the
[Python profile](docs/PYTHON_PROFILE.md) defines implementation boundaries.
Run the same gates as CI:

```bash
uv run --locked ruff check paa tests mev/llm.py
uv run --locked ty check
uv run --locked pytest -q
```

`ty` currently checks the opportunity record/codec boundary, not the entire
application. Contracts and source-shaped regression fixtures are under
`paa/contracts/`. `paa/` is the maintained application; `mev/` contains earlier
legislative tooling. The frozen PAA build does not import MeV detector conclusions.
The browser checks in `scripts/` exercise record flows and candidate source
navigation using Playwright; set the harness's URL, browser executable and output
options for your environment.
