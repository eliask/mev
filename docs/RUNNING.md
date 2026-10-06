# Running and research

The [README](../README.md) contains the self-contained offline build and release
commands. This page covers optional work with acquired data and prepared sources.
None of these paths requires private working notes or a sibling checkout.

## Public-record acquisition and queries

Adapters support candidate rosters, specified Yle campaign material, parliamentary
members, ballots, initiatives, questions, government answers and speeches. Each
slice declares its period, provenance, completeness and failures. A fresh checkout
contains fixtures, not complete live datasets.

```bash
uv run --locked paa acquire
uv run --locked paa ballots --year 2023
uv run --locked paa initiatives --year 2023 --partition-identifiers
uv run --locked paa questions --year 2023
uv run --locked paa speeches --year 2023
uv run --locked paa compile
uv run --locked paa check
uv run --locked paa site
```

Live inputs default to `data/`; browser output defaults to `dist/browser/`.
Inspect command help before choosing a different period or output. Sources may
be unavailable or incomplete; failed acquisition is not evidence of no action.

With an acquired database, exact matter queries are read-only:

```bash
uv run --locked paa matter --db data/paa.sqlite --matter 'HE 131/2024 vp' --output dist/matter.json --html dist/matter.html
uv run --locked paa matter --db data/paa.sqlite --matter 'HE 131/2024 vp' --actor-name 'Anna Kontula'
uv run --locked paa matter --db data/paa.sqlite --search 'diesel'
```

Matter joins use official identifiers. Text search retrieves candidates and does
not establish policy equivalence. An original speech is not automatically a
reviewed position; first signature does not prove drafting; a package ballot does
not establish agreement with each component. Historical group comparisons use
ballot-event membership and exclude the target member from the peer majority.

## Model-assisted research

The local client uses `LLAMA_API_BASE`, default `http://127.0.0.1:8080`.
It retains request configuration, sources, outputs and failures. Compilation
keeps model interpretations proposed.

```bash
uv run --locked paa llm extract --output data/llm_runs/my-run --compact --prompt extract_batch_multi_v2 --batch-size 16 --concurrency 2
uv run --locked paa compile --llm-run data/llm_runs/my-run
```

For a self-contained question/source bank, select tasks explicitly:

```bash
uv run --locked paa source-question --questions path/questions.json --sources path/sources.json --split dev --output dist/answers
uv run --locked paa research-import --questions path/questions.json --sources path/sources.json --submission path/submission.json --output dist/answer.json
```

The import uses `paa.research_submission.v1`: task and researcher IDs, a
timezone-bearing production date, reading scope and the existing result schema.
Indexed line/window/ballot pointers bind claims to exact supplied text. A valid
import remains a candidate; binding does not certify meaning or completeness.
Optional database context is labelled outside the supplied citation universe;
`--no-db-context` keeps the inquiry within its supplied bundle.

Remote inference requires explicit provider selection, such as `--provider muse`,
and a literal credential file supplied with `--key-file` or `OPENCODE_KEY_FILE`.
There is no automatic provider fallback. Send only authorized public material;
keep credentials outside source and output artifacts. `--context-limit-tokens`
is a caller planning limit, not a provider-capacity claim.

## Portable readers and reviewed projections

An integrated reader takes explicit portable inputs. Paths resolve relative to
the manifest directory or an explicit input root:

```bash
uv run --locked paa integrated-reader --manifest path/inputs_manifest.json --output dist/reader
```

The manifest selects matter packets, question packets, source banks, reviews and
optional `journey_builds`. Each journey names a route packet, mapping, optional
answer, source review and projection review. Source review does not admit graph
meaning: a `paa.journey_projection_review.v1` receipt separately binds the
mapping, candidate and source-bank versions. Unreviewed projections stay labelled.

For bundles that also need route generation and ballot enrichment before
composition, use the offline helper:

```bash
uv run --locked python scripts/paa_preview_offline.py --inputs path/inputs --output dist/reader-replay
```

The input directory must supply its `inputs_manifest.json` and retained files.
The helper does not acquire missing inputs, call models or modify a live database.
Use `paa route --help`, `paa journey --help` and `paa source-revision --help` for
individual stages. `paa purpose-demo` builds additional bundled attribution and
monitoring examples; these examples are not estimates of general reliability.

## Content and source terms

Keep source-data terms separate from the code license, including when packaging
new reader inputs. Yle's 2011 dataset is attributed to **Yle Uutisten vaalikone
2011**, under **CC BY-NC-SA 3.0**, as recorded by its
[data publication](https://yle.fi/aihe/a/20-162059). Other Yle records do not inherit
that license automatically. See [DATA_NOTICE.md](../DATA_NOTICE.md).

## Legislative tooling

The `mev` CLI is an optional legislative pipeline; the offline PAA build does not
require its data or a LawVM checkout. `mev --help` lists its stages. Pipeline
workspaces default to the source project root, or the current directory when
installed from a wheel. Configure `MEV_WORKSPACE_ROOT` explicitly to reuse a
particular workspace and `MEV_SHARED_ROOT` for a shared legislative index.

Finlex bulk inputs default to `data/statute.zip` and
`data/government-proposal.zip`. Set `STATUTE_ZIP` and `AKN_ZIP_PATH` to use archives
elsewhere. A LawVM integration requires `LAWVM_DIR` to name that checkout; it is
not discovered in a private parent tree. The HE atom parser also requires the
native Voikko library and Finnish dictionary; the Python `libvoikko` dependency
alone does not provide them.
