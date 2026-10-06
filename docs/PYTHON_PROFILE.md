# Python implementation profile

## Scope and runtime

This self-contained profile governs new and materially changed MeV/PAA Python.
It adopts strict implementation discipline for evidence processing without
claiming whole-repository conformance to another project's standard. Existing
violations are migration debt; unrelated cleanup is not a prerequisite to
answering a useful question.

Use CPython 3.14, the interpreter pin, locked uv environment and configured Ruff
and pytest gates. New maintained code uses deferred annotations without
`from __future__ import annotations`; verify consumers of annotations when
migrating old code. Use built-in generics and `T | None`. The configured ty scope
currently comprises `paa/opportunity_records.py` and `paa/opportunity_codec.py`.
A green scoped check is not a whole-application typing claim.

## Profiles selected by responsibility

**Semantic:** code deciding identity, scope, temporal meaning, relations,
denominators, authority, assessment and admission. Use closed nominal records
with explicit policy/source/time inputs and owned immutable state at maintained
boundaries. Default new records are final, frozen, slotted dataclasses with
keyword-only construction and disabled positional matching. Validate legitimate
states; frozen outer objects must not retain mutable caller aliases. Equality,
ordering, multiplicity and identity are domain decisions.

**Boundary:** HTTP/XML/JSON, SQL, model transport, files, clocks, environment,
orchestration and public rendering. Dynamic external data is allowed locally;
validate it before a semantic consumer relies on it. Expected missingness,
ambiguity, truncation, cancellation and transport failure remain distinct.
Programming defects remain exceptional. Broad containment belongs only at a
named external-worker boundary with the original job and failure preserved.

**Tooling:** exploration, notebooks and benchmark drivers may use pragmatic
Python. When their outputs become maintained evidence or public meaning, their
producer and consumer need the relevant boundary/semantic contract. Do not build
a second production path around an unchecked experiment.

## Retention and execution

Preserve exact source bytes/text and named normalization projections. Do not
collapse whitespace in the only retained witness. Use explicit retained codecs,
validation and stable versioned identities; Python hashes or positional list
indices are not evidence identities. Reject duplicate/invalid required fields
and unresolved references according to the declared codec. Define numeric units,
denominators, finite values and rounding wherever they affect meaning.

Account for the declared population, with deterministic merge ordering and
explicit duplicate/missing item checks. A live cursor or unfinished task set is
not a completed result. Async effects own bounded concurrency, retry histories
and cancellation propagation. Retained writes use transactions or atomic file
replacement as appropriate. Source text/model output must not execute as code;
parameterize SQL, sanitize public HTML and protect credentials.

A model cache includes all interpretation-relevant model/session identity,
source, exact input, prompt, template, settings, parser version and output limits.
Validate reused responses. Corruption is an explicit miss/failure, not trusted
reuse. Retained-response replay and fresh inference are different checks.

Revise interpretations through explicit successors and revalidate dependent
results. Implementation exactness does not settle empirical correctness. Use a
bug witness, a valid control and the maintained producer-to-consumer path for
load-bearing changes; extend migration only where a real consumer benefits.
