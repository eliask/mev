"""Focused exact-source pointer tests."""

import hashlib

import pytest

from paa.source_pointers import (
    CHUNK_LINE,
    CHUNK_SENTENCE,
    SourcePointerError,
    build_source_pointer_index,
)


def test_line_index_preserves_headings_and_covers_original_text_exactly() -> None:
    source = "JOHDANTO\nEnsimmäinen rivi.\n\nLoppu ilman rivinvaihtoa"
    index = build_source_pointer_index("source-1", source, granularity="line")

    assert [span.pointer_id for span in index.spans] == list(range(len(index.spans)))
    assert [span.kind for span in index.spans] == [CHUNK_LINE] * len(index.spans)
    assert "".join(span.text for span in index.spans) == source
    assert index.spans[0].text == "JOHDANTO\n"
    assert index.spans[-1].text == "Loppu ilman rivinvaihtoa"
    assert index.source_text_sha256 == hashlib.sha256(source.encode("utf-8")).hexdigest()
    assert index.indexed_text().startswith("[0] JOHDANTO\n")


def test_sentence_index_keeps_exact_whitespace_and_repeated_text_has_distinct_ids() -> None:
    source = "Otsikko\nSama lause. Sama lause.\n"
    index = build_source_pointer_index("source-2", source, granularity="sentence")

    assert "".join(span.text for span in index.spans) == source
    assert index.spans[0].text == "Otsikko\n"
    repeated = [span for span in index.spans if "Sama lause." in span.text]
    assert len(repeated) == 2
    assert repeated[0].pointer_id != repeated[1].pointer_id
    assert repeated[0].text == "Sama lause."
    assert repeated[1].text == " Sama lause.\n"
    assert repeated[0].kind == CHUNK_SENTENCE

    first = index.resolve(
        span_start=repeated[0].pointer_id,
        span_end=repeated[0].pointer_id,
        source_text_sha256=index.source_text_sha256,
    )
    second = index.resolve(
        span_start=repeated[1].pointer_id,
        span_end=repeated[1].pointer_id,
        source_text_sha256=index.source_text_sha256,
    )
    assert first.quote == "Sama lause."
    assert second.quote == " Sama lause.\n"
    assert first.start != second.start
    assert first.to_anchor_input() == {"quote": "Sama lause.", "start": first.start}


def test_contiguous_range_returns_exact_anchor() -> None:
    source = "A. B. C."
    index = build_source_pointer_index("source-3", source, granularity="sentence")
    assert [span.text for span in index.spans] == ["A.", " B.", " C."]

    combined = index.resolve(
        span_start=0,
        span_end=1,
        source_text_sha256=index.source_text_sha256,
    )
    assert combined.to_anchor_input() == {"quote": "A. B.", "start": 0}

    # Non-contiguous references remain separate calls/records; the resolver
    # has no operation which can silently concatenate them.
    first = index.resolve(span_start=0, span_end=0, source_text_sha256=index.source_text_sha256)
    third = index.resolve(span_start=2, span_end=2, source_text_sha256=index.source_text_sha256)
    assert [first.quote, third.quote] == ["A.", " C."]
    assert first.start == 0 and third.start == 5


@pytest.mark.parametrize(
    ("span_start", "span_end"),
    [
        (True, 0),
        (0, False),
        (-1, 0),
        (0, 99),
        (1, 0),
        (0.0, 0),
        ("0", 0),
    ],
)
def test_invalid_pointer_ranges_are_rejected(span_start: object, span_end: object) -> None:
    index = build_source_pointer_index("source-4", "one\ntwo")
    with pytest.raises(SourcePointerError):
        index.resolve(
            span_start=span_start,  # type: ignore[arg-type]
            span_end=span_end,  # type: ignore[arg-type]
            source_text_sha256=index.source_text_sha256,
        )


def test_wrong_source_hash_and_source_id_are_rejected() -> None:
    index = build_source_pointer_index("source-5", "one\ntwo")
    wrong_hash = hashlib.sha256(b"different").hexdigest()
    with pytest.raises(SourcePointerError, match="hash"):
        index.resolve(span_start=0, span_end=0, source_text_sha256=wrong_hash)
    with pytest.raises(SourcePointerError, match="source ID"):
        index.resolve(span_start=0, span_end=0, source_text_sha256=index.source_text_sha256, source_id="other")


def test_index_tuple_is_immutable_and_source_hash_is_bound() -> None:
    index = build_source_pointer_index("source-6", "one\ntwo")
    with pytest.raises(TypeError):
        index.spans[0] = index.spans[1]  # type: ignore[index]
    with pytest.raises(SourcePointerError, match="supplied hash"):
        build_source_pointer_index("source-6", "one\ntwo", source_text_sha256="0" * 64)
