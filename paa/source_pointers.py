"""Exact, source-version-bound pointers for model-facing text chunks.

The index is a transport aid, not an interpretation or admission mechanism.
It gives a caller a deterministic ``[0] ... [1] ...`` view over the original
text and resolves an inclusive chunk-id range back to one exact contiguous
source span.  Source text is never stripped, whitespace-normalized, or
reconstructed from separate chunks.
"""

import hashlib
import re
from dataclasses import dataclass
from typing import Final, Literal, final

SCHEMA_VERSION: Final = "paa.source_pointers.v1"
CHUNK_LINE: Final = "LINE"
CHUNK_SENTENCE: Final = "SENTENCE"
ChunkKind = Literal["LINE", "SENTENCE"]


class SourcePointerError(ValueError):
    """Raised when a source pointer cannot be bound safely."""


def text_sha256(text: str) -> str:
    """Return the hash of the exact retained text, without normalization."""

    if type(text) is not str:
        raise TypeError("source text must be an exact string")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _validate_sha256(value: str, *, field: str) -> str:
    if type(value) is not str or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise SourcePointerError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


@final
@dataclass(frozen=True, slots=True, kw_only=True, match_args=False)
class SourcePointerSpan:
    """One exact contiguous chunk of the retained source text.

    ``start`` and ``end`` are Python string offsets with an exclusive end,
    matching the ordinary PAA evidence-anchor convention.  ``pointer_id`` is
    the zero-based index shown to a model; ranges over pointer IDs are
    inclusive at both ends.
    """

    pointer_id: int
    source_id: str
    source_text_sha256: str
    start: int
    end: int
    text: str
    kind: ChunkKind

    def __post_init__(self) -> None:
        if type(self.pointer_id) is not int or isinstance(self.pointer_id, bool) or self.pointer_id < 0:
            raise SourcePointerError("pointer ID must be a non-negative integer")
        if not self.source_id:
            raise SourcePointerError("source ID is required")
        _validate_sha256(self.source_text_sha256, field="source text hash")
        if type(self.start) is not int or isinstance(self.start, bool):
            raise SourcePointerError("span start must be an integer")
        if type(self.end) is not int or isinstance(self.end, bool):
            raise SourcePointerError("span end must be an integer")
        if self.start < 0 or self.end <= self.start:
            raise SourcePointerError("span offsets must be a non-empty forward range")
        if type(self.text) is not str or not self.text or self.end - self.start != len(self.text):
            raise SourcePointerError("span text must exactly match its declared offsets")
        if self.kind not in (CHUNK_LINE, CHUNK_SENTENCE):
            raise SourcePointerError(f"unsupported source pointer chunk kind: {self.kind!r}")


@final
@dataclass(frozen=True, slots=True, kw_only=True, match_args=False)
class SourcePointerAnchor:
    """One ordinary contiguous evidence anchor resolved from pointer IDs."""

    source_id: str
    source_text_sha256: str
    span_start: int
    span_end: int
    start: int
    end: int
    quote: str

    def __post_init__(self) -> None:
        if not self.source_id or not self.quote:
            raise SourcePointerError("resolved anchor needs source identity and quote")
        _validate_sha256(self.source_text_sha256, field="source text hash")
        for name, value in (
            ("span_start", self.span_start),
            ("span_end", self.span_end),
            ("start", self.start),
            ("end", self.end),
        ):
            if type(value) is not int or isinstance(value, bool) or value < 0:
                raise SourcePointerError(f"{name} must be a non-negative integer")
        if self.span_end < self.span_start or self.end <= self.start or self.end - self.start != len(self.quote):
            raise SourcePointerError("resolved anchor range is invalid")

    def to_anchor_input(self) -> dict[str, str | int]:
        """Return the ordinary ``quote``/``start`` input for ``paa.inquiry_cases.anchor``."""

        return {"quote": self.quote, "start": self.start}


@final
@dataclass(frozen=True, slots=True, kw_only=True, match_args=False)
class SourcePointerIndex:
    """Immutable index covering one exact source-text version."""

    schema_version: str
    source_id: str
    source_text_sha256: str
    source_text: str
    spans: tuple[SourcePointerSpan, ...]

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SourcePointerError("unsupported source pointer schema version")
        if not self.source_id or type(self.source_text) is not str or not self.source_text:
            raise SourcePointerError("a non-empty source ID and source text are required")
        _validate_sha256(self.source_text_sha256, field="source text hash")
        if text_sha256(self.source_text) != self.source_text_sha256:
            raise SourcePointerError("source text does not match its declared hash")
        if type(self.spans) is not tuple or not self.spans:
            raise SourcePointerError("source pointer index needs at least one span")
        expected_start = 0
        for expected_id, span in enumerate(self.spans):
            if span.pointer_id != expected_id:
                raise SourcePointerError("source pointer IDs must be contiguous and zero-based")
            if span.source_id != self.source_id or span.source_text_sha256 != self.source_text_sha256:
                raise SourcePointerError("source pointer span is bound to a different source version")
            if span.start != expected_start or span.end > len(self.source_text):
                raise SourcePointerError("source pointer spans do not cover the source contiguously")
            if self.source_text[span.start:span.end] != span.text:
                raise SourcePointerError("source pointer span text changed")
            expected_start = span.end
        if expected_start != len(self.source_text):
            raise SourcePointerError("source pointer spans do not cover the complete source text")

    def indexed_text(self) -> str:
        """Render deterministic ``[pointer_id] exact chunk`` model input."""

        return "\n".join(f"[{span.pointer_id}] {span.text}" for span in self.spans)

    def resolve(
        self,
        *,
        span_start: int,
        span_end: int,
        source_text_sha256: str,
        source_id: str | None = None,
    ) -> SourcePointerAnchor:
        """Resolve an inclusive pointer-ID range to one exact anchor."""

        _validate_source_binding(self, source_text_sha256=source_text_sha256, source_id=source_id)
        _validate_pointer_range(self, span_start=span_start, span_end=span_end)
        first = self.spans[span_start]
        last = self.spans[span_end]
        return SourcePointerAnchor(
            source_id=self.source_id,
            source_text_sha256=self.source_text_sha256,
            span_start=span_start,
            span_end=span_end,
            start=first.start,
            end=last.end,
            quote=self.source_text[first.start:last.end],
        )

def _validate_source_binding(index: SourcePointerIndex, *, source_text_sha256: str, source_id: str | None) -> None:
    _validate_sha256(source_text_sha256, field="source text hash")
    if source_text_sha256 != index.source_text_sha256:
        raise SourcePointerError("source text hash does not match the pointer index")
    if source_id is not None and (type(source_id) is not str or source_id != index.source_id):
        raise SourcePointerError("source ID does not match the pointer index")


def _validate_pointer_range(index: SourcePointerIndex, *, span_start: int, span_end: int) -> None:
    for name, value in (("span_start", span_start), ("span_end", span_end)):
        if type(value) is not int or isinstance(value, bool):
            raise SourcePointerError(f"{name} must be an integer pointer ID")
    if span_start < 0 or span_end < 0 or span_start > span_end or span_end >= len(index.spans):
        raise SourcePointerError("pointer range is reversed or outside the source index")
    for left, right in zip(index.spans[span_start:span_end], index.spans[span_start + 1:span_end + 1]):
        if left.end != right.start:
            raise SourcePointerError("pointer range contains disjoint source spans")


def _line_ranges(text: str) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []
    start = 0
    for match in re.finditer(r".*?(?:\n|$)", text, flags=re.DOTALL):
        end = match.end()
        if end <= start:
            continue
        ranges.append((start, end))
        start = end
        if end == len(text):
            break
    return tuple(ranges)


def _sentence_ranges(text: str, line_start: int, line_end: int) -> tuple[tuple[int, int], ...]:
    """Split one line at conservative sentence punctuation without editing it."""

    content_end = line_end - 1 if line_end > line_start and text[line_end - 1] == "\n" else line_end
    if content_end <= line_start:
        return ((line_start, line_end),)
    boundaries = [match.end() for match in re.finditer(r"[.!?](?=\s|$)", text[line_start:content_end])]
    if not boundaries:
        return ((line_start, line_end),)
    ranges: list[tuple[int, int]] = []
    start = line_start
    for relative_end in boundaries:
        end = line_start + relative_end
        ranges.append((start, end))
        start = end
    if ranges and ranges[-1][1] == content_end and content_end < line_end:
        ranges[-1] = (ranges[-1][0], line_end)
    elif start < line_end:
        ranges.append((start, line_end))
    return tuple(ranges)


def build_source_pointer_index(
    source_id: str,
    source_text: str,
    *,
    source_text_sha256: str | None = None,
    granularity: Literal["line", "sentence"] = "line",
) -> SourcePointerIndex:
    """Build a deterministic exact-text pointer index.

    ``line`` preserves each source line as one chunk.  ``sentence`` performs a
    conservative punctuation split inside each line and keeps any residual
    whitespace/newline in the neighbouring exact chunk.  Both modes cover the
    entire original string byte-for-byte at the Python string level; neither
    strips headings or normalizes whitespace.
    """

    if type(source_id) is not str or not source_id:
        raise SourcePointerError("source ID is required")
    if type(source_text) is not str or not source_text:
        raise SourcePointerError("source text must be a non-empty string")
    if granularity not in ("line", "sentence"):
        raise SourcePointerError("granularity must be 'line' or 'sentence'")
    actual_hash = text_sha256(source_text)
    if source_text_sha256 is not None:
        _validate_sha256(source_text_sha256, field="source text hash")
        if source_text_sha256 != actual_hash:
            raise SourcePointerError("source text does not match the supplied hash")
    ranges: list[tuple[int, int]] = []
    for line_start, line_end in _line_ranges(source_text):
        if granularity == "line":
            ranges.append((line_start, line_end))
        else:
            ranges.extend(_sentence_ranges(source_text, line_start, line_end))
    spans = tuple(
        SourcePointerSpan(
            pointer_id=pointer_id,
            source_id=source_id,
            source_text_sha256=actual_hash,
            start=start,
            end=end,
            text=source_text[start:end],
            kind=CHUNK_LINE if granularity == "line" else (
                CHUNK_SENTENCE if len(_sentence_ranges(source_text, start, end)) == 1 and source_text[start:end].rstrip("\n").rstrip().endswith((".", "!", "?")) else CHUNK_LINE
            ),
        )
        for pointer_id, (start, end) in enumerate(ranges)
    )
    return SourcePointerIndex(
        schema_version=SCHEMA_VERSION,
        source_id=source_id,
        source_text_sha256=actual_hash,
        source_text=source_text,
        spans=spans,
    )


__all__ = [
    "CHUNK_LINE",
    "CHUNK_SENTENCE",
    "SCHEMA_VERSION",
    "SourcePointerAnchor",
    "SourcePointerError",
    "SourcePointerIndex",
    "SourcePointerSpan",
    "build_source_pointer_index",
    "text_sha256",
]
