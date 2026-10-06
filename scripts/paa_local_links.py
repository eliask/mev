"""Audit generated local files, evidence fragments and dynamic person routes."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


class References(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []
        self.ids: set[str] = set()
        self.duplicate_ids: set[str] = set()

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if values.get("id"):
            if values["id"] in self.ids:
                self.duplicate_ids.add(values["id"])
            self.ids.add(values["id"])
        for key in ("href", "src"):
            if values.get(key):
                self.links.append(values[key])


def audit(root: Path) -> dict:
    root = root.resolve()
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    person_ids = {person["id"] for person in index["people"]}
    ids: dict[Path, set[str]] = {}
    pending = []
    counts = Counter()
    failures = []
    for page in sorted(root.rglob("*.html")):
        parser = References()
        parser.feed(page.read_text(encoding="utf-8"))
        ids[page] = parser.ids
        counts["html_pages"] += 1
        for duplicate in sorted(parser.duplicate_ids):
            counts["duplicate_dom_ids"] += 1
            failures.append({"page": str(page.relative_to(root)), "href": "#" + duplicate,
                             "reason": "duplicate DOM ID makes the source anchor ambiguous"})
        for href in parser.links:
            parts = urlsplit(href)
            if parts.scheme or parts.netloc:
                counts["external_links_not_fetched"] += 1
                continue
            target = (root / unquote(parts.path).lstrip("/")) if parts.path.startswith("/") else (page.parent / unquote(parts.path))
            target = target.resolve() if parts.path else page
            pending.append((page, href, target, unquote(parts.fragment)))
    for page, href, target, fragment in pending:
        counts["local_links_checked"] += 1
        reason = None
        if not target.is_relative_to(root):
            reason = "local link escapes the build root"
        elif not target.is_file():
            reason = "target file missing"
        elif fragment and target in ids:
            counts["fragments_checked"] += 1
            if target == root / "index.html" and fragment.startswith("person="):
                counts["person_routes_checked"] += 1
                if fragment.removeprefix("person=") not in person_ids:
                    reason = "dynamic person route missing from index"
            elif fragment not in ids[target]:
                reason = "target fragment missing"
        if reason:
            failures.append({"page": str(page.relative_to(root)), "href": href, "reason": reason})
    for person_id in sorted(person_ids):
        path = root / "people" / (person_id + ".json")
        counts["person_payloads_checked"] += 1
        try:
            packet = json.loads(path.read_text(encoding="utf-8"))
            if packet["id"] != person_id:
                raise ValueError("person payload identity mismatch")
        except (OSError, ValueError, KeyError) as error:
            failures.append({"page": "index.json", "href": str(path.relative_to(root)), "reason": str(error)})
    return {"root": str(root), "verified_at": datetime.now(UTC).isoformat(),
            "counts": dict(counts), "failure_count": len(failures), "failures": failures,
            "scope": "All generated HTML local file/fragment links and indexed person payloads; external URLs are counted, not fetched."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("dist/browser"))
    parser.add_argument("--report", type=Path, default=Path("reports/local_links.json"))
    args = parser.parse_args()
    result = audit(args.root)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": result["counts"], "failure_count": result["failure_count"]}))
    return int(bool(result["failure_count"]))


if __name__ == "__main__":
    raise SystemExit(main())
