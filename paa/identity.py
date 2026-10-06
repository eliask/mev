"""Cross-election identity. Same name is not the same person."""


import re
import unicodedata
from dataclasses import dataclass


def fold(value: str) -> str:
    text = unicodedata.normalize("NFKC", value or "").casefold().strip()
    text = text.replace("−", "-").replace("–", "-")
    # Keep hyphens and letters such as š. Deleting them made Šek and Ek the same key.
    text = "".join(ch if (ch.isalnum() or ch in "- ") else " " for ch in text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _name_tokens(value: str) -> set[str]:
    text = "".join(
        ch for ch in unicodedata.normalize("NFKD", fold(value)) if not unicodedata.combining(ch)
    )
    text = text.replace('"', " ")
    return {token for token in text.replace("-", " ").split() if token}


def names_compatible(left_first: str, left_last: str, right_first: str, right_last: str) -> bool:
    """True when the names share a surname and a given name.

    Accent and hyphen differences match. A different given name does not.
    """
    if not (_name_tokens(left_last) & _name_tokens(right_last)):
        return False
    left_given, right_given = _name_tokens(left_first), _name_tokens(right_first)
    return bool(left_given and right_given and (left_given & right_given))


def name_key(first: str, last: str) -> str:
    folded_last, folded_first = fold(last), fold(first)
    if not folded_last or not folded_first:
        return ""
    return f"{folded_last}|{folded_first}"


@dataclass
class Candidacy:
    candidacy_id: str
    year: int
    district: str
    number: int
    first: str
    last: str
    age: int | None


@dataclass
class Member:
    person_id: str
    first: str
    last: str
    birth_year: int | None


def birth_window(year: int, age: int | None) -> tuple[int, int] | None:
    """Age is on election day, so the birth year is election_year-age or one year earlier."""
    if age is None:
        return None
    return (year - age - 1, year - age)


def _window_intersection(rows: list[Candidacy]) -> tuple[int, int] | None:
    """Birth years that fit every candidacy. Pairwise overlap is not enough."""
    low, high = -10**9, 10**9
    for row in rows:
        window = birth_window(row.year, row.age)
        if window is None:
            return None
        low, high = max(low, window[0]), min(high, window[1])
        if low > high:
            return None
    return (low, high)


def windows_overlap(left: tuple[int, int] | None, right: tuple[int, int] | None) -> bool:
    if left is None or right is None:
        return False
    return left[0] <= right[1] and right[0] <= left[1]


def cluster_candidacies(rows: list[Candidacy]) -> dict[str, str]:
    """Return candidacy_id -> cluster id.

    Two candidacies in the same election stay apart. Across elections, a shared
    name is one cluster only when the age windows overlap and each election
    contributes at most one row.
    """
    by_name: dict[str, list[Candidacy]] = {}
    for row in rows:
        key = name_key(row.first, row.last) or f"noname:{row.candidacy_id}"
        by_name.setdefault(key, []).append(row)
    assignment: dict[str, str] = {}
    for key, group in by_name.items():
        parent = {row.candidacy_id: row.candidacy_id for row in group}

        def find(item: str, roots: dict[str, str] = parent) -> str:
            while roots[item] != item:
                roots[item] = roots[roots[item]]
                item = roots[item]
            return item

        def union(left: str, right: str, roots: dict[str, str] = parent, find_root=find) -> None:
            left_root, right_root = find_root(left), find_root(right)
            if left_root != right_root:
                roots[right_root] = left_root

        for i, left in enumerate(group):
            for right in group[i + 1 :]:
                if left.year == right.year:
                    continue
                left_rows = [row for row in group if find(row.candidacy_id) == find(left.candidacy_id)]
                right_rows = [row for row in group if find(row.candidacy_id) == find(right.candidacy_id)]
                combined = left_rows + right_rows
                years = [row.year for row in combined]
                if len(years) != len(set(years)):
                    continue
                if _window_intersection(combined) is None:
                    continue
                # Another person of the same age stood in that election. Attach neither.
                rival = any(
                    row.candidacy_id != right.candidacy_id
                    and row.year == right.year
                    and _window_intersection(left_rows + [row]) is not None
                    for row in group
                )
                if rival:
                    continue
                union(left.candidacy_id, right.candidacy_id)
        for row in group:
            assignment[row.candidacy_id] = f"name:{key}:{find(row.candidacy_id)}"
    return assignment


def match_members(
    clusters: dict[str, list[Candidacy]],
    members: list[Member],
) -> dict[str, tuple[str, str]]:
    """Map cluster id -> (actor_id, status).

    Status is MP_UNIQUE, AMBIGUOUS, or CANDIDATE_ONLY. Ambiguous clusters do
    not receive an MP id.
    """
    by_name: dict[str, list[Member]] = {}
    for member in members:
        by_name.setdefault(name_key(member.first, member.last), []).append(member)
    result: dict[str, tuple[str, str]] = {}
    for cluster_id, rows in clusters.items():
        key = name_key(rows[0].first, rows[0].last)
        pool = by_name.get(key, [])
        compatible = []
        aged = [row for row in rows if row.age is not None]
        for member in pool:
            if member.birth_year is None or not aged:
                continue
            if all(
                windows_overlap(birth_window(row.year, row.age), (member.birth_year, member.birth_year))
                for row in aged
            ):
                compatible.append(member)
        undated = [member for member in pool if member.birth_year is None]
        if undated:
            # A missing birth year does not rule a second person out, and it is not a match.
            status = "AMBIGUOUS" if compatible or len(pool) > 1 else "CANDIDATE_ONLY"
            result[cluster_id] = (f"amb-{_safe(cluster_id)}" if status == "AMBIGUOUS" else f"cand-{_safe(cluster_id)}", status)
        elif len(compatible) == 1:
            result[cluster_id] = (f"mp-{compatible[0].person_id}", "MP_UNIQUE")
        elif len(pool) > 1:
            result[cluster_id] = (f"amb-{_safe(cluster_id)}", "AMBIGUOUS")
        else:
            result[cluster_id] = (f"cand-{_safe(cluster_id)}", "CANDIDATE_ONLY")
    claimed: dict[str, list[str]] = {}
    for cluster_id, (actor_id, status) in result.items():
        if status == "MP_UNIQUE":
            claimed.setdefault(actor_id, []).append(cluster_id)
    for actor_id, cluster_ids in claimed.items():
        if len(cluster_ids) > 1:
            for cluster_id in cluster_ids:
                result[cluster_id] = (f"amb-{_safe(cluster_id)}", "AMBIGUOUS")
    return result


def _safe(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")[:80]
