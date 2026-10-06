"""Official election-candidate results from tulospalvelu.vaalit.fi.

The downloaded file is one row per candidate per voting area. Vote totals use
the electoral-district rows when they exist, and polling-district rows otherwise.
"""


import hashlib
import zipfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from paa.config import RAW, ensure_dirs
from paa.http_client import get_bytes
from paa.identity import name_key
from paa.store import add_manifest, connect

SOURCES = {
    2023: "https://tulospalvelu.vaalit.fi/EKV-2023/ekv-2023_ehd_maa.csv.zip",
    2019: "https://tulospalvelu.vaalit.fi/EKV-2019/ekv-2019_ehd_maa.csv.zip",
    2015: "https://tulospalvelu.vaalit.fi/E-2015/e-2015_ehd_maa.csv.zip",
    2011: "https://tulospalvelu.vaalit.fi/EKV-2011/e-2011_ehd_maa.csv.zip",
}

# FI ehdokastiedosto column order from the ministry header workbook.
COL_DISTRICT = 1
COL_AREA = 3
COL_ABBR = 5
COL_PARTY = 11
COL_NUMBER = 14
COL_FIRST = 17
COL_LAST = 18
COL_AGE = 20
COL_JOB = 21
COL_HOME = 23
COL_VOTES = 34
COL_ELECTED = 38


def _int(value: str) -> int:
    text = (value or "").strip()
    if not text:
        return 0
    return int(text)


def aggregate_rows(lines: list[str], year: int) -> tuple[list[dict], dict]:
    """Aggregate area rows into one candidacy per district and candidate number."""
    buckets: dict[tuple, dict] = {}
    area_counts: dict[str, int] = defaultdict(int)
    for line in lines:
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(";")]
        if len(parts) <= COL_ELECTED:
            continue
        area_counts[parts[COL_AREA] or "?"] += 1
        number = _int(parts[COL_NUMBER])
        if number <= 0:
            continue
        key = (parts[COL_DISTRICT], number, parts[COL_FIRST], parts[COL_LAST])
        row = buckets.get(key)
        if row is None:
            row = {
                "election_year": year,
                "district_code": parts[COL_DISTRICT].zfill(2),
                "district_abbr": parts[COL_ABBR],
                "party": parts[COL_PARTY],
                "candidate_number": number,
                "first_name": parts[COL_FIRST],
                "last_name": parts[COL_LAST],
                "age": _int(parts[COL_AGE]) or None,
                "occupation": parts[COL_JOB],
                "home_municipality": parts[COL_HOME],
                "votes_by_area": defaultdict(int),
                "elected_flags": set(),
            }
            buckets[key] = row
        area = parts[COL_AREA] or "?"
        row["votes_by_area"][area] += _int(parts[COL_VOTES])
        if parts[COL_ELECTED]:
            row["elected_flags"].add(parts[COL_ELECTED])
    candidacies = []
    for row in buckets.values():
        areas = row["votes_by_area"]
        if "V" in areas:
            votes = areas["V"]
            vote_basis = "district_row"
        elif "A" in areas:
            votes = areas["A"]
            vote_basis = "sum_polling_districts"
        else:
            votes = sum(areas.values())
            vote_basis = "sum_other_areas"
        elected = "1" in row["elected_flags"]
        candidacies.append(
            {
                "candidacy_id": (
                    f"vaalit-{row['election_year']}-{row['district_code']}-{row['candidate_number']}"
                ),
                "election_year": row["election_year"],
                "district_code": row["district_code"],
                "district_abbr": row["district_abbr"],
                "party": row["party"],
                "candidate_number": row["candidate_number"],
                "first_name": row["first_name"],
                "last_name": row["last_name"],
                "name_key": name_key(row["first_name"], row["last_name"]),
                "age": row["age"],
                "occupation": row["occupation"],
                "home_municipality": row["home_municipality"],
                "votes": votes,
                "elected": int(elected),
                "valintatieto": ",".join(sorted(row["elected_flags"])),
                "vote_basis": vote_basis,
            }
        )
    stats = {
        "candidates": len(candidacies),
        "elected": sum(item["elected"] for item in candidacies),
        "area_rows": dict(area_counts),
        "vote_basis": {
            basis: sum(1 for item in candidacies if item["vote_basis"] == basis)
            for basis in ("district_row", "sum_polling_districts", "sum_other_areas")
        },
    }
    return candidacies, stats


def _zip_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        name = archive.namelist()[0]
        return archive.read(name).decode("latin-1")


def load_year(conn, year: int, path: Path) -> dict:
    text = _zip_text(path)
    candidacies, stats = aggregate_rows(text.splitlines(), year)
    conn.execute("DELETE FROM candidacies WHERE election_year = ?", (year,))
    conn.executemany(
        """INSERT INTO candidacies(
            candidacy_id, election_year, district_code, district_abbr, party,
            candidate_number, first_name, last_name, name_key, age, occupation,
            home_municipality, votes, elected, valintatieto, actor_id
        ) VALUES (
            :candidacy_id, :election_year, :district_code, :district_abbr, :party,
            :candidate_number, :first_name, :last_name, :name_key, :age, :occupation,
            :home_municipality, :votes, :elected, :valintatieto, NULL
        )""",
        candidacies,
    )
    stats["year"] = year
    stats["path"] = str(path)
    return stats


def download(year: int) -> Path:
    ensure_dirs()
    destination = RAW / "vaalit" / f"ehd_{year}.csv.zip"
    if destination.exists() and destination.stat().st_size > 1000:
        return destination
    status, body, _ctype = get_bytes(SOURCES[year])
    if status != 200 or not body:
        raise RuntimeError(f"vaalit {year} download failed: HTTP {status}")
    destination.write_bytes(body)
    return destination


def acquire(years: list[int] | None = None) -> list[dict]:
    ensure_dirs()
    conn = connect()
    reports = []
    now = datetime.now(UTC).isoformat()
    for year in years or sorted(SOURCES):
        path = download(year)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        stats = load_year(conn, year, path)
        add_manifest(
            conn,
            source_id=f"SRC-VAALIT-{year}",
            url=SOURCES[year],
            sha256=digest,
            bytes=path.stat().st_size,
            http_status=200,
            retrieved_at=now,
            note=f"candidates={stats['candidates']} elected={stats['elected']}",
        )
        reports.append(stats)
        print(f"vaalit {year}: {stats['candidates']} candidates, {stats['elected']} elected, basis {stats['vote_basis']}")
    conn.commit()
    conn.close()
    return reports
