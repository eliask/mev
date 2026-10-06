"""Aggregate committee, expert-org, minister, and expert-person data into state_causal_map.db.

Adds tables:
  - committee_scrutiny: per-committee aggregated expert hearing stats
  - expert_org_scrutiny: per-organization aggregated expert hearing stats
  - ministry_he_profile: per-ministry HE-side aggregates (claims, scrutiny, pagerank)
  - he_signatories: minister/president signatures on each HE (from AKN XML)
  - signatory_profile: per-signatory aggregated stats (HEs, claims, scrutiny)
  - expert_person: per-named-expert aggregated stats

Additive: can be run on existing DB without full rebuild.

Reads ``expert_statement`` (singular) from each per-HE DB. That name is load
bearing: it was ``expert_statements`` until commit 77137be4 (2026-03-24), and
the window where the per-HE DBs had not yet been rewritten under the new name
is how six of these tables ended up empty for five months without anything
failing. The post-run contract check at the bottom of main() closes that hole.

CLI: ``mev build aggregate-entities``
"""

import sqlite3
import os
import re
import sys
import zipfile
from collections import defaultdict
from xml.etree import ElementTree as ET

from mev.config import CAUSAL_MAP_DB, HE_DB_DIR as _HE_DB_DIR, AKN_ZIP_PATH
from mev.db_contract import assert_causal_map


# Every per-HE database this pipeline could not read, with the reason. Reported
# by main(); a high rate means a schema break, not missing data.
_SKIPPED: list[tuple[str, str]] = []
_XML_SKIPPED: list[tuple[str, str]] = []


def report_skipped(total: int) -> None:
    """Print the skip rate and shout if it looks like a schema break."""
    n = len(_SKIPPED)
    if not total:
        return
    pct = 100.0 * n / total
    print(f"  per-HE databases skipped: {n}/{total} ({pct:.1f}%)")
    if not n:
        return
    reasons: dict[str, int] = {}
    for _, why in _SKIPPED:
        reasons[why.split(":")[0]] = reasons.get(why.split(":")[0], 0) + 1
    for why, c in sorted(reasons.items(), key=lambda kv: -kv[1])[:5]:
        print(f"      {c:>6}  {why}")
    if any("OperationalError" in w for _, w in _SKIPPED) and pct > 50:
        print("  *** MORE THAN HALF THE CORPUS FAILED TO READ ***")
        print("  *** This is the signature of a schema break, not of missing data. ***")
        print(f"  *** Example: {_SKIPPED[0][0]} -> {_SKIPPED[0][1]} ***")


def extract_org(name):
    """Extract organization from expert name string."""
    if not name:
        return None

    # Direct org names (no person prefix)
    direct_orgs = {
        "Terveyden ja hyvinvoinnin laitos (THL)",
        "Kansaneläkelaitos",
        "Suomen Kuntaliitto",
        "Helsingin kaupunki",
        "Tampereen kaupunki",
        "Espoon kaupunki",
        "Vantaan kaupunki",
        "Turun kaupunki",
    }
    for org in direct_orgs:
        if org in name:
            return org

    # "ry" suffix = registered association, often the whole name IS the org
    if " ry" in name and "," not in name:
        return name.strip()

    # Person with org after comma
    if "," in name:
        org = name.split(",", 1)[1].strip()
        # Clean truncated entries
        if len(org) < 5:
            return None
        # Remove leading HE reference artifacts
        if org.startswith("HE ") or org.startswith("vp "):
            return None
        return org

    # Academic titles without org
    academic_prefixes = (
        "professori",
        "dosentti",
        "tutkimusjohtaja",
        "yliopistonlehtori",
        "apulaisprofessori",
    )
    if any(name.lower().startswith(p) for p in academic_prefixes):
        return "Yliopisto/akateeminen"

    return None


def aggregate_expert_data(he_db_dir):
    _SKIPPED.clear()
    _scanned = 0
    """Scan all per-HE DBs and aggregate expert/committee data."""
    committee_stats = defaultdict(
        lambda: {
            "total": 0,
            "STRONG": 0,
            "ATTENTION": 0,
            "GRAVEYARD": 0,
            "SILENT": 0,
            "UNKNOWN": 0,
            "hes": set(),
        }
    )
    org_stats = defaultdict(
        lambda: {
            "total": 0,
            "STRONG": 0,
            "ATTENTION": 0,
            "GRAVEYARD": 0,
            "SILENT": 0,
            "UNKNOWN": 0,
            "hes": set(),
            "committees": set(),
        }
    )
    committee_he_links = set()  # (committee, he_id)

    for fn in sorted(os.listdir(he_db_dir)):
        if not fn.endswith(".db"):
            continue
        he_id = fn.replace(".db", "")
        db_path = os.path.join(he_db_dir, fn)
        _scanned += 1
        try:
            conn = sqlite3.connect(db_path)
            rows = conn.execute(
                "SELECT expert_name, committee, scrutiny_type FROM expert_statement"
            ).fetchall()
            conn.close()
        except Exception as e:
            # A silent `continue` here is what hid commit 77137be4's rename of
            # expert_statements -> expert_statement for five months: this loop
            # swallowed "no such table" 8,439 times and wrote six empty tables,
            # while `mev status` reported green because it checked file size.
            # Individual failures are legitimate (not every HE has expert data),
            # so tolerate them -- but COUNT them, and let main() judge the rate.
            _SKIPPED.append((he_id, f"{type(e).__name__}: {e}"))
            continue

        for name, committee, stype in rows:
            if not committee:
                continue
            stype = (
                stype
                if stype in ("STRONG", "ATTENTION", "GRAVEYARD", "SILENT")
                else "UNKNOWN"
            )

            committee_stats[committee]["total"] += 1
            committee_stats[committee][stype] += 1
            committee_stats[committee]["hes"].add(fn)
            committee_he_links.add((committee, he_id))

            org = extract_org(name)
            if org:
                org_stats[org]["total"] += 1
                org_stats[org][stype] += 1
                org_stats[org]["hes"].add(fn)
                org_stats[org]["committees"].add(committee)

    report_skipped(_scanned)
    return committee_stats, org_stats, sorted(committee_he_links)


def extract_ministers(zip_path):
    """Extract minister/president signatories from AKN XML zip."""
    _XML_SKIPPED.clear()
    _xml_seen = 0
    zf = zipfile.ZipFile(zip_path)
    # he_id -> list of (person, role)
    he_signatories = {}

    for name in zf.namelist():
        if not name.endswith("/main.xml") or "fin@" not in name:
            continue
        # Parse HE id from path: akn/fi/doc/government-proposal/YEAR/NUM/fin@/main.xml
        parts = name.split("/")
        if len(parts) < 6:
            continue
        year, num = parts[4], parts[5]
        he_id = f"he-{num}-{year}"

        _xml_seen += 1
        try:
            xml = zf.read(name)
            root = ET.fromstring(xml)
        except Exception as e:
            # Same shape as the per-HE swallow above: tolerate the instance, count
            # the rate. A malformed document here is normal; a corpus-wide parse
            # failure is a broken zip or a namespace change, and silence cannot
            # tell the two apart.
            _XML_SKIPPED.append((he_id, f"{type(e).__name__}: {e}"))
            continue

        persons = []
        roles = []
        for elem in root.iter():
            tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
            if tag == "person":
                text = "".join(elem.itertext()).strip()
                if text:
                    persons.append(text)
            elif tag == "role":
                text = "".join(elem.itertext()).strip()
                if text:
                    roles.append(text)

        signatories = []
        for i, person in enumerate(persons):
            role = roles[i] if i < len(roles) else ""
            signatories.append((person, role))

        if signatories:
            he_signatories[he_id] = signatories

    zf.close()
    if _xml_seen:
        n = len(_XML_SKIPPED)
        pct = 100.0 * n / _xml_seen
        print(f"  AKN documents unparseable: {n}/{_xml_seen} ({pct:.1f}%)")
        if pct > 50:
            print("  *** MORE THAN HALF THE AKN CORPUS FAILED TO PARSE ***")
            print(f"  *** Example: {_XML_SKIPPED[0][0]} -> {_XML_SKIPPED[0][1]} ***")
    return he_signatories


def aggregate_expert_persons(he_db_dir):
    """Aggregate per-named-expert stats from per-HE DBs."""
    people = defaultdict(
        lambda: {
            "display": "",
            "appearances": 0,
            "hes": set(),
            "committees": set(),
            "orgs": set(),
            "STRONG": 0,
            "ATTENTION": 0,
            "GRAVEYARD": 0,
            "SILENT": 0,
            "UNKNOWN": 0,
        }
    )

    for fn in sorted(os.listdir(he_db_dir)):
        if not fn.endswith(".db"):
            continue
        try:
            conn = sqlite3.connect(os.path.join(he_db_dir, fn))
            rows = conn.execute(
                "SELECT expert_name, committee, scrutiny_type FROM expert_statement"
            ).fetchall()
            conn.close()
        except Exception:
            continue

        for raw_name, committee, stype in rows:
            if not raw_name:
                continue
            name = raw_name.strip()
            # Skip pure org names
            if name in (
                "Terveyden ja hyvinvoinnin laitos (THL)",
                "Kansaneläkelaitos",
            ):
                continue

            # Clean prefixed names (HE 13/2024 vp TyV ...)
            m = re.match(r"HE \d+/\d+ vp \w+ [\d.]+ (.+)", name)
            if m:
                name = m.group(1)

            org = ""
            person = name
            if "," in name:
                parts = name.split(",", 1)
                person = parts[0].strip()
                org = parts[1].strip()
                if len(org) < 5:
                    org = ""

            # Skip if looks like an org
            org_markers = (" ry", "liitto", "keskus", "yhtymä", "kaupunki")
            if any(x in person.lower() for x in org_markers) and "," not in raw_name:
                continue

            key = person.lower().strip()
            stype = (
                stype
                if stype in ("STRONG", "ATTENTION", "GRAVEYARD", "SILENT")
                else "UNKNOWN"
            )

            people[key]["display"] = person
            people[key]["appearances"] += 1
            people[key]["hes"].add(fn.replace(".db", ""))
            if committee:
                people[key]["committees"].add(committee)
            if org:
                people[key]["orgs"].add(org)
            people[key][stype] += 1

    return people


def main():
    db_path = str(CAUSAL_MAP_DB)
    he_db_dir = str(_HE_DB_DIR)
    zip_path = str(AKN_ZIP_PATH)

    if not os.path.exists(db_path):
        print(f"ERROR: {db_path} not found")
        sys.exit(1)
    if not os.path.isdir(he_db_dir):
        print(f"ERROR: {he_db_dir} not found")
        sys.exit(1)

    print("Scanning per-HE DBs for expert data...")
    committee_stats, org_stats, committee_he_links = aggregate_expert_data(he_db_dir)
    print(
        f"  {len(committee_stats)} committees, {len(org_stats)} organizations, {len(committee_he_links)} committee-HE links found"
    )

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Committee scrutiny table
    c.execute("DROP TABLE IF EXISTS committee_scrutiny")
    c.execute("""
        CREATE TABLE committee_scrutiny (
            committee TEXT PRIMARY KEY,
            total_experts INTEGER,
            strong INTEGER,
            attention INTEGER,
            graveyard INTEGER,
            silent INTEGER,
            unknown INTEGER,
            he_count INTEGER,
            ignored_rate REAL
        )
    """)
    for committee, s in sorted(
        committee_stats.items(), key=lambda x: x[1]["total"], reverse=True
    ):
        ignored = s["GRAVEYARD"] + s["SILENT"]
        rate = ignored / s["total"] if s["total"] else 0
        c.execute(
            "INSERT INTO committee_scrutiny VALUES (?,?,?,?,?,?,?,?,?)",
            (
                committee,
                s["total"],
                s["STRONG"],
                s["ATTENTION"],
                s["GRAVEYARD"],
                s["SILENT"],
                s["UNKNOWN"],
                len(s["hes"]),
                round(rate, 4),
            ),
        )
    print(f"  Wrote {len(committee_stats)} committee_scrutiny rows")

    # Expert org scrutiny table
    c.execute("DROP TABLE IF EXISTS expert_org_scrutiny")
    c.execute("""
        CREATE TABLE expert_org_scrutiny (
            organization TEXT PRIMARY KEY,
            total_experts INTEGER,
            strong INTEGER,
            attention INTEGER,
            graveyard INTEGER,
            silent INTEGER,
            unknown INTEGER,
            he_count INTEGER,
            committee_count INTEGER,
            ignored_rate REAL
        )
    """)
    for org, s in sorted(
        org_stats.items(), key=lambda x: x[1]["total"], reverse=True
    ):
        ignored = s["GRAVEYARD"] + s["SILENT"]
        rate = ignored / s["total"] if s["total"] else 0
        c.execute(
            "INSERT INTO expert_org_scrutiny VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                org,
                s["total"],
                s["STRONG"],
                s["ATTENTION"],
                s["GRAVEYARD"],
                s["SILENT"],
                s["UNKNOWN"],
                len(s["hes"]),
                len(s["committees"]),
                round(rate, 4),
            ),
        )
    print(f"  Wrote {len(org_stats)} expert_org_scrutiny rows")

    # Committee → HE links (for cross-linking in UI)
    c.execute("DROP TABLE IF EXISTS committee_he_link")
    c.execute("""
        CREATE TABLE committee_he_link (
            committee TEXT NOT NULL,
            he_id TEXT NOT NULL,
            PRIMARY KEY (committee, he_id)
        )
    """)
    c.executemany(
        "INSERT OR IGNORE INTO committee_he_link VALUES (?,?)",
        committee_he_links,
    )
    print(f"  Wrote {len(committee_he_links)} committee_he_link rows")

    # Ministry HE profile (HE-side aggregates per ministry)
    c.execute("DROP TABLE IF EXISTS ministry_he_profile")
    c.execute("""
        CREATE TABLE ministry_he_profile (
            ministry TEXT PRIMARY KEY,
            ministry_fi TEXT,
            he_count INTEGER,
            claims_total INTEGER,
            claims_fiscal INTEGER,
            claims_fiscal_eur REAL,
            scrutiny_he_count INTEGER,
            avg_ignored_rate REAL,
            avg_he_pagerank REAL,
            total_he_pagerank REAL
        )
    """)

    ministry_slug_to_fi = {
        "fi.ministry-of-finance": "Valtiovarainministeriö",
        "fi.ministry-of-social-affairs-and-health": "Sosiaali- ja terveysministeriö",
        "fi.ministry-of-economic-affairs-and-employment": "Työ- ja elinkeinoministeriö",
        "fi.ministry-of-justice": "Oikeusministeriö",
        "fi.ministry-of-transport-and-communications": "Liikenne- ja viestintäministeriö",
        "fi.ministry-of-education-and-culture": "Opetus- ja kulttuuriministeriö",
        "fi.ministry-of-agriculture-and-forestry": "Maa- ja metsätalousministeriö",
        "fi.ministry-for-foreign-affairs": "Ulkoministeriö",
        "fi.ministry-of-the-interior": "Sisäministeriö",
        "fi.ministry-of-the-environment": "Ympäristöministeriö",
        "fi.ministry-of-defence": "Puolustusministeriö",
        "fi.prime-ministers-office": "Valtioneuvoston kanslia",
        "fi.parliament": "Eduskunta",
    }

    rows = c.execute("""
        SELECT h.ministry,
            COUNT(DISTINCT h.he_id) as he_count,
            COUNT(DISTINCT cl.claim_id) as claims_total,
            SUM(CASE WHEN cl.claim_type='FISCAL' THEN 1 ELSE 0 END) as claims_fiscal,
            COALESCE(SUM(CASE WHEN cl.claim_type='FISCAL' AND cl.amount_eur IS NOT NULL
                THEN ABS(cl.amount_eur) ELSE 0 END), 0) as claims_fiscal_eur,
            COUNT(DISTINCT sc.he_id) as scrutiny_he_count,
            AVG(sc.ignored_rate) as avg_ignored_rate,
            AVG(h.he_pagerank) as avg_he_pagerank,
            SUM(h.he_pagerank) as total_he_pagerank
        FROM he_nodes h
        LEFT JOIN he_claims cl ON h.he_id = cl.he_id
        LEFT JOIN he_scrutiny sc ON h.he_id = sc.he_id
        WHERE h.ministry IS NOT NULL AND h.ministry <> ''
        GROUP BY h.ministry
    """).fetchall()

    for row in rows:
        slug = row[0]
        fi_name = ministry_slug_to_fi.get(slug, slug)
        c.execute(
            "INSERT INTO ministry_he_profile VALUES (?,?,?,?,?,?,?,?,?,?)",
            (slug, fi_name, row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8]),
        )
    print(f"  Wrote {len(rows)} ministry_he_profile rows")

    # ── Minister/President signatories ──
    akn_tables = ("he_signatories", "signatory_profile")
    akn_skipped = not os.path.exists(zip_path)
    if os.path.exists(zip_path):
        print("Extracting minister signatories from AKN XML...")
        he_signatories = extract_ministers(zip_path)
        print(f"  {len(he_signatories)} HEs with signatories")

        c.execute("DROP TABLE IF EXISTS he_signatories")
        c.execute("""
            CREATE TABLE he_signatories (
                he_id TEXT,
                person TEXT,
                role TEXT,
                PRIMARY KEY (he_id, person, role)
            )
        """)
        sig_count = 0
        for he_id, sigs in he_signatories.items():
            for person, role in sigs:
                c.execute(
                    "INSERT OR IGNORE INTO he_signatories VALUES (?,?,?)",
                    (he_id, person, role),
                )
                sig_count += 1
        print(f"  Wrote {sig_count} he_signatories rows")

        # Signatory profile — aggregate per person with claims/scrutiny from DB
        c.execute("DROP TABLE IF EXISTS signatory_profile")
        c.execute("""
            CREATE TABLE signatory_profile (
                person TEXT PRIMARY KEY,
                he_count INTEGER,
                roles TEXT,
                year_first INTEGER,
                year_last INTEGER,
                claims_total INTEGER,
                claims_fiscal INTEGER,
                scrutiny_he_count INTEGER,
                avg_ignored_rate REAL
            )
        """)
        # Aggregate per person
        person_data = defaultdict(
            lambda: {
                "hes": set(),
                "roles": set(),
                "years": set(),
            }
        )
        for he_id, sigs in he_signatories.items():
            for person, role in sigs:
                key = person.strip()
                person_data[key]["hes"].add(he_id)
                if role:
                    person_data[key]["roles"].add(role)
                # Extract year from he_id
                parts = he_id.split("-")
                if len(parts) >= 3:
                    try:
                        person_data[key]["years"].add(int(parts[-1]))
                    except ValueError:
                        pass

        # Now enrich with claims and scrutiny from DB
        for person, pd in sorted(
            person_data.items(), key=lambda x: len(x[1]["hes"]), reverse=True
        ):
            he_ids = pd["hes"]
            placeholders = ",".join(["?" for _ in he_ids])
            he_list = list(he_ids)

            # Claims
            claims_row = c.execute(
                f"SELECT COUNT(*), SUM(CASE WHEN claim_type='FISCAL' THEN 1 ELSE 0 END) "
                f"FROM he_claims WHERE he_id IN ({placeholders})",
                he_list,
            ).fetchone()
            claims_total = claims_row[0] or 0
            claims_fiscal = claims_row[1] or 0

            # Scrutiny
            scrutiny_rows = c.execute(
                f"SELECT COUNT(*), AVG(ignored_rate) FROM he_scrutiny WHERE he_id IN ({placeholders})",
                he_list,
            ).fetchone()
            scrutiny_count = scrutiny_rows[0] or 0
            avg_ignored = scrutiny_rows[1]

            years = sorted(pd["years"]) if pd["years"] else [0]
            roles_str = ", ".join(sorted(pd["roles"]))[:200]

            c.execute(
                "INSERT INTO signatory_profile VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    person,
                    len(he_ids),
                    roles_str,
                    years[0],
                    years[-1],
                    claims_total,
                    claims_fiscal,
                    scrutiny_count,
                    round(avg_ignored, 4) if avg_ignored is not None else None,
                ),
            )
        print(f"  Wrote {len(person_data)} signatory_profile rows")
    else:
        print(f"  SKIP minister extraction: {zip_path} not found")

    # ── Expert persons ──
    print("Aggregating expert persons...")
    people = aggregate_expert_persons(he_db_dir)
    print(f"  {len(people)} unique expert persons found")

    c.execute("DROP TABLE IF EXISTS expert_person")
    c.execute("""
        CREATE TABLE expert_person (
            person TEXT PRIMARY KEY,
            appearances INTEGER,
            he_count INTEGER,
            committee_count INTEGER,
            organizations TEXT,
            committees TEXT,
            strong INTEGER,
            attention INTEGER,
            graveyard INTEGER,
            silent INTEGER,
            ignored_rate REAL
        )
    """)
    ep_count = 0
    for key, p in sorted(people.items(), key=lambda x: x[1]["appearances"], reverse=True):
        if p["appearances"] < 2:
            continue  # Skip one-off appearances
        ignored = p["GRAVEYARD"] + p["SILENT"]
        rate = ignored / p["appearances"] if p["appearances"] else 0
        c.execute(
            "INSERT INTO expert_person VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                p["display"],
                p["appearances"],
                len(p["hes"]),
                len(p["committees"]),
                ", ".join(sorted(p["orgs"]))[:300],
                ", ".join(sorted(p["committees"]))[:300],
                p["STRONG"],
                p["ATTENTION"],
                p["GRAVEYARD"],
                p["SILENT"],
                round(rate, 4),
            ),
        )
        ep_count += 1
    print(f"  Wrote {ep_count} expert_person rows (≥2 appearances)")

    # Expert person → HE links
    c.execute("DROP TABLE IF EXISTS expert_person_he_link")
    c.execute("""
        CREATE TABLE expert_person_he_link (
            person TEXT NOT NULL,
            he_id TEXT NOT NULL,
            PRIMARY KEY (person, he_id)
        )
    """)
    ep_he_links = []
    for key, p in people.items():
        if p["appearances"] < 2:
            continue
        for he_id in p["hes"]:
            ep_he_links.append((p["display"], he_id))
    c.executemany("INSERT OR IGNORE INTO expert_person_he_link VALUES (?,?)", ep_he_links)
    print(f"  Wrote {len(ep_he_links)} expert_person_he_link rows")

    # Expert org → HE links
    c.execute("DROP TABLE IF EXISTS expert_org_he_link")
    c.execute("""
        CREATE TABLE expert_org_he_link (
            organization TEXT NOT NULL,
            he_id TEXT NOT NULL,
            PRIMARY KEY (organization, he_id)
        )
    """)
    org_he_links = []
    for org, s in org_stats.items():
        for fn in s["hes"]:
            org_he_links.append((org, fn.replace(".db", "")))
    c.executemany("INSERT OR IGNORE INTO expert_org_he_link VALUES (?,?)", org_he_links)
    print(f"  Wrote {len(org_he_links)} expert_org_he_link rows")

    conn.commit()
    conn.close()

    # Post-run contract check: a silent zero must fail THIS run, not the next
    # reader. This is the guard whose absence let six tables sit empty from
    # 2026-03-24 to 2026-08-17 while `mev status` reported green.
    assert_causal_map(db_path, skip=akn_tables if akn_skipped else ())
    print("Done. (state_causal_map.db contract verified)")


if __name__ == "__main__":
    main()
