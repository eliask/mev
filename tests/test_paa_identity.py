"""Identity joins must fail closed on same names."""

from paa.identity import Candidacy, Member, cluster_candidacies, match_members


def _cand(cid: str, year: int, age: int | None, district: str = "01", number: int = 2) -> Candidacy:
    return Candidacy(cid, year, district, number, "Matti", "Virtanen", age)


def test_same_year_same_name_is_not_one_person():
    rows = [_cand("a", 2023, 40, "01", 2), _cand("b", 2023, 40, "02", 3)]
    clusters = cluster_candidacies(rows)
    assert clusters["a"] != clusters["b"]


def test_age_gap_blocks_cross_election_merge():
    rows = [_cand("a", 2019, 40), _cand("b", 2023, 60)]
    clusters = cluster_candidacies(rows)
    assert clusters["a"] != clusters["b"]


def test_consistent_age_can_link_two_elections():
    rows = [_cand("a", 2019, 40, "01", 2), _cand("b", 2023, 44, "01", 5)]
    clusters = cluster_candidacies(rows)
    assert clusters["a"] == clusters["b"]


def test_two_members_with_the_same_name_are_not_merged():
    rows = [_cand("a", 2023, 40)]
    clusters = cluster_candidacies(rows)
    grouped = {clusters["a"]: rows}
    members = [
        Member("1", "Matti", "Virtanen", 1983),
        Member("2", "Matti", "Virtanen", 1982),
    ]
    matched = match_members(grouped, members)
    assert matched[clusters["a"]][1] == "AMBIGUOUS"
    assert not matched[clusters["a"]][0].startswith("mp-")


def test_birth_year_mismatch_does_not_attach_the_only_member():
    rows = [_cand("a", 2023, 40)]
    clusters = cluster_candidacies(rows)
    matched = match_members({clusters["a"]: rows}, [Member("9", "Matti", "Virtanen", 1950)])
    assert matched[clusters["a"]][1] == "CANDIDATE_ONLY"


def test_same_year_namesake_does_not_inherit_the_earlier_career():
    rows = [
        _cand("a", 2015, 24, "12", 97),
        _cand("b", 2019, 28, "12", 122),
        _cand("c", 2023, 32, "12", 51),
        _cand("d", 2023, 32, "12", 161),
    ]
    clusters = cluster_candidacies(rows)
    assert clusters["a"] == clusters["b"]
    assert clusters["c"] != clusters["a"]
    assert clusters["d"] != clusters["a"]
    assert clusters["c"] != clusters["d"]


def test_birth_windows_do_not_chain_past_a_gap():
    rows = [_cand("a", 2011, 30), _cand("b", 2015, 35), _cand("c", 2019, 40)]
    clusters = cluster_candidacies(rows)
    assert clusters["a"] != clusters["c"]


def test_one_member_is_not_given_two_same_year_candidates():
    rows = [_cand("a", 2019, 40, number=2), _cand("b", 2019, 41, number=3)]
    clusters = cluster_candidacies(rows)
    grouped = {clusters["a"]: [rows[0]], clusters["b"]: [rows[1]]}
    matched = match_members(grouped, [Member("1", "Matti", "Virtanen", 1978)])
    assert matched[clusters["a"]][1] == "AMBIGUOUS"
    assert matched[clusters["b"]][1] == "AMBIGUOUS"


def test_missing_birth_year_does_not_create_an_mp_id():
    rows = [_cand("a", 2019, 40)]
    clusters = cluster_candidacies(rows)
    matched = match_members({clusters["a"]: rows}, [Member("9", "Matti", "Virtanen", None)])
    assert not matched[clusters["a"]][0].startswith("mp-")


def test_accent_and_extra_given_name_can_confirm_a_candidate_number():
    from paa.identity import names_compatible

    assert names_compatible("Tom", "Packalen", "Tom", "Packalén")
    assert names_compatible("Sami-pekka", "Säynevirta", "Sami", "Säynevirta")
    assert not names_compatible("Marko", "Karhunen", "Eerikki", "Karhunen")
    assert not names_compatible("Jouni", "Jouni", "Jouni", "Kuitunen")


def test_caron_is_not_deleted_from_the_name():
    from paa.identity import fold, name_key

    assert "š" in fold("Šek")
    assert name_key("A", "Šek") != name_key("A", "Ek")


def test_district_numbers_change_meaning_after_2011():
    from paa.districts import same_numbered_district

    assert same_numbered_district(2011, "09", 2015, "09") is False
    assert same_numbered_district(2019, "09", 2023, "09") is True
    assert same_numbered_district(2011, "01", 2023, "01") is True


def test_missing_ages_do_not_count_as_compatible_with_every_birth_year():
    rows = [_cand("a", 2023, None)]
    clusters = cluster_candidacies(rows)
    matched = match_members({clusters["a"]: rows}, [Member("9", "Matti", "Virtanen", 1950)])
    assert matched[clusters["a"]][1] != "MP_UNIQUE"
