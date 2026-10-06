from paa.llm_client import expand, parse_codes


def test_line_parser_accepts_bare_and_bracketed_numbers():
    parsed = parse_codes("1 SL\n[2] PO\njotain selitystä\n3 XX\n4 PD")
    assert parsed == {1: "SL", 2: "PO", 4: "PD"}


def test_none_means_no_rows():
    assert parse_codes("NONE") == {}


def test_codes_expand_to_storage_labels():
    assert expand("PA") == "PERSONAL_ACTION_COMMITMENT"
    assert expand("PD") == "POLICY_DESIDERATUM"
