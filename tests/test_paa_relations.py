from paa.records import alternatives_from_title
from paa.relations import ObjectRetriever


def retrieve(target: str, text: str) -> list[dict]:
    return ObjectRetriever([{"object_id": "test", "title": text, "text": ""}]).search(target)


def test_diesel_word_does_not_alias_to_a_different_tax():
    assert retrieve("dieselvero", "Hallituksen esitys käyttövoimaverosta") == []


def test_a_named_proposal_is_not_a_diesel_choice():
    from paa.records import alternatives_from_title

    alts = alternatives_from_title("Lausumaehdotus, mietintö /Saara Hyrkkö", "e")
    choice = " ".join(item["alternative_text"] for item in alts)
    assert retrieve("dieselvero", choice) == []


def test_fuel_price_words_do_not_match_a_nuclear_matter():
    assert retrieve("polttoaineen hinta", "ydinpolttoaineen hankinta") == []


def test_literal_diesel_title_is_only_a_candidate_word():
    results = retrieve("dieselvero", "Dieselveron poistaminen / esityksen hylkääminen")
    assert results and all(r["status"] == "CANDIDATE" for r in results)


def test_vote_title_slash_without_spaces_is_still_two_alternatives():
    alts = alternatives_from_title("Hyväksyminen/hylkääminen", "e")
    assert [item["alternative_text"] for item in alts] == ["Hyväksyminen", "hylkääminen"]
    assert alternatives_from_title("", "e") == []


def test_retrieval_is_topic_agnostic_and_deduplicates_object_ids():
    results = retrieve("Eläinlääkäripalvelujen turvaaminen", "Eläinlääkäripalvelujen turvaaminen")
    assert len(results) == 1
    assert results[0]["status"] == "CANDIDATE"
