"""Human-readable evidence must preserve people, alternatives and source scope."""
import json

from paa.evidence_site import render_evidence
from paa.inquiry_cases import anchor, digest, source_record

PERSON_COLUMNS = ['EdustajaId', 'AanestysId', 'EdustajaEtunimi', 'EdustajaSukunimi',
                  'EdustajaHenkiloNumero', 'EdustajaRyhmaLyhenne', 'EdustajaAanestys', 'Imported']
META_COLUMNS = ['AanestysId', 'AanestysOtsikko', 'AanestysAlkuaika', 'Url', 'AanestysTulosJaa', 'AanestysTulosEi']


def source(identity, text, title='Internal snapshot annotation'):
    return source_record(identity, text, title=title, url='https://example.org/source',
                         locator='raw/person.json; source_manifest.json', raw_sha256=digest(text))


def metadata(event='10'):
    text = json.dumps({'tableName': 'SaliDBAanestys', 'columnNames': META_COLUMNS,
                       'rowData': [[event, 'Mietintö JAA / Nimetty vaihtoehto EI', '2024-01-02 14:00:00',
                                    '/aanestystulos/1/2/2024', '90', '80']]})
    return source('SALIDB-META-' + event, text)


def test_fragmented_person_anchors_render_one_named_vote_without_transport_details():
    # Imported is acquisition time, not the decision date. Another member voted JAA.
    row = ['3', '10', 'Kaisa', 'Garedew', '1530', 'ps', 'Ei', '2026-10-11 10:00:00']
    other = ['4', '10', 'Kim', 'Berg', '1443', 'sd', 'Jaa', '2026-10-11 10:00:00']
    text = json.dumps({'tableName': 'SaliDBAanestysEdustaja', 'columnNames': PERSON_COLUMNS,
                       'rowData': [row, other]})
    s = source('SALIDBAANESTYSEDUSTAJA', text)
    name = anchor(s, 'Kaisa')
    vote = anchor(s, 'Ei', start=text.index('"Ei"') + 1)
    meta = metadata()
    refs = {r['evidence_id']: r for r in [name, vote]}
    before = json.dumps(refs, sort_keys=True)
    result = render_evidence(list(refs), refs, {s['source_id']: s, meta['source_id']: meta})
    assert result.count('Kaisa Garedew: EI') == 1
    assert 'Kim Berg' not in result
    assert '2024-01-02' in result and '2026-10-11' not in result
    assert 'Mietintö JAA / Nimetty vaihtoehto EI' in result
    assert 'https://www.eduskunta.fi/aanestystulos/1/2/2024' in result
    assert all(key in result for key in refs)  # all retained fragment links still resolve
    assert all(token not in result for token in ['columnNames', 'rowData', 'Imported', 'source_manifest', 'raw/person', digest(text)])
    assert json.dumps(refs, sort_keys=True) == before  # projection does not rewrite evidence


def test_jsonl_projection_keeps_distinct_votes_and_a_whole_array_record():
    rows = [['3', '10', 'Kaisa', 'Garedew', '1530', 'ps', 'Jaa', 'capture time'],
            ['4', '11', 'Kaisa', 'Garedew', '1530', 'ps', 'Ei', 'capture time']]
    text = '\n'.join(json.dumps({'file': 'raw/private.json', 'row_index': 25, 'row': row}) for row in rows)
    s = source('SALIDBAANESTYSEDUSTAJA', text)
    refs = [anchor(s, line) for line in text.splitlines()]
    result = render_evidence([r['evidence_id'] for r in refs], {r['evidence_id']: r for r in refs},
                             {s['source_id']: s, 'SALIDB-META-10': metadata(), 'SALIDB-META-11': metadata('11')})
    assert 'Kaisa Garedew: JAA' in result and 'Kaisa Garedew: EI' in result
    assert 'row_index' not in result and 'raw/private' not in result
    array = source('SALIDBAANESTYSEDUSTAJA-10', json.dumps(rows[0]))
    ref = anchor(array, array['text'])
    assert 'Kaisa Garedew: JAA' in render_evidence([ref['evidence_id']], {ref['evidence_id']: ref}, {array['source_id']: array})


def test_unknown_json_is_linked_without_raw_syntax_or_field_fragments():
    s = source('official-query', '{"query": "Kaisa", "internal": "source_manifest.json"}')
    r = anchor(s, 'Kaisa')
    result = render_evidence([r['evidence_id']], {r['evidence_id']: r}, {s['source_id']: s})
    assert 'https://example.org/source' in result
    assert 'query' not in result and 'source_manifest' not in result and '<blockquote>' not in result


def test_readable_quote_and_citation_survive_without_duplicate_acquisition_notes():
    s = source('speech', 'A < B. This is the recorded statement.', 'Eduskunta · puheenvuoro — full extracted text')
    r = anchor(s, 'A < B.')
    result = render_evidence([r['evidence_id'], r['evidence_id']], {r['evidence_id']: r}, {s['source_id']: s})
    assert result.count('A &lt; B.') == 1
    assert 'Eduskunta · puheenvuoro' in result
    assert 'full extracted' not in result and 'source_manifest' not in result


def test_metadata_totals_are_named_and_do_not_infer_the_result():
    s = metadata()
    r = anchor(s, '90')
    result = render_evidence([r['evidence_id']], {r['evidence_id']: r}, {s['source_id']: s})
    assert 'JAA 90 · EI 80' in result
    assert 'Mietintö JAA / Nimetty vaihtoehto EI' in result
    assert 'rowData' not in result and 'hyväksyttiin' not in result
