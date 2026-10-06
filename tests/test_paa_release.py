"""Selected source reviews reach the exact publication archive, or fail closed."""

import hashlib
import json
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

from paa.config import FIXTURE_DIR
from paa.inquiry_cases import validate_case
from paa.release import ReleaseError, build_release, load_selection

SELECTION = FIXTURE_DIR / 'public_beta_selection.json'


def changed_selection(tmp_path, change):
    raw = json.loads(SELECTION.read_text())
    change(raw)
    path = tmp_path / 'selection.json'
    path.write_text(json.dumps(raw))
    return path


def test_selected_sources_to_exact_site_archive_and_evidence(tmp_path):
    result = build_release(SELECTION, tmp_path/'site', source_revision='test-revision',
                           base_url='https://example.org/paa/')
    root = Path(result['destination'])
    manifest = json.loads((root/'manifest.json').read_text())
    assert len(manifest['cases']) == 3
    assert manifest['deployment_status'] == 'NOT_DEPLOYED'
    assert manifest['code_license'] == 'MIT'
    assert {'release.py', 'case_site.py', 'inquiry_cases.py'} <= manifest['producer_versions'].keys()
    assert (root/'LICENSE.txt').read_text() == (Path(__file__).parents[1]/'LICENSE').read_text()
    for name, expected in manifest['files'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected
    with zipfile.ZipFile(result['archive']) as archive:
        assert set(archive.namelist()) == {*manifest['files'], 'manifest.json'}
        for name in archive.namelist():
            assert archive.read(name) == (root/name).read_bytes()
    assert not any('people' in name or '.tmp' in name for name in manifest['files'])
    assert {name for name in manifest['files'] if name.endswith('-research.html')} == {
        entry['research_path'] for entry in manifest['cases']
    }
    for entry in manifest['cases']:
        path = root/entry['path']
        page = path.read_text()
        packet = json.loads(path.with_suffix('.json').read_text())
        assert validate_case(packet)
        assert 'legal_comparison_artifacts' not in packet
        assert packet['publication']['revision'] == manifest['revision']
        assert 'ei riippumatonta ihmistarkistusta' in page
        assert entry['review_date'] in page and entry['source_period'] in page
        assert f'https://example.org/paa/{entry["path"]}' in page
        assert '<meta property="og:description"' in page
        assert 'Lataa lähdepaketti' in page and 'Ilmoita korjaus' in page
        assert 'README' not in page
        assert all(c['state'] == 'SOURCE_REVIEWED' for c in packet['claims'])
        assert path.with_suffix('.txt').is_file()
        research = (root/entry['research_path']).read_text()
        assert all(ref['evidence_id'] in research for ref in packet['evidence'])
        assert 'Takaisin vastaukseen' in research
        assert 'class="inline-evidence"' in page
    person = json.loads((root/'cases/person-bergbom-councils-followup-2023.json').read_text())
    assert person['canonical_target_scope'] == 'PARTIAL_COMPONENT'
    assert person['question']['target_scope'] == 'PARTIAL_COMPONENT'
    assert 'Sastamala' in person['publication']['main_limit']
    assert any('Alueviesti' in claim['text'] for claim in person['claims'])
    assert 'Tutkimus kesken' in (root/'index.html').read_text()
    secondary = next(source for source in person['sources'] if source['source_id'] == 'SRC-ALUEVIESTI-20230427')
    assert secondary['text_origin'] == 'WEB_TEXT_RETRIEVAL_SELECTED_EXCERPTS'
    assert 'capture_raw_sha256' not in secondary
    assert secondary['retrieval_text_sha256'] != secondary['raw_sha256']


@pytest.mark.parametrize('change,message', [
    (lambda r: r['cases'][0].update(packet_sha256='0'*64), 'changed since editorial'),
    (lambda r: r['cases'][0]['permissions'].pop(), 'every included source'),
    (lambda r: r['cases'].append(r['cases'][0]), 'distinct cases'),
    (lambda r: r['cases'][0].update(sources_file='../private.jsonl'), 'bundled fixtures'),
])
def test_changed_or_unscoped_publication_leaves_no_artifact(tmp_path, change, message):
    selection = changed_selection(tmp_path, change)
    with pytest.raises(ReleaseError, match=message):
        build_release(selection, tmp_path/'site', source_revision='test')
    assert not (tmp_path/'site').exists()
    assert not (tmp_path/'site.zip').exists()


def test_release_refuses_to_copy_over_a_workbench(tmp_path):
    root = tmp_path/'workbench'
    root.mkdir()
    (root/'private.json').write_text('preserve this')
    with pytest.raises(ReleaseError, match='must be new'):
        build_release(SELECTION, root, source_revision='test')
    assert (root/'private.json').read_text() == 'preserve this'


def test_selection_does_not_retain_mutable_record_aliases():
    selection = load_selection(SELECTION)
    with pytest.raises(ReleaseError, match='owned tuple'):
        replace(selection, cases=list(selection.cases))
    with pytest.raises(ReleaseError, match='owned tuple'):
        replace(selection.cases[0], permissions=list(selection.cases[0].permissions))


def test_preview_does_not_invent_canonical_hostname(tmp_path):
    result = build_release(SELECTION, tmp_path/'local', source_revision='test')
    manifest = json.loads((tmp_path/'local/manifest.json').read_text())
    assert manifest['base_url'] is None
    page = next((tmp_path/'local/cases').glob('*.html')).read_text()
    assert 'rel="canonical"' not in page
    assert 'og:url' not in page
    assert result['archive_sha256'] == hashlib.sha256(Path(result['archive']).read_bytes()).hexdigest()


def test_public_index_addition_changes_release_identity(tmp_path):
    original = build_release(SELECTION, tmp_path/'original', source_revision='same-source')
    revised = build_release(SELECTION, tmp_path/'revised', source_revision='same-source',
                            index_extra_html='<p>Additional public explanation.</p>')
    assert original['revision'] != revised['revision']
    assert 'Additional public explanation.' in (tmp_path/'revised/index.html').read_text()


def test_comparison_keeps_two_denominators_explicit_and_refuses_unbound_evidence(tmp_path):
    selection = load_selection(SELECTION)
    therapy = selection.cases[0]
    assert [(row.numerator, row.denominator) for row in therapy.comparison] == [
        (310, 340), (310, 340), (291, 334), (178, 334)
    ]
    assert therapy.comparison[0].measure != therapy.comparison[1].measure
    bad = changed_selection(tmp_path, lambda r: r['cases'][0]['sections'][0].update(evidence_ids=['missing-evidence']))
    with pytest.raises(ReleaseError, match='article evidence must support'):
        build_release(bad, tmp_path/'site', source_revision='test')
    assert not (tmp_path/'site').exists()


def test_comparison_rejects_invalid_or_missing_population(tmp_path):
    bad = changed_selection(tmp_path, lambda r: r['cases'][0]['comparison'][0].update(denominator=0))
    with pytest.raises(ReleaseError, match='explicit valid counts'):
        load_selection(bad)
    bad = changed_selection(tmp_path, lambda r: r['cases'][0].update(comparison_label=''))
    with pytest.raises(ReleaseError, match='explicit population label'):
        load_selection(bad)
