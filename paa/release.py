"""Build a selected, source-reviewed static beta; never copy a workbench tree."""

import hashlib
import json
import re
import shutil
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import final
from urllib.parse import urlencode, urlsplit

from paa.case_site import _document, _e, render_case
from paa.config import FIXTURE_DIR
from paa.inquiry_cases import load_reviewed_cases, validate_case


class ReleaseError(ValueError):
    """The declared publication cannot be built at its selected scope."""


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n'


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _url(value: str) -> None:
    parts = urlsplit(value)
    if parts.scheme not in {'http', 'https'} or not parts.netloc or parts.username or parts.password:
        raise ReleaseError('public URLs must be credential-free HTTP(S)')


@final
@dataclass(frozen=True, slots=True, kw_only=True, match_args=False)
class SourcePermission:
    source_id: str
    basis: str
    notice: str
    terms_url: str

    def __post_init__(self) -> None:
        if self.basis not in {'OFFICIAL_DOCUMENT', 'LICENSED_CONTENT', 'SHORT_EXCERPTS'}:
            raise ReleaseError('explicit source distribution basis required')
        if not self.source_id or not self.notice:
            raise ReleaseError('source permission identity and notice required')
        _url(self.terms_url)


@final
@dataclass(frozen=True, slots=True, kw_only=True, match_args=False)
class ArticleSection:
    title: str
    text: str
    claim_dimensions: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    source_label: str

    def __post_init__(self) -> None:
        if not all((self.title, self.text, self.source_label)):
            raise ReleaseError('article section needs a question, answer and source label')
        for rows in (self.claim_dimensions, self.evidence_ids):
            if not isinstance(rows, tuple) or not rows or not all(isinstance(row, str) for row in rows):
                raise ReleaseError('article support must be an owned nonempty tuple')


@final
@dataclass(frozen=True, slots=True, kw_only=True, match_args=False)
class ComparisonRow:
    period: str
    measure: str
    numerator: int
    denominator: int
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if (type(self.numerator) is not int or type(self.denominator) is not int
                or not 0 <= self.numerator <= self.denominator or self.denominator <= 0):
            raise ReleaseError('comparison requires explicit valid counts and denominator')
        if not self.period or not self.measure or not isinstance(self.evidence_ids, tuple) or not self.evidence_ids:
            raise ReleaseError('comparison requires period, measure and owned evidence references')


@final
@dataclass(frozen=True, slots=True, kw_only=True, match_args=False)
class FeaturedCase:
    case_id: str
    sources_file: str
    reviews_file: str
    packet_sha256: str
    takeaway: str
    why_it_matters: str
    main_limit: str
    source_period: str
    review_date: str
    permissions: tuple[SourcePermission, ...]
    sequence: tuple[str, ...] = ()
    article_title: str = ''
    sections: tuple[ArticleSection, ...] = ()
    comparison: tuple[ComparisonRow, ...] = ()
    comparison_label: str = ''
    audience: str = 'GENERAL'

    def __post_init__(self) -> None:
        if not isinstance(self.permissions, tuple) or not all(isinstance(row, SourcePermission) for row in self.permissions):
            raise ReleaseError('source permissions must be an owned tuple of records')
        if not isinstance(self.sequence, tuple) or not all(isinstance(row, str) and row for row in self.sequence):
            raise ReleaseError('publication sequence must be an owned tuple of text')
        if not isinstance(self.sections, tuple) or not self.sections or not all(isinstance(row, ArticleSection) for row in self.sections):
            raise ReleaseError('publication needs owned source-backed article sections')
        if not isinstance(self.comparison, tuple) or not all(isinstance(row, ComparisonRow) for row in self.comparison):
            raise ReleaseError('comparison must be an owned tuple of records')
        if self.comparison and (len(self.comparison) < 2 or not self.comparison_label):
            raise ReleaseError('comparison needs two measures and an explicit population label')
        if not self.article_title or self.audience not in {'GENERAL', 'SPECIALIST', 'ONGOING'}:
            raise ReleaseError('article title and declared reader audience required')
        if not re.fullmatch(r'[a-z0-9][a-z0-9-]*', self.case_id):
            raise ReleaseError('case ID must be a stable path-safe slug')
        for value in (self.sources_file, self.reviews_file):
            if Path(value).name != value or not value.endswith(('.json', '.jsonl')):
                raise ReleaseError('publication inputs must be named bundled fixtures')
        if not re.fullmatch(r'[a-f0-9]{64}', self.packet_sha256):
            raise ReleaseError('editorial reading must bind the complete compiled packet')
        if not all((self.takeaway, self.why_it_matters, self.main_limit, self.source_period)):
            raise ReleaseError('answer, usefulness, limit and time scope required')
        date.fromisoformat(self.review_date)
        if len({row.source_id for row in self.permissions}) != len(self.permissions):
            raise ReleaseError('duplicate source permission')


@final
@dataclass(frozen=True, slots=True, kw_only=True, match_args=False)
class ReleaseSelection:
    release_id: str
    checked_on: str
    publisher: str
    correction_url: str
    introduction: str
    cases: tuple[FeaturedCase, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.cases, tuple) or not all(isinstance(row, FeaturedCase) for row in self.cases):
            raise ReleaseError('selected cases must be an owned tuple of records')
        if not re.fullmatch(r'[a-z0-9][a-z0-9-]*', self.release_id):
            raise ReleaseError('release ID must be a stable path-safe slug')
        checked = date.fromisoformat(self.checked_on)
        if not self.publisher or not self.introduction:
            raise ReleaseError('publisher and introduction required')
        _url(self.correction_url)
        if not 3 <= len(self.cases) <= 5 or len({row.case_id for row in self.cases}) != len(self.cases):
            raise ReleaseError('beta selection needs three to five distinct cases')
        if any(date.fromisoformat(row.review_date) > checked for row in self.cases):
            raise ReleaseError('case review cannot follow the declared release check')


def load_selection(path: Path) -> ReleaseSelection:
    raw = json.loads(path.read_text(encoding='utf-8'))
    if raw.pop('schema_version', None) != 'paa.publication_selection.v2':
        raise ReleaseError('unsupported publication selection')
    cases = tuple(FeaturedCase(**{**row, 'permissions': tuple(SourcePermission(**item) for item in row['permissions']),
                                'sequence': tuple(row.get('sequence', ())),
                                'sections': tuple(ArticleSection(**{**item,
                                    'claim_dimensions': tuple(item['claim_dimensions']),
                                    'evidence_ids': tuple(item['evidence_ids'])}) for item in row.get('sections', ())),
                                'comparison': tuple(ComparisonRow(**{**item,
                                    'evidence_ids': tuple(item['evidence_ids'])}) for item in row.get('comparison', ()))})
                  for row in raw.pop('cases'))
    return ReleaseSelection(**raw, cases=cases)


def _selected_packets(selection: ReleaseSelection, fixture_dir: Path) -> list[dict]:
    packets = []
    for entry in selection.cases:
        cases = load_reviewed_cases(fixture_dir / entry.sources_file, fixture_dir / entry.reviews_file)
        matches = [row for row in cases if row['case_id'] == entry.case_id]
        if len(matches) != 1:
            raise ReleaseError(f'selected case must resolve once: {entry.case_id}')
        packet = matches[0]
        if _sha(_json(packet)) != entry.packet_sha256:
            raise ReleaseError(f'case changed since editorial reading: {entry.case_id}')
        if not packet['claims'] or any(row['state'] != 'SOURCE_REVIEWED' for row in packet['claims']):
            raise ReleaseError('model candidates cannot enter the selected release')
        claims = {row['dimension']: row for row in packet['claims']}
        evidence_ids = {row['evidence_id'] for row in packet['evidence']}
        for section in entry.sections:
            if not set(section.claim_dimensions) <= claims.keys():
                raise ReleaseError('article section refers to an unknown reviewed claim')
            supported = {key for dimension in section.claim_dimensions for key in claims[dimension]['evidence_ids']}
            if not set(section.evidence_ids) <= supported:
                raise ReleaseError('article evidence must support its selected reviewed claims')
        if any(not set(row.evidence_ids) <= evidence_ids for row in entry.comparison):
            raise ReleaseError('comparison evidence must resolve in the reviewed packet')
        source_ids = {row['source_id'] for row in packet['sources']}
        if source_ids != {row.source_id for row in entry.permissions}:
            raise ReleaseError('every included source needs an explicit distribution basis')
        for permission in entry.permissions:
            source = next(row for row in packet['sources'] if row['source_id'] == permission.source_id)
            if permission.basis == 'SHORT_EXCERPTS' and not (
                source.get('coverage_basis') == 'SELECTED_EXCERPTS'
                or source.get('coverage_basis', '').endswith('SHORT_PROJECTION')
            ):
                raise ReleaseError('short-excerpt permission cannot redistribute a full source')
        # Public inquiry contract only: keep exact evidence, omit optional internal
        # CLI/replay artifacts rather than uploading their entire research history.
        keys = ('schema_version', 'case_id', 'title', 'question', 'sources', 'evidence',
                'claims', 'unknowns', 'selection_basis', 'limits', 'source_episode_id', 'question_id',
                'canonical_relation_review_id', 'canonical_statement_id', 'canonical_object_id', 'canonical_target_scope')
        projected = {key: packet[key] for key in keys if key in packet}
        validate_case(projected)
        if re.search(r'/home/|file://|OPENCODE_API_KEY|Authorization: Bearer', _json(projected)):
            raise ReleaseError('selected export contains a private path or credential marker')
        packets.append(projected)
    return packets


_SHARE_JS = '''/* MIT License; Copyright (c) 2026 MeV/PAA contributors. See LICENSE.txt. */
"use strict";
const input = document.getElementById("case-url");
const status = document.getElementById("share-status");
if (input && !input.value) input.value = location.href.split("#")[0];
const copy = document.getElementById("copy-url");
if (copy) copy.disabled = false;
if (copy) copy.addEventListener("click", async () => {
  try {
    if (!navigator.clipboard) throw new Error("clipboard unavailable");
    await navigator.clipboard.writeText(input.value);
    status.textContent = "Linkki kopioitu.";
  } catch (_) {
    input.focus(); input.select();
    status.textContent = "Automaattinen kopiointi ei onnistu. Kopioi valittu osoite.";
  }
});
function openEvidence() {
  const element = document.getElementById(decodeURIComponent(location.hash.slice(1)));
  if (!element) return;
  for (let parent = element.parentElement; parent; parent = parent.parentElement)
    if (parent.tagName === "DETAILS") parent.open = true;
  element.scrollIntoView();
}
addEventListener("hashchange", openEvidence);
if (location.hash) openEvidence();
'''


def _citation_text(packet: dict, metadata: dict) -> str:
    sources = {row['source_id']: row for row in packet['sources']}
    lines = [packet['title'], packet['question']['text'], metadata['takeaway'],
             f"Julkaisija: {metadata['publisher']}", f"Lähdeluku: {metadata['review_date']}",
             'Tekoälyavusteinen lähdeluku; ei riippumatonta ihmistarkistusta.',
             f"Jakso: {metadata['source_period']}", f"Versio: {metadata['revision']}",
             f"Keskeinen rajaus: {metadata['main_limit']}", '']
    for ref in packet['evidence']:
        source = sources[ref['source_id']]
        lines.extend([ref['evidence_id'], source['title'], ref['url'], ref['locator'], ref['quote'], ''])
    return '\n'.join(lines) + '\n'


def build_release(selection_path: Path, destination: Path, *, source_revision: str,
                  base_url: str = '', fixture_dir: Path = FIXTURE_DIR,
                  extra_files: dict[str, str | bytes] | None = None,
                  index_extra_html: str = '',
                  manifest_extra: dict | None = None,
                  shared_publication: dict | None = None) -> dict:
    """Build fresh selected files and a ZIP from the same validated bytes."""
    selection = load_selection(selection_path)
    if not source_revision or '\n' in source_revision or '/' in source_revision:
        raise ReleaseError('explicit code/source revision required')
    if base_url:
        _url(base_url)
        if urlsplit(base_url).query or urlsplit(base_url).fragment:
            raise ReleaseError('hosting base URL must not contain a query or fragment')
        base_url = base_url.rstrip('/') + '/'
    archive = destination.with_name(destination.name + '.zip')
    if destination.exists() or archive.exists():
        raise ReleaseError('release destination and archive must be new')
    packets = _selected_packets(selection, fixture_dir)
    config = asdict(selection)
    producer_versions = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in sorted(Path(__file__).parent.glob('*.py'))}
    extra_digests = {
        name: hashlib.sha256(content if isinstance(content, bytes) else content.encode('utf-8')).hexdigest()
        for name, content in sorted((extra_files or {}).items())
    }
    revision = _sha(_json({
        'selection': config,
        'source_revision': source_revision,
        'producer_versions': producer_versions,
        'base_url': base_url,
        'extra_file_sha256': extra_digests,
        'index_extra_html': index_extra_html,
        'manifest_extra': manifest_extra or {},
        'shared_publication': shared_publication or {},
    }))
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='paa-beta-', dir=destination.parent))
    archive_created = False
    try:
        (staging / 'cases').mkdir()
        items, manifest_cases = [], []
        for packet, entry in zip(packets, selection.cases, strict=True):
            relative = f'cases/{entry.case_id}.html'
            canonical = base_url + relative if base_url else ''
            correction = selection.correction_url + ('&' if '?' in selection.correction_url else '?') + urlencode({
                'title': f'PAA-korjaus: {entry.case_id}',
                'body': f'Tapaus: {entry.case_id}\nVersio: {revision}\nKorjattava väite ja lähde:\n',
            })
            metadata = {'publisher': selection.publisher, 'review_date': entry.review_date,
                        'source_period': entry.source_period, 'takeaway': entry.takeaway,
                        'why_it_matters': entry.why_it_matters, 'main_limit': entry.main_limit,
                        'revision': revision, 'canonical_url': canonical, 'correction_url': correction,
                        'sequence': list(entry.sequence),
                        'article_title': entry.article_title,
                        'sections': [asdict(row) for row in entry.sections],
                        'comparison': [asdict(row) for row in entry.comparison],
                        'comparison_label': entry.comparison_label,
                        'permissions': [asdict(row) for row in entry.permissions]}
            if shared_publication:
                metadata.update(shared_publication)
            packet = {**packet, 'publication': metadata}
            (staging / relative).write_text(render_case(packet, publication=metadata), encoding='utf-8')
            research_path = relative.replace('.html', '-research.html')
            (staging / research_path).write_text(render_case(packet, research_for=entry.case_id), encoding='utf-8')
            (staging / relative.replace('.html', '.json')).write_text(_json(packet), encoding='utf-8')
            (staging / relative.replace('.html', '.txt')).write_text(_citation_text(packet, metadata), encoding='utf-8')
            label = {'SPECIALIST': 'Menetelmäesimerkki · lainsäädäntö',
                     'ONGOING': 'Tutkimus kesken · päätösasiakirja puuttuu', 'GENERAL': 'Tutkimus'}[entry.audience]
            items.append(f'<article class="case-card" data-audience="{_e(entry.audience)}"><p class="eyebrow">{label}</p>'
                         f'<h2><a href="{_e(relative)}">{_e(entry.article_title)}</a></h2>'
                         f'<p>{_e(entry.takeaway)}</p></article>')
            manifest_cases.append({'case_id': entry.case_id, 'reviewed_packet_sha256': entry.packet_sha256,
                'public_packet_sha256': _sha(_json(packet)), 'path': relative, 'research_path': research_path,
                'review_date': entry.review_date,
                'source_period': entry.source_period, 'sources': [{key: row[key] for key in
                    ('source_id', 'url', 'text_sha256', 'raw_sha256')} for row in packet['sources']],
                'permissions': [asdict(row) for row in entry.permissions]})
        case_count = len(selection.cases)
        count_label = {1: 'Yksi', 2: 'Kaksi', 3: 'Kolme', 4: 'Neljä'}.get(case_count, str(case_count))
        body = ('<h1>Poliittinen muisti</h1>'
                f'<p>{_e(selection.introduction)}</p>'
                f'<p class="review-status">{count_label} lähteisiin perustuvaa tutkimusta · rajattu julkaisu</p>'
                + ''.join(items) +
                f'<p class="review-status">Julkaisija: {_e(selection.publisher)} · {_e(selection.checked_on)}<br>'
                'PAA: Political Accountability Analytics. Tekoälyavusteisia lähdelukuja; '
                'ei riippumatonta ihmistarkistusta.</p>' +
                f'<p><a href="{_e(selection.correction_url)}">Ilmoita korjaus</a> · '
                '<a download href="manifest.json">Lataa julkaisun sisältöluettelo</a> · '
                '<a href="terms.html">Aineiston käyttö ja menetelmä</a></p>')
        if index_extra_html:
            body += index_extra_html
        description = 'Lähteisiin perustuvia vastauksia politiikan tekoja, sääntöjä ja toteutumista koskeviin kysymyksiin.'
        (staging / 'index.html').write_text(_document('Tutkitut kysymykset', body,
            description=description, canonical_url=base_url + 'index.html' if base_url else ''), encoding='utf-8')
        (staging / 'share.js').write_text(_SHARE_JS, encoding='utf-8')
        notices = ''.join(f'<h2>{_e(row["case_id"])}</h2><ul>' + ''.join(
            f'<li>{_e(p["source_id"])}: {_e(p["notice"])} <a href="{_e(p["terms_url"])}">Peruste</a></li>'
            for p in row['permissions']) + '</ul>' for row in manifest_cases)
        (staging / 'terms.html').write_text(_document('Aineiston käyttö ja menetelmä',
            '<nav><a href="index.html">Tutkitut kysymykset</a></nav><h1>Aineiston käyttö ja menetelmä</h1>'
            '<p>Alkuperäislähteiden oikeudet ja julkaistun analyysin oikeudet ovat eri asioita. '
            'Ohjelmisto on <a href="LICENSE.txt">MIT-lisensoitu</a>. Lisenssi ei muuta '
            'alkuperäislähteiden ehtoja eikä myönnä uutta lupaa analyysitekstien uudelleenkäyttöön. '
            'Lähdekohtaiset perusteet on lueteltu alla.</p>'
            '<p>Väitteet perustuvat nimettyjen lähdeversioiden tekoälyavusteiseen lähdelukuun. '
            'Täsmällinen lainaus tai toimiva ohjelmisto ei yksin varmista tulkinnan oikeellisuutta. '
            'Tarkastelun ajallinen ja sisällöllinen rajaus on esitetty tapaussivulla.</p>' + notices), encoding='utf-8')
        license_text = (Path(__file__).parent / 'code_license.txt').read_text(encoding='utf-8')
        (staging / 'LICENSE.txt').write_text(license_text, encoding='utf-8')
        for relative, content in (extra_files or {}).items():
            target = staging / relative
            if target.exists() or '..' in Path(relative).parts or relative.startswith('/'):
                raise ReleaseError(f'refusing extra artifact path {relative!r}')
            target.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                target.write_bytes(content)
            else:
                target.write_text(content, encoding='utf-8')
        files = {str(path.relative_to(staging)): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in sorted(staging.rglob('*')) if path.is_file()}
        manifest = {'schema_version': 'paa.public_beta_manifest.v1', 'release_id': selection.release_id,
                    'revision': revision, 'checked_on': selection.checked_on, 'source_revision': source_revision,
                    'base_url': base_url or None, 'deployment_status': 'NOT_DEPLOYED',
                    'selection_sha256': hashlib.sha256(selection_path.read_bytes()).hexdigest(),
                    'producer_versions': producer_versions,
                    'publisher': selection.publisher, 'cases': manifest_cases, 'files': files,
                    'code_license': 'MIT', 'source_data_license': 'SOURCE_SPECIFIC',
                    'exclusions': ['unselected cases', 'person corpus', 'model proposals', 'internal replay commands',
                                   'working notes', 'credentials', 'live database'],
                    'checks': ['version-bound selected reviews', 'candidate exclusion', 'source permission accounting',
                               'fresh destination', 'content hashes'],
                    'limits': ['No independent human review or general analytical accuracy certification.',
                               'Browser and source-link checks are separate from build validation.',
                               'No public destination or analysis reuse license is assumed.']}
        if manifest_extra:
            overlap = set(manifest_extra) & set(manifest)
            if overlap:
                raise ReleaseError(f'manifest_extra collides with reserved keys: {sorted(overlap)}')
            manifest.update(manifest_extra)
        (staging / 'manifest.json').write_text(_json(manifest), encoding='utf-8')
        with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as stream:
            archive_created = True
            for path in sorted(staging.rglob('*')):
                if path.is_file():
                    stream.write(path, arcname=str(path.relative_to(staging)))
        staging.rename(destination)
    except BaseException:
        if archive_created:
            archive.unlink()
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {'destination': str(destination), 'archive': str(archive), 'release_id': selection.release_id,
            'revision': revision, 'cases': len(packets), 'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}
