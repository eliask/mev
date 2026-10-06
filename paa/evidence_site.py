"""Readable evidence excerpts; transport records stay in downloadable packets.

This projection formats explicit fields, never infers what a ballot means.
"""

import html
import json
import re
from urllib.parse import urljoin, urlsplit


def _e(value) -> str:
    return html.escape(str(value), quote=True)


def _documents(text: str) -> list[tuple[int, int, object]] | None:
    """Decode retained JSON or JSONL with its original character positions."""
    if not text.lstrip().startswith(("{", "[")):
        return None
    decoder = json.JSONDecoder()
    documents = []
    position = 0
    while position < len(text):
        while position < len(text) and text[position].isspace():
            position += 1
        if position == len(text):
            break
        try:
            value, end = decoder.raw_decode(text, position)
        except json.JSONDecodeError:
            # A partial/unknown transport view is linked, not printed as prose.
            return []
        documents.append((position, end, value))
        position = end
    return documents


def _table_rows(value: object) -> list[dict]:
    if not isinstance(value, dict):
        return []
    columns, rows = value.get("columnNames"), value.get("rowData")
    if not isinstance(columns, list) or not isinstance(rows, list):
        return []
    if any(not isinstance(c, str) for c in columns) or len(set(columns)) != len(columns):
        return []
    return [dict(zip(columns, row, strict=True)) for row in rows
            if isinstance(row, list) and len(row) == len(columns)]


def _array_rows(text: str, offset: int, key: str) -> list[tuple[int, int, list]]:
    """Locate rows in a validated table/projection without quotation searching."""
    match = re.search(r'"' + re.escape(key) + r'"\s*:\s*\[', text)
    if not match:
        return []
    decoder = json.JSONDecoder()
    position = match.end()
    found = []
    while position < len(text):
        while position < len(text) and (text[position].isspace() or text[position] == ','):
            position += 1
        if position >= len(text) or text[position] == ']':
            break
        value, end = decoder.raw_decode(text, position)
        if isinstance(value, list):
            found.append((offset + position, offset + end, value))
        position = end
    return found


def _person_row(row: list) -> dict | None:
    # The retained SaliDB person-row projection declares this eight-field order.
    if len(row) != 8 or not all(isinstance(v, str) for v in row):
        return None
    if not row[1].isdigit() or not row[4].isdigit():
        return None
    return dict(zip(("EdustajaId", "AanestysId", "EdustajaEtunimi", "EdustajaSukunimi",
                     "EdustajaHenkiloNumero", "EdustajaRyhmaLyhenne", "EdustajaAanestys", "Imported"),
                    row, strict=True))


def _selected_rows(source: dict, ref: dict, documents: list) -> list[dict]:
    selected = []
    for start, end, value in documents:
        if ref['end'] <= start or ref['start'] >= end:
            continue
        if isinstance(value, dict) and value.get('tableName') == 'SaliDBAanestys':
            selected.extend(_table_rows(value))
        elif isinstance(value, dict) and value.get('tableName') == 'SaliDBAanestysEdustaja':
            columns = value.get('columnNames', [])
            for row_start, row_end, row in _array_rows(source['text'][start:end], start, 'rowData'):
                if ref['start'] < row_end and ref['end'] > row_start and len(row) == len(columns):
                    selected.append(dict(zip(columns, row, strict=True)))
        elif 'SALIDBAANESTYSEDUSTAJA' in source['source_id'].upper():
            row = value.get('row') if isinstance(value, dict) else value
            if isinstance(row, list) and (person := _person_row(row)):
                selected.append(person)
    return selected


def _label(source: dict) -> str:
    title = source.get('title', 'Alkuperäinen lähde')
    if 'SALIDB' in title.upper() or 'SALIDB' in source['source_id'].upper():
        return 'Eduskunta · Äänestystulos'
    if 'QUERY' in source['source_id'].upper():
        return 'Eduskunta · Puhehaku'
    # Acquisition annotations are not the document title.
    return title.split(' — ')[0]


def _link(url: str, label: str) -> str:
    if urlsplit(url).scheme not in {'http', 'https'} or '{' in url or '}' in url:
        return _e(label)
    return f'<a href="{_e(url)}">{_e(label)}</a>'


def render_evidence(evidence_ids: list[str], refs: dict[str, dict], sources: dict[str, dict]) -> str:
    """Show exact prose or named vote fields; deduplicate fragmented row anchors.

    Evidence IDs remain HTML anchors. Hashes, offsets, acquisition notes and raw
    schemas are retained in the packet, never rendered, even in disclosures.
    """
    parsed = {key: _documents(source['text']) for key, source in sources.items()}
    events = {}
    for documents in parsed.values():
        for _, _, value in documents or []:
            if isinstance(value, dict) and value.get('tableName') == 'SaliDBAanestys':
                for row in _table_rows(value):
                    identity = str(row.get('AanestysId', ''))
                    if identity in events and events[identity] != row:
                        raise ValueError('Conflicting retained vote metadata')
                    events[identity] = row
    seen = set()
    parts = []
    for key in dict.fromkeys(evidence_ids):
        ref = refs[key]
        source = sources[ref['source_id']]
        anchor = f'<span id="{_e(key)}"></span>'
        parts.append(anchor)
        documents = parsed[ref['source_id']]
        if documents is None:
            identity = (ref['source_id'], ref['start'], ref['end'])
            if identity not in seen:
                seen.add(identity)
                parts.append(f'<blockquote>{_e(ref["quote"])}</blockquote><p>'
                             + _link(ref['url'], _label(source)) + '</p>')
            continue
        rows = _selected_rows(source, ref, documents)
        for row in rows:
            event_id = str(row.get('AanestysId', ''))
            person = row.get('EdustajaHenkiloNumero')
            identity = (event_id, person)
            if identity in seen:
                continue
            seen.add(identity)
            meta = events.get(event_id, row if person is None else {})
            title = str(meta.get('AanestysOtsikko') or 'Äänestys')
            date = str(meta.get('AanestysAlkuaika') or meta.get('IstuntoIlmoitettuAlkuaika') or '')[:10]
            url = urljoin('https://www.eduskunta.fi', str(meta.get('Url') or source['url']))
            if person is not None:
                name = str(row.get('EdustajaEtunimi', '')).strip() + ' ' + str(row.get('EdustajaSukunimi', '')).strip()
                vote = str(row.get('EdustajaAanestys', '')).strip().upper()
                parts.append(f'<p class="vote-evidence"><strong>{_e(name)}: {_e(vote)}</strong><br>'
                             f'{_e(date)} · {_link(url, title)}</p>')
            else:
                tally = ' · '.join(f'{label} {str(row[field]).strip()}' for field, label in (
                    ('AanestysTulosJaa', 'JAA'), ('AanestysTulosEi', 'EI'),
                    ('AanestysTulosTyhjia', 'TYHJÄ'), ('AanestysTulosPoissa', 'POISSA')) if field in row)
                parts.append(f'<p>{_e(date)} · {_link(url, title)}<br>{_e(tally)}</p>')
        if not rows and ('source', ref['source_id']) not in seen:
            seen.add(('source', ref['source_id']))
            parts.append('<p>' + _link(ref['url'], _label(source)) + '</p>')
    return ''.join(parts)
