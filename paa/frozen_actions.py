"""Replay the archived written-question and speech source slices offline."""

import hashlib
import json

from paa.config import FIXTURE_DIR
from paa.question_ledger import import_result, normalize_question_records, parse_question_row
from paa.speech_ledger import _search_request, import_speeches, normalize_speech_records, parse_speech_result


def import_action_fixtures(conn):
    from paa.frozen_vote import import_frozen_vote_fixture

    vote_stats = import_frozen_vote_fixture(conn)
    question_receipt = json.loads((FIXTURE_DIR / 'vaski_kk_fixture_coverage.json').read_text())
    question_records = []
    for query in question_receipt['queries']:
        body = (FIXTURE_DIR / query['fixture']).read_bytes()
        if hashlib.sha256(body).hexdigest() != query['fixture_sha256']:
            raise ValueError('Frozen written-question source changed')
        payload = json.loads(body)
        for row in payload['rowData']:
            question_records.append(parse_question_row(dict(zip(payload['columnNames'], row)),
                                                       retrieved_at=question_receipt['retrieved_at']))
    questions = normalize_question_records(question_records, coverage_scope='frozen_one_question_and_answer')
    questions['coverage'] = {
        'coverage_id': 'question-frozen-kk1-2023', 'source_id': 'SRC-EDUSKUNTA-VASKI',
        'kind': 'WRITTEN_QUESTION_REGISTER', 'state': 'DECLARED_SLICE', 'complete': False,
        'retrieved_at': question_receipt['retrieved_at'], 'source_records': len(question_records),
        'queries': question_receipt['queries'], 'limitations': ['One archived question and its answer only.'],
    }
    question_stats = import_result(conn, questions)
    speech_receipt = json.loads((FIXTURE_DIR / 'eduskunta_speeches_coverage.json').read_text())
    speech_records = []
    pages = []
    for window in speech_receipt['declared_windows']:
        for start in range(0, window['count'], 100):
            filename = f"eduskunta_speeches_{window['window_id']}_page{start}.json"
            body = (FIXTURE_DIR / filename).read_bytes()
            sha = hashlib.sha256(body).hexdigest()
            if sha != speech_receipt['fixture_response_sha256'][f"search-{window['window_id']}-{start}"]:
                raise ValueError('Frozen speech source changed')
            payload = json.loads(body)
            rows = payload['results']
            request = _search_request(window, start=start, max_results=100)
            for index, row in enumerate(rows):
                speech_records.append(parse_speech_result(row, raw_sha256=sha, raw_bytes=len(body), request=request,
                    window_id=window['window_id'], result_index=index, page_start=start,
                    retrieved_at=speech_receipt['retrieved_at']))
            pages.append({'fixture': filename, 'raw_sha256': sha, 'raw_bytes': len(body), 'rows': len(rows)})
    speeches = normalize_speech_records(speech_records)
    speeches['coverage'] = {
        'coverage_id': 'speech-frozen-four-day-windows', 'source_id': 'SRC-EDUSKUNTA-SPEECHES',
        'kind': 'PARLIAMENTARY_SPEECH_REGISTER', 'state': 'DECLARED_SLICE', 'complete': False,
        'retrieved_at': speech_receipt['retrieved_at'], 'source_records': len(speech_records),
        'windows': speech_receipt['declared_windows'], 'source_pages': pages,
        'limitations': ['Four archived one-day windows only; no assertion of national or term completeness.'],
    }
    speech_stats = import_speeches(conn, speeches)
    return {'questions': question_stats, 'speeches': speech_stats, 'votes': vote_stats, 'source_pages': pages}
