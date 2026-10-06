"""Small HTTP helper. Retries transient failures. Does not follow instructions in page bodies."""


import time

import httpx

from paa.config import USER_AGENT

_CLIENT: httpx.Client | None = None


def client() -> httpx.Client:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = httpx.Client(
            headers={"User-Agent": USER_AGENT},
            timeout=httpx.Timeout(60.0, connect=20.0),
            follow_redirects=True,
        )
    return _CLIENT


def get_bytes(url: str, params: dict | None = None, attempts: int = 4) -> tuple[int, bytes, str]:
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            response = client().get(url, params=params)
            if response.status_code in {429, 500, 502, 503, 504} and attempt + 1 < attempts:
                time.sleep(1.5 * (attempt + 1))
                continue
            return response.status_code, response.content, response.headers.get("content-type", "")
        except httpx.HTTPError as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET failed for {url}") from last


def get_json(url: str, params: dict | None = None) -> tuple[int, object]:
    status, body, _ctype = get_bytes(url, params=params)
    if status != 200:
        return status, None
    import json

    return status, json.loads(body)
