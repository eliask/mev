"""Local inference with immutable request receipts and a legacy line adapter."""


import asyncio
import fcntl
import hashlib
import json
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import jsonschema

from paa.config import DATA

ENDPOINT = os.environ.get("LLAMA_API_BASE", "http://127.0.0.1:8080").rstrip("/") + "/v1/chat/completions"
CLIENT_VERSION = "source-grounded-local-v1"


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


class LocalLLMClient:
    """Use the configured local server without replacing its model or jobs.

    Cache identity includes model/runtime metadata, prompts, inputs, schema
    and all decoding settings. Every request is replayable. Format validation
    remains separate from semantic validation and public admission.
    """

    def __init__(self, base_url: str | None = None, *, cache_dir: Path | None = None,
                 timeout: float = 240, retries: int = 2):
        self.base_url = (base_url or os.environ.get("LLAMA_API_BASE", "http://127.0.0.1:8080")).rstrip("/")
        self.base_url = self.base_url.removesuffix("/v1")
        if urlsplit(self.base_url).hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("PAA local inference requires a loopback server URL")
        self.cache_dir = cache_dir or DATA / "llm_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.retries = retries
        self.manifest: dict | None = None
        self._http = httpx.AsyncClient(timeout=timeout)

    @asynccontextmanager
    async def inference_slot(self, slots: int | None = None):
        """Bound simultaneous requests across cooperating local processes.

        Advisory locks release on cancellation or process exit. Cache hits do
        not occupy a slot. The serving API supplies the capacity; this lock
        coordinates PAA jobs and cannot account for unrelated server clients.
        """
        advertised = slots or (self.manifest or {}).get("total_slots") or 1
        capacity = max(1, int(advertised))
        configured = os.environ.get("PAA_LLM_MAX_INFLIGHT")
        if configured:
            capacity = min(capacity, max(1, int(configured)))
        directory = Path("/tmp/paa-inference-slots") / digest(self.base_url)[:24]
        directory.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        handle = None
        try:
            while handle is None:
                for index in range(capacity):
                    candidate = (directory / f"slot-{index}.lock").open("a")
                    try:
                        fcntl.flock(candidate.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        candidate.close()
                    else:
                        handle = candidate
                        break
                if handle is None:
                    if time.monotonic() - started > self.timeout:
                        raise TimeoutError("Timed out waiting for a shared local inference slot")
                    await asyncio.sleep(0.1)
            yield round(time.monotonic() - started, 4)
        finally:
            if handle is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()

    async def close(self) -> None:
        await self._http.aclose()

    async def discover(self) -> dict:
        response = await self._http.get(self.base_url + "/v1/models")
        response.raise_for_status()
        models = response.json().get("data") or []
        if not models:
            raise RuntimeError("local server advertises no loaded model")
        props_response = await self._http.get(self.base_url + "/props")
        props = props_response.json() if props_response.is_success else {}
        # The model-list endpoint omits the engine build. Bind cache identity
        # to the fingerprint exposed by a tiny inference, not a guessed version.
        runtime_payload = {
            "model": models[0]["id"], "messages": [{"role": "user", "content": "Reply with a period."}],
            "max_tokens": 1, "temperature": 0, "seed": 42,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        async with self.inference_slot(props.get("total_slots")):
            runtime = await self._http.post(self.base_url + "/v1/chat/completions", json=runtime_payload)
        runtime.raise_for_status()
        self.manifest = {
            "client_version": CLIENT_VERSION, "endpoint": self.base_url,
            "model_id": models[0]["id"], "model_meta": models[0].get("meta") or {},
            "model_path": props.get("model_path"), "model_ftype": props.get("model_ftype"),
            "model_digest": models[0].get("digest") or None,
            "chat_template_sha256": digest(props.get("chat_template")),
            "context_length": (props.get("default_generation_settings") or {}).get("n_ctx"),
            "total_slots": props.get("total_slots"), "server_build": runtime.json().get("system_fingerprint"),
            "identity_limitations": ["Model digest unavailable when the serving API does not expose it."],
        }
        return self.manifest

    async def request(self, task: str, system: str, user: str, *, schema: dict | None = None,
                      max_tokens: int = 1500, cache: bool = True, seed: int = 42,
                      enable_thinking: bool = False) -> dict:
        if self.manifest is None:
            await self.discover()
        payload = {
            "model": self.manifest["model_id"],
            "messages": [{"role": "system", "content": system.strip()},
                         {"role": "user", "content": user.strip()}],
            "max_tokens": max_tokens, "temperature": 0, "seed": seed,
            "presence_penalty": 0, "frequency_penalty": 0, "repeat_penalty": 1,
            "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        if schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "paa_source_task", "strict": True, "schema": schema},
            }
        key = digest({"task": task, "model": self.manifest, "request": payload})
        path = self.cache_dir / (key + ".json")
        if cache and path.exists():
            result = json.loads(path.read_text())
            if result.get("status") == "OK":
                return {**result, "cache_hit": True}
        if path.exists():
            # Keep failed/truncated attempts replayable when the same request
            # is retried after recovery. The canonical key points to the latest.
            previous_bytes = path.read_bytes()
            archive = self.cache_dir / "attempts" / (key + "-" + hashlib.sha256(previous_bytes).hexdigest()[:16] + ".json")
            archive.parent.mkdir(exist_ok=True)
            if not archive.exists():
                archive.write_bytes(previous_bytes)
        started = time.monotonic()
        result = {
            "request_id": key, "task": task, "model": self.manifest,
            "prompt_sha256": digest(system), "input_sha256": digest(user),
            "schema_sha256": digest(schema), "request": payload,
            "started_at": datetime.now(UTC).isoformat(), "cache_hit": False,
        }
        for attempt in range(self.retries + 1):
            try:
                async with self.inference_slot() as waited:
                    response = await self._http.post(self.base_url + "/v1/chat/completions", json=payload)
                result["shared_slot_wait_seconds"] = waited
                response.raise_for_status()
                raw = response.json()
                choice = raw["choices"][0]
                content = choice["message"].get("content") or ""
                result.update(
                    raw_response=raw, content=content, usage=raw.get("usage") or {},
                    timings=raw.get("timings") or {}, attempts=attempt + 1,
                    server_build=raw.get("system_fingerprint"),
                )
                if choice.get("finish_reason") == "length":
                    result.update(status="TRUNCATED", error="Output token limit reached.")
                    break
                if schema:
                    try:
                        parsed = json.loads(content)
                        jsonschema.Draft202012Validator(schema).validate(parsed)
                    except (json.JSONDecodeError, jsonschema.ValidationError) as error:
                        result.update(status="INVALID_OUTPUT", error=str(error)[:1200])
                        break
                    result["parsed"] = parsed
                result["status"] = "OK"
                break
            except (httpx.HTTPError, KeyError, IndexError, ValueError, TimeoutError) as error:
                result.update(status="FAILED", error=f"{type(error).__name__}: {error}", attempts=attempt + 1)
                if isinstance(error, httpx.HTTPStatusError):
                    result["http_error_body"] = error.response.text[:2000]
                    if 400 <= error.response.status_code < 500 and error.response.status_code not in {408, 429}:
                        break
                if attempt < self.retries:
                    await asyncio.sleep(min(2 ** attempt, 4))
        result["elapsed_seconds"] = round(time.monotonic() - started, 4)
        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        temporary.replace(path)
        return result

# Internal codes. Expand before anything is stored or shown.
TYPE_LABELS = {
    "SL": "VALUE_OR_SLOGAN",
    "BR": "BROAD_OBJECTIVE",
    "PO": "POSITION",
    "PR": "PROCESS_COMMITMENT",
    "PA": "PERSONAL_ACTION_COMMITMENT",
    "PD": "POLICY_DESIDERATUM",
    "CO": "COLLECTIVE_ACTION_COMMITMENT",
    "CF": "CAUSAL_EFFECT_FORECAST",
    "RS": "REPORTED_SPEECH",
}
_LINE = re.compile(r"^\s*\[?(\d+)\]?\s+([A-Z]{2})\b", re.MULTILINE)

PROMPT_A = """Luokittele vaalikampanjan virkkeet.

TYYPPI (SL/BR/PO/PR/PA/PD/CO/CF/RS):
SL iskulause tai arvolause
BR laaja tavoite ilman mittaria
PO kanta, ei omaa tekoa
PR pyrkimys tai päätöstapa
PA puhujan oma konkreettinen teko
PD passiivinen toive
CO joukon sitoumus
CF seurausarvio
RS toisen lainaus

Kentän nimi ei tee virkkeestä lupausta.
Tulosta VAIN rivit: NUMERO TYYPPI
Ei selityksiä, ei sulkeita, ei muuta tekstiä.
Jos et tiedä, älä tulosta riville mitään.
Jos koko erässä ei ole yhtään luokkaa, tulosta NONE.

Esimerkki — syöte:
[1] Suomi ensin.
[2] Kannatan koulutusta.
[3] Teen aloitteen asiasta 1.6.2025 mennessä.
[4] Vero on alennettava.
[5] Tämä on taustalause.

Esimerkki — tuloste:
1 SL
2 PO
3 PA
4 PD
"""

PROMPT_B = PROMPT_A + """
Kannatan X on PO.
Teen aloitteen on PA.
Äänestän esityksen puolesta on PA.
on saatava on PD.
on poistettava on PD.
Pyrin on PR.
Lupaan pohjata päätökset on PR.
esitämme on CO.
laskemme on CO.
"""

PROMPT_C = PROMPT_B + """
kuntoon ilman lukua on BR.
Parannamme ilman mittaria on BR.
Lupaan tehdä teon on PA.
Lupaan pohjata on PR.
"""


def complete(system: str, user: str, max_tokens: int) -> tuple[str, dict]:
    payload = {
        "messages": [
            {"role": "system", "content": system.strip()},
            {"role": "user", "content": user.strip()},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    response = httpx.post(ENDPOINT, json=payload, timeout=180)
    response.raise_for_status()
    data = response.json()
    choice = data["choices"][0]
    if choice.get("finish_reason") == "length":
        raise RuntimeError(f"truncated at max_tokens={max_tokens}")
    return (choice["message"].get("content") or "").strip(), data.get("usage") or {}


def parse_codes(text: str) -> dict[int, str]:
    """Lenient line parse. Unknown codes and prose are dropped."""
    if text.strip().upper() == "NONE":
        return {}
    found: dict[int, str] = {}
    for match in _LINE.finditer(text):
        code = match.group(2)
        if code in TYPE_LABELS:
            found[int(match.group(1))] = code
    return found


def expand(code: str) -> str:
    return TYPE_LABELS[code]
