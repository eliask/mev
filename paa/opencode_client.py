"""Explicit public-source inference through selected OpenCode models.

The key is read as data, never sourced as shell code. Only the documented free
LongCat route and explicitly selected Go Muse route are supported. There is no
model or endpoint fallback.
Requests and responses are retained without credentials. The analytical
consumer still owns quotation binding and semantic admission.
"""

import asyncio
import hashlib
import json
import re
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import httpx

from paa.llm_client import digest

MODEL = "longcat-2.5-preview-free"
MUSE_MODEL = "muse-spark-1.3-contributor"
BASE_URL = "https://opencode.ai/zen/v1"
GO_BASE_URL = "https://opencode.ai/zen/go/v1"
CLIENT_VERSION = "paa.opencode.public_source.v8"


def _model_identity(manifest: dict) -> dict:
    # The gateway regenerates the advertised creation time on every discovery.
    # Retain it as a receipt observation, never as a model revision identifier.
    return {**manifest, "model_metadata": {key: value for key, value in manifest["model_metadata"].items()
                                          if key != "created"}}


def read_key(path: Path) -> str:
    """Accept one literal shell assignment, without evaluating any shell code."""
    matches = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        match = re.fullmatch(r"\s*(?:export\s+)?OPENCODE_API_KEY\s*=\s*(.*?)\s*", line)
        if match:
            value = match.group(1)
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            if not value or any(c.isspace() or c in "`$\\\"'" for c in value):
                raise ValueError("API key must be a nonempty literal assignment")
            matches.append(value)
    if len(matches) != 1:
        raise ValueError("Exactly one literal OPENCODE_API_KEY assignment is required")
    return matches[0]


class OpenCodeClient:
    """Bounded selected-model boundary with source-sensitive replay receipts."""

    def __init__(self, *, key_file: Path, cache_dir: Path,
                 timeout: float = 180, subscription: bool = False, model: str = MODEL,
                 transport: httpx.AsyncBaseTransport | None = None):
        if type(subscription) is not bool:
            raise TypeError("Subscription route selection requires a boolean")
        if model not in {MODEL, MUSE_MODEL}:
            raise ValueError("Unsupported explicit model; no fallback")
        if model == MUSE_MODEL and not subscription:
            raise ValueError("Muse requires the explicitly selected Go subscription route")
        self._model_id = model
        self._responses = model == MUSE_MODEL
        self._base_url = GO_BASE_URL if subscription else BASE_URL
        self._endpoint = self._base_url + ("/responses" if self._responses else "/chat/completions")
        self._key = read_key(key_file)
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._http = httpx.AsyncClient(timeout=timeout, follow_redirects=False,
                                      transport=transport)
        self.manifest: dict | None = None
        self._discovery_lock = asyncio.Lock()

    async def close(self) -> None:
        await self._http.aclose()

    async def discover(self) -> dict:
        response = await self._http.get(self._base_url + "/models")
        response.raise_for_status()
        metadata = response.json()
        if not isinstance(metadata, dict) or not isinstance(metadata.get("data"), list):
            raise TypeError("Provider model list is not an object containing data")
        selected = [row for row in metadata["data"] if isinstance(row, dict) and row.get("id") == self._model_id]
        if len(selected) != 1:
            raise ValueError("Selected model is not uniquely advertised; no fallback")
        self.manifest = {"provider": "OPENCODE_GO" if self._base_url == GO_BASE_URL else "OPENCODE_ZEN",
                         "client_version": CLIENT_VERSION,
                         "endpoint": self._endpoint, "model_id": self._model_id,
                         "model_metadata": selected[0], "model_digest": None,
                         "identity_limit": "Gateway advertises a model ID, not immutable model weights.",
                         "public_source_only": True, "paid_fallback": False,
                         "free_model": self._model_id == MODEL,
                         "wire_format": "responses" if self._responses else "chat_completions"}
        return self.manifest

    def _safe(self, value):
        if isinstance(value, str):
            return value.replace(self._key, "[REDACTED]")
        if isinstance(value, dict):
            return {self._safe(key): self._safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._safe(item) for item in value]
        return value

    async def request(self, task: str, system: str, user: str, *, max_tokens: int = 4096,
                      public_sources: bool, reuse_cache: bool = True,
                      enable_thinking: bool | None = None,
                      reasoning_effort: str | None = None) -> dict:
        if public_sources is not True:
            raise ValueError("Remote inference requires an explicit public-source declaration")
        if type(max_tokens) is not int or not 1 <= max_tokens <= 8192:
            raise ValueError("Output limit must be an integer in 1..8192")
        if enable_thinking is not None and type(enable_thinking) is not bool:
            raise TypeError("Thinking selection must be a boolean or explicit provider default")
        if self._responses and enable_thinking is not None:
            raise ValueError("LongCat thinking selection is not a Muse Responses parameter")
        if reasoning_effort is not None and (
                not self._responses or reasoning_effort not in {"minimal", "low", "medium", "high"}):
            raise ValueError("Explicit reasoning effort requires a supported Muse setting")
        if any(self._key in item for item in (task, system, user)):
            raise ValueError("Credential cannot be part of a model input")
        if self.manifest is None:
            try:
                async with self._discovery_lock:
                    if self.manifest is None:
                        await self.discover()
            except (asyncio.CancelledError, httpx.HTTPError, ValueError, TypeError) as error:
                cancelled = isinstance(error, asyncio.CancelledError)
                result = {"status": "CANCELLED" if cancelled else "FAILED", "task": task,
                          "error": "Discovery: " + type(error).__name__,
                          "endpoint": self._endpoint, "model_id": self._model_id, "started_at": datetime.now(UTC).isoformat(),
                          "model_admission": "PROPOSED_NOT_ADMITTED"}
                result["request_id"] = digest({"task": task, "system": system, "user": user,
                                               "endpoint": self._endpoint, "model": self._model_id})
                (self.cache_dir / (result["request_id"] + "-discovery.json")).write_text(
                    json.dumps(self._safe(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                if cancelled:
                    raise
                return self._safe(result)
        payload = {"model": self._model_id, "messages": [{"role": "system", "content": system},
                                                 {"role": "user", "content": user}],
                   "max_tokens": max_tokens, "temperature": 0}
        if self._responses:
            payload = {"model": self._model_id, "instructions": system, "input": user,
                       "max_output_tokens": max_tokens, "temperature": 0, "store": False}
            if reasoning_effort is not None:
                payload["reasoning"] = {"effort": reasoning_effort}
        if enable_thinking is not None:
            payload["thinking"] = {"type": "enabled" if enable_thinking else "disabled"}
        key = digest({"task": task, "manifest": _model_identity(self.manifest), "request": payload})
        path = self.cache_dir / (key + ".json")
        cache_error = None
        # Reuse is optional; this research boundary always retains each attempt.
        if reuse_cache and path.exists():
            try:
                previous = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeError):
                previous = None
                cache_error = "CORRUPT_CACHE_ARCHIVED"
            if (isinstance(previous, dict) and previous.get("status") == "OK" and previous.get("request") == payload
                    and previous.get("request_id") == key and isinstance(previous.get("model"), dict)
                    and isinstance(previous["model"].get("model_metadata"), dict)
                    and _model_identity(previous["model"]) == _model_identity(self.manifest)):
                checked = {}
                try:
                    raw = previous.get("raw_response")
                    self._read_selected(raw, checked, reasoning_effort)
                    valid = (checked.get("status") == "OK" and checked.get("content") == previous.get("content")
                             and previous.get("response_sha256") == digest(raw))
                except (ValueError, TypeError, KeyError, IndexError):
                    valid = False
                if valid:
                    return self._safe({**previous, "cache_hit": True})
                cache_error = "INVALID_CACHE_ARCHIVED"
        started = time.monotonic()
        session = "paa-public-research-" + key[:32]
        result = {"request_id": key, "task": task, "model": self.manifest, "request": payload,
                  "session_id": session,
                  "prompt_sha256": digest(system), "input_sha256": digest(user),
                  "started_at": datetime.now(UTC).isoformat(), "cache_hit": False,
                  "model_admission": "PROPOSED_NOT_ADMITTED"}
        if cache_error:
            result["cache_error"] = cache_error
        try:
            response = await self._http.post(self._endpoint, json=payload,
                                              headers={"Authorization": "Bearer " + self._key,
                                                       "User-Agent": "paa-codex-public-research/0.1",
                                                       "x-opencode-session": session})
            result["http_status"] = response.status_code
            if not response.is_success:
                result.update(status="FAILED", error="Provider returned HTTP " + str(response.status_code),
                              error_body=self._safe(response.text)[:2000])
            else:
                raw = self._safe(response.json())
                result.update(raw_response=raw, response_sha256=digest(raw))
                self._read_selected(raw, result, reasoning_effort)
        except asyncio.CancelledError:
            result.update(status="CANCELLED", error="Request cancelled")
            raise
        except (httpx.HTTPError, ValueError, TypeError, KeyError, IndexError) as error:
            # Exceptions can include request representations; retain the class only.
            result.update(status="FAILED", error=type(error).__name__)
        finally:
            result["elapsed_seconds"] = round(time.monotonic() - started, 4)
            self._retain(path, key, result)
        return self._safe(result)

    def _read_selected(self, raw: object, result: dict, reasoning_effort: str | None) -> None:
        if not self._responses:
            self._read_chat(raw, result)
            return
        self._read_responses(raw, result)
        observed = raw.get("reasoning") if isinstance(raw, dict) else None
        result["observed_reasoning_effort"] = observed.get("effort") if isinstance(observed, dict) else None
        if (reasoning_effort is not None and result["observed_reasoning_effort"] is not None
                and result["observed_reasoning_effort"] != reasoning_effort):
            result.update(status="INVALID_OUTPUT", error="Provider returned different reasoning configuration")

    def _retain(self, path: Path, key: str, result: dict) -> None:
        if path.exists():
            previous_bytes = path.read_bytes()
            archive = self.cache_dir / "attempts" / (key + "-" + hashlib.sha256(previous_bytes).hexdigest()[:16] + ".json")
            archive.parent.mkdir(exist_ok=True)
            if not archive.exists():
                archive.write_bytes(previous_bytes)
        temporary = path.with_suffix("." + uuid.uuid4().hex + ".tmp")
        temporary.write_text(json.dumps(self._safe(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def _read_chat(raw: object, result: dict) -> None:
        if not isinstance(raw, dict) or not isinstance(raw.get("choices"), list):
            raise TypeError("Malformed provider response")
        choice = raw["choices"][0]
        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
            raise TypeError("Malformed provider choice")
        content = choice["message"].get("content")
        result.update(raw_response=raw, content=content, usage=raw.get("usage", {}),
                      finish_reason=choice.get("finish_reason"))
        if choice.get("finish_reason") == "length":
            result.update(status="TRUNCATED", error="Output token limit reached")
        elif not isinstance(content, str) or not content.strip():
            result.update(status="INVALID_OUTPUT", error="No completed textual answer")
        else:
            result["status"] = "OK"

    @staticmethod
    def _read_responses(raw: object, result: dict) -> None:
        if not isinstance(raw, dict):
            raise TypeError("Malformed Responses payload")
        result.update(raw_response=raw, usage=raw.get("usage", {}), response_status=raw.get("status"),
                      provider_error=raw.get("error"), incomplete_details=raw.get("incomplete_details"))
        if raw.get("status") in {"failed", "cancelled"}:
            result.update(status="CANCELLED" if raw["status"] == "cancelled" else "FAILED",
                          error="Provider response did not complete", content="")
            return
        if not isinstance(raw.get("output"), list):
            raise TypeError("Malformed Responses output")
        parts = []
        for item in raw["output"]:
            if not isinstance(item, dict):
                raise TypeError("Malformed Responses output item")
            if item.get("type") == "message":
                if not isinstance(item.get("content"), list):
                    raise TypeError("Malformed Responses message")
                for block in item["content"]:
                    if not isinstance(block, dict):
                        raise TypeError("Malformed Responses content block")
                    if block.get("type") == "output_text":
                        if not isinstance(block.get("text"), str):
                            raise TypeError("Malformed Responses output text")
                        parts.append(block["text"])
        content = "".join(parts)
        result.update(raw_response=raw, content=content, usage=raw.get("usage", {}),
                      response_status=raw.get("status"), incomplete_details=raw.get("incomplete_details"))
        if raw.get("status") == "incomplete":
            details = raw.get("incomplete_details")
            if isinstance(details, dict) and details.get("reason") == "max_output_tokens":
                result.update(status="TRUNCATED", error="Output token limit reached")
            else:
                result.update(status="INVALID_OUTPUT", error="Incomplete provider response")
        elif raw.get("status") in {"failed", "cancelled"}:
            result.update(status="FAILED", error="Provider response did not complete")
        elif raw.get("status") != "completed" or not content.strip():
            result.update(status="INVALID_OUTPUT", error="No completed textual answer")
        elif any(item.get("type") not in {"message", "reasoning"} for item in raw["output"]):
            result.update(status="INVALID_OUTPUT", error="Unrequested nontextual output")
        else:
            result["status"] = "OK"
