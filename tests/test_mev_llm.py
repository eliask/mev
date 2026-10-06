"""Source structure and model identity must survive the legacy MeV adapter."""

import asyncio

import pytest

from mev import llm


class MemoryCache(dict):
    def set(self, key, value, expire=None):
        self[key] = value


class Response:
    def __init__(self, body, status=200):
        self.body, self.status = body, status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError("HTTP failure")

    async def json(self, **kwargs):
        return self.body


class Session:
    def __init__(self, model="model-a", fail=False):
        self.model, self.fail = model, fail
        self.requests = []

    def get(self, url, **kwargs):
        if url.endswith("/v1/models"):
            return Response({"data": [{"id": self.model, "meta": {"n_params": 27}}]})
        return Response({"model_path": self.model, "chat_template": "test template", "model_ftype": "Q4"})

    def post(self, url, *, json, **kwargs):
        self.requests.append(json)
        if json["max_tokens"] != 1 and self.fail:
            return Response({"error": "test failure"}, 500)
        return Response({"system_fingerprint": "test-build", "choices": [
            {"message": {"content": "source answer"}, "finish_reason": "stop"}], "usage": {}})


def test_model_bound_cache_preserves_source_whitespace(monkeypatch):
    monkeypatch.setattr(llm, "_cache", MemoryCache())

    async def run():
        first = Session()
        source = "  Heading\n\n1. First\n2. Second  "
        await llm.call_llm(first, "read source", source, max_tokens=20)
        await llm.call_llm(first, "read source", source, max_tokens=20)
        assert len(first.requests) == 2  # discovery probe plus one live source call
        assert first.requests[-1]["messages"][1]["content"] == source
        await llm.call_llm(first, "read source", " ".join(source.split()), max_tokens=20)
        assert len(first.requests) == 3
        second = Session(model="model-b")
        await llm.call_llm(second, "read source", source, max_tokens=20)
        assert len(second.requests) == 2
        assert second.requests[-1]["model"] == "model-b"

    asyncio.run(run())


def test_exhausted_failure_cannot_become_an_empty_source_result(monkeypatch):
    monkeypatch.setattr(llm, "_cache", MemoryCache())
    monkeypatch.setattr(llm, "MAX_RETRIES", 0)
    with pytest.raises(llm.LLMRequestFailed, match="inference failed"):
        asyncio.run(llm.call_llm(Session(fail=True), "read", "original source", max_tokens=20))
