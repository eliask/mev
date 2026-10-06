"""Provider boundary witnesses: credentials, explicit failure and replay."""

import asyncio
import json

import httpx
import pytest

from paa.opencode_client import MODEL, MUSE_MODEL, OpenCodeClient, read_key


def test_literal_key_parser_never_executes_shell(tmp_path):
    path = tmp_path / "key.env"
    path.write_text("export OPENCODE_API_KEY='test-secret-value-xyz'\n")
    assert read_key(path) == "test-secret-value-xyz"
    path.write_text("OPENCODE_API_KEY=$(echo secret)\n")
    with pytest.raises(ValueError, match="literal"):
        read_key(path)


@pytest.mark.asyncio
async def test_public_request_and_cached_receipt_do_not_retain_key(tmp_path):
    secret = "test-secret-value-xyz"
    key = tmp_path / "key.env"
    key.write_text("OPENCODE_API_KEY=" + secret)
    calls = []

    def transport(request):
        calls.append(request)
        if request.url.path.endswith("/models"):
            assert "authorization" not in request.headers
            return httpx.Response(200, json={"data": [{"id": MODEL}]})
        assert request.headers["Authorization"] == "Bearer " + secret
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": "A public-source answer."}}]})

    client = OpenCodeClient(key_file=key, cache_dir=tmp_path / "cache", transport=httpx.MockTransport(transport))
    try:
        with pytest.raises(ValueError, match="public-source"):
            await client.request("task", "system", "user", public_sources=False)
        first = await client.request("task", "system", "user", public_sources=True)
        second = await client.request("task", "system", "user", public_sources=True)
        assert first["status"] == "OK" and second["cache_hit"]
        assert len(calls) == 2
        changed = await client.request("task", "system", "user", public_sources=True, enable_thinking=False)
        assert changed["request_id"] != first["request_id"] and not changed["cache_hit"]
        assert changed["request"]["thinking"] == {"type": "disabled"}
        assert len(calls) == 3
        assert secret not in json.dumps(first)
        assert all(secret not in path.read_text() for path in (tmp_path / "cache").rglob("*.json"))
        with pytest.raises(ValueError, match="Credential"):
            await client.request("task", "system", secret, public_sources=True)
        with pytest.raises(ValueError, match="Credential"):
            await client.request(secret, "system", "public record", public_sources=True)
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_mutated_success_cache_is_not_accepted_as_a_model_answer(tmp_path):
    key = tmp_path / "key.env"
    key.write_text("OPENCODE_API_KEY=test-secret-value-xyz")
    posts = []

    def transport(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": MODEL}]})
        posts.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "retained answer"}, "finish_reason": "stop"}]})

    client = OpenCodeClient(key_file=key, cache_dir=tmp_path / "cache", transport=httpx.MockTransport(transport))
    try:
        first = await client.request("task", "system", "public", public_sources=True)
        path = client.cache_dir / (first["request_id"] + ".json")
        tampered = json.loads(path.read_text())
        tampered["content"] = "unrelated answer"
        path.write_text(json.dumps(tampered))
        second = await client.request("task", "system", "public", public_sources=True)
        assert second["content"] == "retained answer" and len(posts) == 2
        assert second["cache_error"] == "INVALID_CACHE_ARCHIVED"
        assert list((client.cache_dir / "attempts").glob("*.json"))
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_failed_muse_without_output_retains_provider_error(tmp_path):
    key = tmp_path / "key.env"
    secret = "test-secret-value-xyz"
    key.write_text("OPENCODE_API_KEY=" + secret)

    def transport(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": MUSE_MODEL}]})
        return httpx.Response(200, json={"status": "failed", "error": {"message": "failure " + secret}})

    client = OpenCodeClient(key_file=key, cache_dir=tmp_path / "cache", subscription=True,
                            model=MUSE_MODEL, transport=httpx.MockTransport(transport))
    try:
        r = await client.request("task", "system", "public", public_sources=True)
        assert r["status"] == "FAILED" and r["response_status"] == "failed"
        assert r["provider_error"] == {"message": "failure [REDACTED]"}
        assert secret not in json.dumps(r)
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_cancelled_post_retains_receipt_and_propagates_cancellation(tmp_path):
    key = tmp_path / "key.env"
    key.write_text("OPENCODE_API_KEY=test-secret-value-xyz")
    entered = asyncio.Event()

    async def transport(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": MUSE_MODEL}]})
        entered.set()
        await asyncio.Event().wait()

    client = OpenCodeClient(key_file=key, cache_dir=tmp_path / "cache", subscription=True,
                            model=MUSE_MODEL, transport=httpx.MockTransport(transport))
    try:
        task = asyncio.create_task(client.request("task", "system", "public", public_sources=True))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        retained = [json.loads(path.read_text()) for path in client.cache_dir.glob("*.json")]
        assert len(retained) == 1 and retained[0]["status"] == "CANCELLED"
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["discovery", "malformed", "cache"])
async def test_boundary_failures_are_accounted_for_and_corrupt_cache_recoverable(tmp_path, failure):
    key = tmp_path / "key.env"
    key.write_text("OPENCODE_API_KEY=test-secret-value-xyz")

    def transport(request):
        if request.url.path.endswith("/models"):
            if failure == "discovery":
                return httpx.Response(503)
            return httpx.Response(200, json={"data": [{"id": MODEL}]})
        if failure == "malformed":
            return httpx.Response(200, json=[])
        return httpx.Response(200, json={"choices": [{"message": {"content": "public"}, "finish_reason": "stop"}]})

    client = OpenCodeClient(key_file=key, cache_dir=tmp_path / "cache", subscription=True,
                                transport=httpx.MockTransport(transport))
    try:
        result = await client.request("task", "system", "public", public_sources=True)
        if failure == "cache":
            (client.cache_dir / (result["request_id"] + ".json")).write_text("invalid json")
            result = await client.request("task", "system", "public", public_sources=True)
            assert result["status"] == "OK" and result["cache_error"] == "CORRUPT_CACHE_ARCHIVED"
            assert list((client.cache_dir / "attempts").glob("*.json"))
        else:
            assert result["status"] == "FAILED"
        assert list(client.cache_dir.glob("*.json"))
        assert "test-secret-value-xyz" not in json.dumps(result)
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_provider_failure_redacts_echoed_credentials(tmp_path):
    secret = "test-secret-value-xyz"
    key = tmp_path / "key.env"
    key.write_text("OPENCODE_API_KEY=" + secret)

    def transport(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": MODEL}]})
        return httpx.Response(429, text="rate limited " + secret)

    client = OpenCodeClient(key_file=key, cache_dir=tmp_path / "cache", transport=httpx.MockTransport(transport))
    try:
        result = await client.request("task", "system", "public record", public_sources=True)
        assert result["status"] == "FAILED"
        assert secret not in json.dumps(result)
        assert "[REDACTED]" in result["error_body"]
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["completed", "incomplete", "failed"])
async def test_explicit_muse_responses_route_and_statuses(tmp_path, status):
    key = tmp_path / "key.env"
    secret = "test-secret-value-xyz"
    key.write_text("OPENCODE_API_KEY=" + secret)
    calls = []

    def transport(request):
        calls.append(request)
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": MODEL}, {"id": MUSE_MODEL}]})
        body = json.loads(request.content)
        effort = body.pop("reasoning", {}).get("effort")
        assert request.url.path.endswith("/responses")
        assert request.headers["authorization"] == "Bearer " + secret
        assert body == {"model": MUSE_MODEL, "instructions": "system", "input": "public",
                        "max_output_tokens": 4096, "temperature": 0, "store": False}
        return httpx.Response(200, json={"status": status,
            "reasoning": {"effort": effort or "high"},
            "incomplete_details": {"reason": "max_output_tokens"} if status == "incomplete" else None,
            "output": [{"type": "reasoning", "summary": []},
                       {"type": "message", "content": [{"type": "output_text", "text": "source-backed "},
                                                        {"type": "output_text", "text": "candidate"}]}],
            "usage": {"input_tokens": 20, "output_tokens": 10}})

    with pytest.raises(ValueError, match="subscription"):
        OpenCodeClient(key_file=key, cache_dir=tmp_path / "wrong-route", model=MUSE_MODEL)
    with pytest.raises(ValueError, match="Unsupported"):
        OpenCodeClient(key_file=key, cache_dir=tmp_path / "wrong-model", model="other-model")
    client = OpenCodeClient(key_file=key, cache_dir=tmp_path / "cache", subscription=True,
                            model=MUSE_MODEL, transport=httpx.MockTransport(transport))
    try:
        with pytest.raises(ValueError, match="Responses parameter"):
            await client.request("task", "system", "public", public_sources=True, enable_thinking=False)
        result = await client.request("task", "system", "public", public_sources=True)
        assert result["status"] == {"completed": "OK", "incomplete": "TRUNCATED", "failed": "FAILED"}[status]
        assert result["content"] == ("" if status == "failed" else "source-backed candidate")
        assert result["model"]["model_id"] == MUSE_MODEL and not result["model"]["free_model"]
        assert secret not in json.dumps(result)
        if status == "completed":
            again = await client.request("task", "system", "public", public_sources=True)
            assert again["cache_hit"] and len(calls) == 2
            low = await client.request("task", "system", "public", public_sources=True, reasoning_effort="low")
            assert low["request_id"] != result["request_id"] and not low["cache_hit"]
            assert low["request"]["reasoning"] == {"effort": "low"}
            assert low["observed_reasoning_effort"] == "low"
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('corruption', [None, 'reasoning'])
async def test_gateway_creation_timestamp_is_observation_and_cache_revalidates_reasoning(tmp_path, corruption):
    from paa.llm_client import digest

    key = tmp_path / 'key.env'
    key.write_text('OPENCODE_API_KEY=test-secret-value-xyz')
    discoveries, posts = 0, 0

    def transport(request):
        nonlocal discoveries, posts
        if request.url.path.endswith('/models'):
            discoveries += 1
            return httpx.Response(200, json={'data': [{'id': MUSE_MODEL, 'created': discoveries}]})
        posts += 1
        return httpx.Response(200, json={'status': 'completed', 'reasoning': {'effort': 'low'},
            'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': 'candidate'}]}]})

    async def attempt():
        client = OpenCodeClient(key_file=key, cache_dir=tmp_path / 'cache', subscription=True,
                                model=MUSE_MODEL, transport=httpx.MockTransport(transport))
        try:
            return await client.request('task', 'system', 'public', public_sources=True, reasoning_effort='low')
        finally:
            await client.close()

    first = await attempt()
    if corruption:
        path = tmp_path / 'cache' / (first['request_id'] + '.json')
        changed = json.loads(path.read_text())
        changed['raw_response']['reasoning']['effort'] = 'high'
        changed['response_sha256'] = digest(changed['raw_response'])
        path.write_text(json.dumps(changed))
    second = await attempt()
    assert first['request_id'] == second['request_id'] and discoveries == 2
    assert second['status'] == 'OK'
    assert posts == (2 if corruption else 1)
    assert second['cache_hit'] is (corruption is None)
    if corruption:
        assert second['cache_error'] == 'INVALID_CACHE_ARCHIVED'
        assert list((tmp_path / 'cache/attempts').glob('*.json'))


def test_provider_cancellation_is_not_a_failed_or_negative_answer():
    result = {}
    OpenCodeClient._read_responses({'status': 'cancelled', 'error': {'message': 'cancelled'}}, result)
    assert result['status'] == 'CANCELLED' and result['content'] == ''
