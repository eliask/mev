import asyncio
import json

import httpx

from paa.llm_client import LocalLLMClient


def client_with_transport(tmp_path, handler):
    client = LocalLLMClient("http://127.0.0.1:65533", cache_dir=tmp_path, retries=0)
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client.manifest = {"model_id": "test-local-model", "runtime": "test"}
    return client


def response(content='{"value":"source"}', finish="stop"):
    return httpx.Response(200, json={
        "choices": [{"finish_reason": finish, "message": {"content": content}}],
        "usage": {"completion_tokens": 5}, "system_fingerprint": "runtime-test",
    })


SCHEMA = {
    "type": "object", "properties": {"value": {"type": "string"}},
    "required": ["value"], "additionalProperties": False,
}


def test_cache_identity_covers_inputs_prompt_schema_model_and_seed(tmp_path):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return response()

    async def run():
        client = client_with_transport(tmp_path, handler)
        first = await client.request("extract", "read source", "text", schema=SCHEMA)
        cached = await client.request("extract", "read source", "text", schema=SCHEMA)
        assert first["status"] == "OK"
        assert cached["cache_hit"] is True
        assert len(calls) == 1
        await client.request("extract", "read source v2", "text", schema=SCHEMA)
        await client.request("extract", "read source", "different", schema=SCHEMA)
        await client.request("extract", "read source", "text", schema=SCHEMA, seed=43)
        client.manifest["runtime"] = "changed"
        await client.request("extract", "read source", "text", schema=SCHEMA)
        assert len(calls) == 5
        await client.request("extract", "read source", "text", schema=SCHEMA, enable_thinking=True)
        assert len(calls) == 6
        assert calls[-1]["chat_template_kwargs"] == {"enable_thinking": True}
        assert all("tools" not in payload for payload in calls)
        await client.close()

    asyncio.run(run())


def test_bad_json_and_truncation_remain_visible_failures(tmp_path):
    async def run():
        client = client_with_transport(tmp_path, lambda request: response('{"value":'))
        broken = await client.request("extract", "read source", "text", schema=SCHEMA)
        assert broken["status"] == "INVALID_OUTPUT"
        assert "parsed" not in broken
        await client.close()
        truncated = client_with_transport(tmp_path, lambda request: response(finish="length"))
        result = await truncated.request("extract", "read source", "text", schema=SCHEMA)
        assert result["status"] == "TRUNCATED"
        assert result["raw_response"]
        await truncated.close()

    asyncio.run(run())


def test_local_client_does_not_silently_use_a_remote_endpoint(tmp_path):
    try:
        LocalLLMClient("https://example.com", cache_dir=tmp_path)
    except ValueError as error:
        assert "loopback" in str(error)
    else:
        raise AssertionError("remote endpoint accepted")


def test_distinct_clients_share_backpressure_and_release_cancelled_slot(tmp_path):
    async def run():
        active = peak = 0

        async def handler(request):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.04)
            active -= 1
            return response()

        clients = [client_with_transport(tmp_path / str(i), handler) for i in range(2)]
        for client in clients:
            client.manifest["total_slots"] = 1
        results = await asyncio.gather(*(client.request("test", "source", str(i), schema=SCHEMA)
                                         for i, client in enumerate(clients)))
        assert peak == 1
        assert all(result["status"] == "OK" for result in results)
        assert any(result["shared_slot_wait_seconds"] >= 0.05 for result in results)
        entered = asyncio.Event()

        async def cancelled_holder():
            async with clients[0].inference_slot():
                entered.set()
                await asyncio.Event().wait()

        holder = asyncio.create_task(cancelled_holder())
        await entered.wait()
        holder.cancel()
        try:
            await holder
        except asyncio.CancelledError:
            pass
        async with asyncio.timeout(1):
            async with clients[1].inference_slot():
                pass
        for client in clients:
            await client.close()

    asyncio.run(run())
