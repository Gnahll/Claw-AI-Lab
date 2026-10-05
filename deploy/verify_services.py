"""Smoke-test this instance's HTTP and WebSocket endpoints without starting jobs."""

import asyncio
import json
import os
from urllib.request import urlopen

import websockets


async def main():
    frontend = os.environ.get("FRONTEND_PORT", "5903")
    with urlopen(f"http://127.0.0.1:{frontend}/", timeout=10) as response:
        assert response.status == 200
        assert b"<html" in response.read().lower()
    print(f"frontend HTTP {frontend}: OK")

    endpoints = (
        ("resource monitor", f"ws://127.0.0.1:{os.environ.get('RESOURCE_MONITOR_PORT', '8905')}/"),
        ("agent bridge", f"ws://127.0.0.1:{os.environ.get('AGENT_BRIDGE_PORT', '8906')}/"),
        ("resource proxy", f"ws://127.0.0.1:{frontend}/ws/resources"),
        ("agent proxy", f"ws://127.0.0.1:{frontend}/ws/agents"),
    )
    for name, url in endpoints:
        async with websockets.connect(url, open_timeout=10) as connection:
            payload = json.loads(await asyncio.wait_for(connection.recv(), timeout=10))
            assert isinstance(payload, dict)
        print(f"{name}: OK")


if __name__ == "__main__":
    asyncio.run(main())
