"""Streaming chat-completions helper for tool-calling turn loops.

Long non-streaming requests (e.g. an agent writing a whole ``main.py`` in one
turn) can exceed proxy/CDN idle limits and fail with HTTP 524/502. Streaming
keeps bytes flowing on the connection. This helper sends ``stream: true`` and
reassembles the SSE chunks — including ``tool_calls`` deltas — into the same
dict shape as a non-streaming response, so callers need no changes.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any


def stream_chat_completion(
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    """POST a chat-completions request with streaming and return a
    non-streaming-shaped response dict.

    Falls back to plain JSON parsing when the server ignores ``stream``.
    """
    payload = json.dumps({**body, "stream": True}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers=headers)

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        content_type = resp.headers.get("Content-Type", "")
        if "text/event-stream" not in content_type:
            return json.loads(resp.read().decode("utf-8"))

        content: list[str] = []
        reasoning: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        finish_reason = None
        model_name = ""
        usage: dict[str, Any] = {}

        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except (json.JSONDecodeError, ValueError):
                continue
            if "error" in event:
                raise RuntimeError(f"Streaming API error: {event['error']}")
            model_name = model_name or event.get("model", "")
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    content.append(delta["content"])
                if delta.get("reasoning_content"):
                    reasoning.append(delta["reasoning_content"])
                for tc in delta.get("tool_calls") or []:
                    slot = tool_calls.setdefault(
                        tc.get("index", len(tool_calls)),
                        {"id": "", "type": "function",
                         "function": {"name": "", "arguments": ""}},
                    )
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["function"]["name"] += fn["name"]
                    if fn.get("arguments"):
                        slot["function"]["arguments"] += fn["arguments"]
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]

    message: dict[str, Any] = {"role": "assistant", "content": "".join(content)}
    if reasoning:
        message["reasoning_content"] = "".join(reasoning)
    if tool_calls:
        message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
    return {
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "model": model_name,
        "usage": usage,
    }
