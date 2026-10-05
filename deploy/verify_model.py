"""Check the configured OpenAI-compatible model without printing credentials."""

import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml


root = Path(__file__).resolve().parent.parent
with (root / "examples/config_template.yaml").open(encoding="utf-8") as handle:
    llm = yaml.safe_load(handle)["llm"]

base_url = llm["base_url"].rstrip("/")
model = llm["primary_model"]
key = os.environ.get(llm["api_key_env"], "")
if not key:
    raise SystemExit("API key environment variable is empty")

headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
for path, payload in (
    ("models", None),
    ("chat/completions", {"model": model, "messages": [{"role": "user", "content": "Reply OK"}], "max_tokens": 8}),
):
    body = json.dumps(payload).encode() if payload is not None else None
    request = Request(f"{base_url}/{path}", data=body, headers=headers)
    try:
        with urlopen(request, timeout=45) as response:
            result = json.load(response)
        if path == "models":
            ids = [item.get("id", "") for item in result.get("data", [])]
            print(f"models: HTTP 200; {len(ids)} listed; configured model listed: {model in ids}")
            if model not in ids:
                print("available models: " + ", ".join(ids))
        else:
            print(f"chat/completions: HTTP 200; model: {result.get('model', 'unknown')}")
    except HTTPError as exc:
        print(f"{path}: HTTP {exc.code}")
    except (URLError, TimeoutError, ValueError) as exc:
        print(f"{path}: {type(exc).__name__}: {exc}")
