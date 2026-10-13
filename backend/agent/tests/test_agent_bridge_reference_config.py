from __future__ import annotations

import base64
import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_agent_bridge_module():
    module_path = Path(__file__).resolve().parents[2] / "services" / "agent_bridge.py"
    spec = importlib.util.spec_from_file_location("agent_bridge_for_tests", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_broadcast_tolerates_client_disconnect_during_send() -> None:
    agent_bridge = _load_agent_bridge_module()
    state = SimpleNamespace(clients=set())
    received = []

    class DisconnectingClient:
        async def send(self, payload):
            received.append(payload)
            state.clients.remove(self)
            await asyncio.sleep(0)

    state.clients.add(DisconnectingClient())
    asyncio.run(agent_bridge.broadcast(state, [{"type": "test"}]))

    assert len(received) == 1
    assert not state.clients


def test_generate_config_from_template_persists_reference_papers(tmp_path: Path) -> None:
    agent_bridge = _load_agent_bridge_module()
    state = SimpleNamespace(
        runs_base_dir=str(tmp_path / "runs"),
        agent_package_dir=str(tmp_path / "agent"),
    )

    config_path = agent_bridge._generate_config_from_template(
        state,
        project_id="proj-ref-test",
        topic="test topic",
        reference_papers=[
            "1234.56789",
            "/data/papers/local_reference.pdf",
        ],
    )

    content = Path(config_path).read_text(encoding="utf-8")
    assert 'reference_papers:\n    - "1234.56789"\n    - "/data/papers/local_reference.pdf"' in content
    assert "__REFERENCE_PAPERS__" not in content


def test_local_template_is_selected_without_modifying_repository_template(tmp_path, monkeypatch):
    agent_bridge = _load_agent_bridge_module()
    template = tmp_path / "config_template.local.yaml"
    template.write_text('project:\n  name: __PROJECT_ID__\nresearch:\n  topic: "__TOPIC__"\n  reference_papers: __REFERENCE_PAPERS__\nllm:\n  api_key_env: RESEARCHCLAW_API_KEY\n', encoding="utf-8")
    original = template.read_bytes()
    monkeypatch.setenv("CLAW_CONFIG_TEMPLATE", str(template))
    state = SimpleNamespace(runs_base_dir=str(tmp_path / "runs"), agent_package_dir=str(tmp_path / "agent"))
    output = agent_bridge._generate_config_from_template(state, "local-test", "test topic")
    assert "api_key_env: RESEARCHCLAW_API_KEY" in Path(output).read_text()
    assert template.read_bytes() == original


def test_missing_explicit_local_template_fails_instead_of_using_wrong_provider(tmp_path, monkeypatch):
    import pytest

    agent_bridge = _load_agent_bridge_module()
    monkeypatch.setenv("CLAW_CONFIG_TEMPLATE", str(tmp_path / "missing.yaml"))
    state = SimpleNamespace(runs_base_dir=str(tmp_path / "runs"), agent_package_dir=str(tmp_path / "agent"))
    with pytest.raises(FileNotFoundError, match="Configured local template"):
        agent_bridge._generate_config_from_template(state, "local-test", "test topic")


def test_persist_reference_uploads_saves_pdf_files(tmp_path: Path) -> None:
    agent_bridge = _load_agent_bridge_module()
    project_dir = tmp_path / "project"

    saved_paths = agent_bridge._persist_reference_uploads(
        project_dir,
        [
            {
                "name": "paper one.pdf",
                "contentBase64": base64.b64encode(b"%PDF-1.4 sample").decode("ascii"),
            }
        ],
    )

    assert len(saved_paths) == 1
    saved_path = Path(saved_paths[0])
    assert saved_path.exists()
    assert saved_path.suffix == ".pdf"
    assert saved_path.read_bytes() == b"%PDF-1.4 sample"
