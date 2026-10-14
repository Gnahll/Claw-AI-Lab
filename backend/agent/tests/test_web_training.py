import asyncio
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from researchclaw.data.coverage import audit_coverage, propose_splits
from researchclaw.training import jobs
from researchclaw.training.web import WebTrainingService, allowed_training_origin

Image = pytest.importorskip("PIL.Image")


@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.delenv("CLAW_TRAINING_SPLIT_MANIFEST", raising=False)
    monkeypatch.delenv("CLAW_WEB_TRAINING_DISABLED", raising=False)
    source = tmp_path / "code"
    for directory in ("models", "common"):
        (source / directory).mkdir(parents=True)
        (source / directory / "fixture.py").write_text("VALUE = 1\n")
    (source / "cotransformer.py").write_text("MODEL = 'fixture, not actual training'\n")
    dataset = tmp_path / "data"
    (dataset / "image").mkdir(parents=True)
    (dataset / "mask").mkdir()
    for identifier in range(1, 11):
        Image.new("RGB", (8, 8), (identifier, 30, 40)).save(dataset / "image" / f"{identifier}t.tif")
        Image.new("RGB", (8, 8), (identifier, 50, 60)).save(dataset / "image" / f"{identifier}.tif")
        mask = Image.new("L", (8, 8), 0)
        mask.putpixel((1, 1), 255)
        mask.save(dataset / "mask" / f"{identifier}forged.tif")
    proposal = tmp_path / "proposal.json"
    jobs.write_json(proposal, propose_splits(audit_coverage(dataset)))
    checkpoint = tmp_path / "checkpoints/initial.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"fixture checkpoint, never executed")
    examples = tmp_path / "examples"
    examples.mkdir()
    jobs.write_json(examples / "cotransformers-pilot.json", {
        "source_dir": str(source), "dataset_root": str(dataset), "split_manifest": str(proposal),
        "initial_checkpoint": str(checkpoint), "initial_checkpoint_sha256": jobs.file_sha256(checkpoint),
        "epochs": 1, "steps_per_epoch": 3, "learning_rate": 1e-5, "weight_decay": .05,
        "seed": 42, "checkpoint_every_steps": 1,
    })
    jobs.write_json(examples / "cotransformers-web.json", {"split_manifest": str(proposal), "topic": "Fixture acceptance"})
    return WebTrainingService(tmp_path / "runs", tmp_path)


def prepare(service):
    response = service.handle({"command": "training_preflight"})["payload"]
    assert response["ok"], response
    identifier = response["preflight"]["id"]
    response = service.handle({"command": "training_prepare", "preflightId": identifier, "protocolApproved": True})["payload"]
    assert response["ok"], response
    return identifier, service._directory(identifier)


def test_check_and_prepare_never_start_gpu_and_preserve_originals(service, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Read-only preflight/preparation attempted GPU launch")

    monkeypatch.setattr(jobs, "start_job", forbidden)
    monkeypatch.setattr(jobs, "gpu_rows", forbidden)
    proposal = Path(service.defaults["split_manifest"])
    before = proposal.read_bytes()
    source_hashes = jobs.source_hashes(Path(service.defaults["source_dir"]))
    identifier, job_dir = prepare(service)
    assert proposal.read_bytes() == before
    assert jobs.read_json(proposal)["protocol_status"] == "proposal_only"
    assert jobs.read_json(job_dir / "splits.json")["protocol_status"] == "approved"
    assert jobs.job_status(job_dir)["status"] == "prepared"
    assert jobs.source_hashes(Path(service.defaults["source_dir"])) == source_hashes
    assert jobs.source_hashes(job_dir / "source") == source_hashes
    again = service.handle({"command": "training_prepare", "preflightId": identifier, "protocolApproved": True})["payload"]
    assert again["last_job_id"] == identifier
    assert len(again["jobs"]) == 1


@pytest.mark.parametrize("approval", [None, False, "true", 1])
def test_preparation_requires_explicit_protocol_approval(service, approval):
    check = service.handle({"command": "training_preflight"})["payload"]["preflight"]
    response = service.handle({"command": "training_prepare", "preflightId": check["id"], "protocolApproved": approval})["payload"]
    assert not response["ok"]
    assert "Explicit approval" in response["error"]
    assert not response["jobs"]


@pytest.mark.parametrize("spec", [{"epochs": 2}, {"steps_per_epoch": 4}, {"steps_per_epoch": True},
                                  {"shell": "anything"}, {"source_dir": "."}])
def test_web_budget_and_spec_fail_closed(service, spec):
    response = service.handle({"command": "training_preflight", "spec": spec})["payload"]
    assert not response["ok"]
    assert "preflight" not in response


def test_resources_outside_configured_scope_are_rejected(service, tmp_path):
    outside = tmp_path / "untrusted-code"
    outside.mkdir()
    response = service.handle({"command": "training_preflight", "spec": {"source_dir": str(outside)}})["payload"]
    assert not response["ok"]
    assert "outside" in response["error"]


@pytest.mark.parametrize("changed", ["manifest", "source", "dataset"])
def test_inputs_changed_after_review_require_new_check(service, changed):
    check = service.handle({"command": "training_preflight"})["payload"]["preflight"]
    if changed == "manifest":
        with Path(service.defaults["split_manifest"]).open("a") as stream:
            stream.write("\n")
    elif changed == "source":
        (Path(service.defaults["source_dir"]) / "cotransformer.py").write_text("CHANGED = True\n")
    else:
        Image.new("RGB", (8, 8), (99, 99, 99)).save(Path(service.defaults["dataset_root"]) / "image/1t.tif")
    response = service.handle({"command": "training_prepare", "preflightId": check["id"], "protocolApproved": True})["payload"]
    assert not response["ok"]
    assert not response["jobs"]


def test_start_requires_separate_gpu_allocation_and_forwards_uuid(service, monkeypatch):
    identifier, job_dir = prepare(service)
    calls = []

    def launch(path, gpu_uuid, *, resume=False):
        calls.append((path, gpu_uuid, resume))
        jobs.write_json(path / "state.json", {"status": "running", "time": time.time()})

    monkeypatch.setattr(jobs, "start_job", launch)
    request = {"command": "training_start", "jobId": identifier, "gpuUuid": "GPU-12345678-1234-1234-1234-123456789abc"}
    assert not service.handle(request)["payload"]["ok"]
    assert not calls
    assert service.handle({**request, "gpuAllocated": True})["payload"]["ok"]
    assert calls == [(job_dir.resolve(), request["gpuUuid"], False)]
    assert jobs.read_json(job_dir / "web_gpu_approval.json")["gpu_uuid"] == request["gpuUuid"]


def test_busy_gpu_is_a_visible_error_not_a_success(service, monkeypatch):
    identifier, _ = prepare(service)

    def launch(*args, **kwargs):
        raise RuntimeError("Allocated GPU is occupied; no training was started")

    monkeypatch.setattr(jobs, "start_job", launch)
    response = service.handle({"command": "training_start", "jobId": identifier,
                               "gpuUuid": "GPU-12345678-1234-1234-1234-123456789abc", "gpuAllocated": True})["payload"]
    assert not response["ok"]
    assert "occupied" in response["error"]
    assert response["jobs"][0]["state"]["status"] == "prepared"


def test_server_can_disable_gpu_start(service, monkeypatch):
    identifier, _ = prepare(service)
    monkeypatch.setenv("CLAW_WEB_TRAINING_DISABLED", "1")
    response = service.handle({"command": "training_start", "jobId": identifier, "gpuAllocated": True})["payload"]
    assert not response["ok"]
    assert "disabled" in response["error"]


def test_restart_recovers_prepared_job_without_launching(service):
    identifier, _ = prepare(service)
    recovered = WebTrainingService(service.root.parent, Path(service.defaults["source_dir"]).parent)
    response = recovered.handle({"command": "training_list"})["payload"]
    assert response["jobs"][0]["id"] == identifier
    assert response["jobs"][0]["state"]["status"] == "prepared"


def test_early_verification_cannot_prevent_automatic_completion_check(service):
    identifier, job_dir = prepare(service)
    response = service.handle({"command": "training_verify", "jobId": identifier})["payload"]
    assert not response["ok"]
    assert not (job_dir / "web_verification.json").exists()


def complete_fixture(service):
    pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    identifier, job_dir = prepare(service)
    manifest = jobs.read_json(job_dir / "splits.json")
    checkpoint = job_dir / "fixture-checkpoint.pt"
    checkpoint.write_bytes(b"synthetic verifier fixture, not a trained model")
    digest = jobs.file_sha256(checkpoint)
    predictions = job_dir / "test_predictions"
    predictions.mkdir()
    records = []
    for sample in manifest["splits"]["test"]:
        record = {"id": sample["id"], "pixel_f1": 1., "pixel_iou": 1.}
        with Image.open(Path(service.defaults["dataset_root"]) / sample["mask"]) as image:
            target = image.convert("L").resize((512, 512), Image.Resampling.NEAREST)
        for kind in ("target", "prediction"):
            filename = f"{sample['id']}-{kind}.png"
            target.save(predictions / filename)
            record[kind] = filename
            record[f"{kind}_sha256"] = jobs.file_sha256(predictions / filename)
        records.append(record)
    proof = {"optimizer_steps": 3, "initial_parameter_sha256": "a" * 64, "trained_parameter_sha256": "b" * 64,
             "gpu_evidence": {"gpu_uuid": "GPU-fixture", "memory_mib": 1, "pid": 123},
             "selected_checkpoint": {"filename": checkpoint.name, "sha256": digest}}
    results = {"metrics": {"pixel_f1": 1., "pixel_iou": 1.}, "records": records,
               "training_evidence": proof, "checkpoint_sha256": digest, "threshold": .5,
               "aggregation": "macro_per_image", "protocol": "fixture", "test_is_blind": False}
    jobs.write_json(job_dir / "test_results.json", results)
    events = [{"event": "optimizer_step", "global_step": step, "sample_id": manifest["splits"]["train"][0]["id"], "loss": .1}
              for step in range(1, 4)]
    (job_dir / "events.jsonl").write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")
    jobs.write_json(job_dir / "state.json", {"status": "completed", "time": time.time(), "gpu_uuid": "GPU-fixture",
                                           "test_results_sha256": jobs.file_sha256(job_dir / "test_results.json")})
    return identifier, job_dir


def test_completion_auto_verifies_and_imports_s14_s15(service):
    identifier, job_dir = complete_fixture(service)
    response = service.handle({"command": "training_list"})["payload"]
    job = response["jobs"][0]
    assert job["verification"]["status"] == "passed"
    assert job["verification"]["result"]["metrics"] == {"pixel_f1": 1., "pixel_iou": 1.}
    assert jobs.read_json(job_dir / "pipeline/stage-14/runs/run_report.json")["has_results"]
    assert (job_dir / "pipeline/stage-15/refinement_disabled.json").is_file()
    assert service.artifact(identifier, "web_verification.json").is_file()
    assert service.artifact(identifier, f"test_predictions/{job['predictions'][0]['prediction']}").is_file()


def test_tampering_invalidates_previous_verification(service):
    identifier, job_dir = complete_fixture(service)
    assert service.handle({"command": "training_verify", "jobId": identifier})["payload"]["ok"]
    checkpoint = jobs.read_json(job_dir / "test_results.json")["training_evidence"]["selected_checkpoint"]["filename"]
    (job_dir / checkpoint).write_bytes(b"tampered")
    response = service.handle({"command": "training_verify", "jobId": identifier})["payload"]
    assert not response["ok"]
    assert response["jobs"][0]["verification"]["status"] == "failed"
    assert not response["jobs"][0]["predictions"]
    assert jobs.read_json(job_dir / "pipeline/stage-14/runs/run_report.json")["has_results"] is False
    with pytest.raises(ValueError, match="passed evidence"):
        service.artifact(identifier, "pipeline/stage-14/runs/run_report.json")


@pytest.mark.parametrize("identifier,filename", [("../outside", "state.json"), ("a" * 32, "../../.env"),
                                                ("a" * 32, "spec.json"), ("a" * 32, "source/cotransformer.py")])
def test_downloads_are_scoped_to_public_evidence(service, identifier, filename):
    with pytest.raises((OSError, ValueError)):
        service.artifact(identifier, filename)


@pytest.mark.parametrize("origin,allowed", [("http://127.0.0.1:5913", True), ("http://localhost:5913", True),
                                         ("http://evil.example", False), ("http://localhost.evil.example", False),
                                         ("http://localhost@evil.example", False), ("null", False), ("", False)])
def test_training_rejects_cross_site_websocket_origins(origin, allowed, monkeypatch):
    monkeypatch.delenv("CLAW_TRAINING_ALLOWED_ORIGINS", raising=False)
    assert allowed_training_origin(origin) is allowed


def test_bridge_dispatches_managed_commands_off_event_loop(service, monkeypatch):
    pytest.importorskip("websockets")
    from .test_agent_bridge_reference_config import _load_agent_bridge_module

    bridge = _load_agent_bridge_module()
    state = SimpleNamespace(web_training=service)
    original = service.handle
    owner = threading.get_ident()

    def dispatch(data):
        assert threading.get_ident() != owner
        return original(data)

    monkeypatch.setattr(service, "handle", dispatch)
    response = asyncio.run(bridge.handle_command(state, {"command": "training_list", "requestId": "check"}))
    assert response[0]["type"] == "training_update"
    assert response[0]["payload"]["request_id"] == "check"


def test_browser_socket_can_check_prepare_and_recover_without_gpu_use(service, monkeypatch):
    websockets = pytest.importorskip("websockets")
    from .test_agent_bridge_reference_config import _load_agent_bridge_module

    bridge = _load_agent_bridge_module()
    monkeypatch.setattr(bridge, "_scan_existing_artifacts", lambda state: [])

    def forbidden(*args, **kwargs):
        pytest.fail("Socket acceptance launched a GPU without allocation")

    monkeypatch.setattr(jobs, "start_job", forbidden)
    state = SimpleNamespace(web_training=service, clients=set(), agents={}, queues={},
                            agent_package_dir=str(Path(__file__).resolve().parents[1]))

    async def exercise():
        async with websockets.serve(lambda socket: bridge.ws_handler(state, socket), "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            async with websockets.connect(f"ws://127.0.0.1:{port}", origin="http://127.0.0.1:5913") as socket:
                await socket.recv()
                await socket.send(json.dumps({"command": "training_preflight"}))
                checked = json.loads(await socket.recv())["payload"]
                assert checked["ok"]
                await socket.send(json.dumps({"command": "training_prepare", "preflightId": checked["preflight"]["id"],
                                              "protocolApproved": True}))
                prepared = json.loads(await socket.recv())["payload"]
                assert prepared["ok"]
                assert prepared["jobs"][0]["state"]["status"] == "prepared"
                await socket.send(json.dumps({"command": "training_start", "jobId": prepared["last_job_id"]}))
                assert json.loads(await socket.recv())["payload"]["ok"] is False
            async with websockets.connect(f"ws://127.0.0.1:{port}", origin="http://127.0.0.1:5913") as socket:
                await socket.recv()
                await socket.send(json.dumps({"command": "training_list"}))
                recovered = json.loads(await socket.recv())["payload"]
                assert recovered["jobs"][0]["id"] == prepared["last_job_id"]

    asyncio.run(exercise())


def test_foreign_origin_cannot_reach_training_dispatch(service, monkeypatch):
    pytest.importorskip("websockets")
    from .test_agent_bridge_reference_config import _load_agent_bridge_module

    bridge = _load_agent_bridge_module()

    class Client:
        request = SimpleNamespace(headers={"Origin": "https://evil.example"})
        sent = []

        async def send(self, payload):
            self.sent.append(json.loads(payload))

        def __aiter__(self):
            return self.messages()

        async def messages(self):
            yield json.dumps({"command": "training_start", "requestId": "rejected"})

    def forbidden(*args, **kwargs):
        pytest.fail("Cross-site origin reached the training service")

    monkeypatch.setattr(bridge, "handle_command", forbidden)
    monkeypatch.setattr(bridge, "_scan_existing_artifacts", lambda state: [])
    client = Client()
    state = SimpleNamespace(clients=set(), agents={}, queues={}, agent_package_dir=str(Path(__file__).resolve().parents[1]))
    asyncio.run(bridge.ws_handler(state, client))
    rejected = [message for message in client.sent if message["type"] == "training_update"]
    assert rejected[0]["payload"]["ok"] is False
