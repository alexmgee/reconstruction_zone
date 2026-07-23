"""Validate real WEB7 API responses against the authoritative JSON Schemas."""

from __future__ import annotations

import copy
import hashlib
import http.client
import json
import re
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator, ValidationError
from referencing import Registry, Resource

from reconstruction_web.jobs import JobRegistry
from reconstruction_web.server import make_server, shutdown_server
from reconstruction_web.state import build_state_config


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONTRACTS_DIR = REPOSITORY_ROOT / "frontend" / "contracts"
GENERATED_TYPES_PATH = REPOSITORY_ROOT / "frontend" / "src" / "api" / "generated.ts"
SCHEMA_NAMES = (
    "common.schema.json",
    "error-response.schema.json",
    "health-response.schema.json",
    "job-detail-response.schema.json",
    "jobs-response.schema.json",
    "version-response.schema.json",
)
TERMINAL_STATES = {"completed", "failed", "cancelled"}


def _load_schemas() -> dict[str, dict[str, Any]]:
    discovered = tuple(sorted(path.name for path in CONTRACTS_DIR.glob("*.schema.json")))
    assert discovered == SCHEMA_NAMES
    schemas = {
        name: json.loads((CONTRACTS_DIR / name).read_text(encoding="utf-8"))
        for name in SCHEMA_NAMES
    }
    for schema in schemas.values():
        Draft202012Validator.check_schema(schema)
    return schemas


@pytest.fixture(scope="module")
def schemas() -> dict[str, dict[str, Any]]:
    return _load_schemas()


@pytest.fixture(scope="module")
def validators(schemas: dict[str, dict[str, Any]]) -> dict[str, Draft202012Validator]:
    def reject_remote_retrieval(uri: str) -> Resource:
        raise AssertionError(f"schema attempted remote retrieval: {uri}")

    registry = Registry(retrieve=reject_remote_retrieval).with_resources(
        (schema["$id"], Resource.from_contents(schema)) for schema in schemas.values()
    )
    return {
        name: Draft202012Validator(schema, registry=registry)
        for name, schema in schemas.items()
        if name != "common.schema.json"
    }


@pytest.fixture
def contract_server(tmp_path: Path):
    state_root = tmp_path / "web_state"
    state_root.mkdir()
    registry = JobRegistry(_step_wait=0)
    server = make_server(build_state_config(state_root), port=0, job_registry=registry)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, registry
    finally:
        shutdown_server(server, thread, join_timeout=3)


def _request_json(server: Any, target: str) -> tuple[int, dict[str, Any]]:
    host, port = server.server_address
    connection = http.client.HTTPConnection(host, port, timeout=5)
    try:
        connection.request("GET", target)
        response = connection.getresponse()
        body = response.read()
        assert response.getheader("Content-Type") == "application/json; charset=utf-8"
        return response.status, json.loads(body)
    finally:
        connection.close()


def _wait_for_terminal(registry: JobRegistry, job_id: str) -> dict[str, object]:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        snapshot = registry.get_snapshot(job_id)
        if snapshot["state"] in TERMINAL_STATES:
            return snapshot
        time.sleep(0.01)
    pytest.fail(f"job {job_id} did not reach a terminal state")


def test_real_server_responses_match_contracts(contract_server, validators) -> None:
    server, registry = contract_server

    status, health = _request_json(server, "/api/health")
    assert status == 200
    validators["health-response.schema.json"].validate(health)

    status, version = _request_json(server, "/api/version")
    assert status == 200
    validators["version-response.schema.json"].validate(version)

    status, empty_jobs = _request_json(server, "/api/jobs")
    assert status == 200
    assert empty_jobs == {"jobs": []}
    validators["jobs-response.schema.json"].validate(empty_jobs)

    completed_id = registry.create_representative(behavior="complete")["job_id"]
    failed_id = registry.create_representative(behavior="fail")["job_id"]
    assert isinstance(completed_id, str)
    assert isinstance(failed_id, str)
    assert _wait_for_terminal(registry, completed_id)["state"] == "completed"
    failed_snapshot = _wait_for_terminal(registry, failed_id)
    assert failed_snapshot["state"] == "failed"
    assert failed_snapshot["error"] is not None

    status, populated_jobs = _request_json(server, "/api/jobs")
    assert status == 200
    assert {job["job_id"] for job in populated_jobs["jobs"]} == {completed_id, failed_id}
    validators["jobs-response.schema.json"].validate(populated_jobs)

    status, completed_detail = _request_json(server, f"/api/jobs/{completed_id}")
    assert status == 200
    assert completed_detail["state"] == "completed"
    assert completed_detail["error"] is None
    validators["job-detail-response.schema.json"].validate(completed_detail)

    status, failed_detail = _request_json(server, f"/api/jobs/{failed_id}")
    assert status == 200
    assert failed_detail["state"] == "failed"
    assert failed_detail["error"] == {
        "code": "representative_failure",
        "message": "Representative job failed as requested.",
    }
    validators["job-detail-response.schema.json"].validate(failed_detail)

    unknown_id = "0" * 32
    assert unknown_id not in {completed_id, failed_id}
    for target in ("/api/jobs/not-a-job-id", f"/api/jobs/{unknown_id}"):
        status, error = _request_json(server, target)
        assert status == 404
        assert error == {"error": "job_not_found", "message": "Job not found."}
        validators["error-response.schema.json"].validate(error)


VALID_TIMESTAMP = "2026-07-17T12:34:56.789Z"
VALID_SUMMARY = {
    "job_id": "0123456789abcdef0123456789abcdef",
    "job_type": "representative",
    "state": "running",
    "progress": 50,
    "current_step": "representative_work",
    "created_at": VALID_TIMESTAMP,
    "started_at": VALID_TIMESTAMP,
    "finished_at": None,
}


def _negative_contract_cases() -> tuple[tuple[str, dict[str, Any]], ...]:
    extra_property = {"jobs": [], "unexpected": True}
    missing_required = {
        "service": "reconstruction_web",
        "version_source": "reconstruction_web",
    }
    wrong_type = {"jobs": [{**VALID_SUMMARY, "progress": "50"}]}
    invalid_timestamp = {
        "jobs": [{**VALID_SUMMARY, "created_at": "2026-07-17T12:34:56Z"}]
    }
    return (
        ("jobs-response.schema.json", extra_property),
        ("version-response.schema.json", missing_required),
        ("jobs-response.schema.json", wrong_type),
        ("jobs-response.schema.json", invalid_timestamp),
    )


@pytest.mark.parametrize(("schema_name", "invalid_payload"), _negative_contract_cases())
def test_contracts_reject_negative_fixtures(
    validators, schema_name: str, invalid_payload: dict[str, Any]
) -> None:
    with pytest.raises(ValidationError):
        validators[schema_name].validate(copy.deepcopy(invalid_payload))


def test_generated_types_embed_current_schema_digest(schemas) -> None:
    canonical_schema_set = json.dumps(
        schemas,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    expected_digest = hashlib.sha256(canonical_schema_set).hexdigest()
    generated_types = GENERATED_TYPES_PATH.read_text(encoding="utf-8")
    match = re.search(
        r'^export const SCHEMA_SHA256 = "([0-9a-f]{64})" as const;$',
        generated_types,
        flags=re.MULTILINE,
    )
    assert match is not None
    assert match.group(1) == expected_digest
