from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.parsers.mineru_api import MinerUApiError, MinerUClient, MinerUFileUrl, _response_status


class FakeResponse:
    def __init__(self, payload: dict[str, Any] | None = None, *, status: int = 200) -> None:
        self.payload = payload or {}
        self.status = status

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def getcode(self) -> int:
        return self.status


class CapturingTransport:
    def __init__(self, responses: list[dict[str, Any]] | None = None) -> None:
        self.requests = []
        self.responses = responses or [{"code": 0, "data": {}}]

    def __call__(self, request, timeout):
        self.requests.append((request, timeout, request.data))
        payload = self.responses.pop(0) if self.responses else {"code": 0, "data": {}}
        return FakeResponse(payload)


def _json_request_body(transport: CapturingTransport) -> dict[str, Any]:
    data = transport.requests[-1][2]
    return json.loads(data.decode("utf-8"))


def test_create_extract_task_posts_vlm_payload() -> None:
    transport = CapturingTransport([{"code": 0, "data": {"task_id": "task-1"}}])
    client = MinerUClient(api_key="mineru-key", base_url="https://mineru.test", transport=transport)

    task = client.create_extract_task("https://example.test/paper.pdf", data_id="paper-1")

    request, timeout, _ = transport.requests[-1]
    assert task.task_id == "task-1"
    assert request.full_url == "https://mineru.test/api/v4/extract/task"
    assert request.get_method() == "POST"
    assert request.headers["Authorization"] == "Bearer mineru-key"
    body = _json_request_body(transport)
    assert body["url"] == "https://example.test/paper.pdf"
    assert body["model_version"] == "vlm"
    assert body["data_id"] == "paper-1"
    assert body["enable_table"] is True
    assert timeout == 60.0


def test_file_urls_batch_extracts_signed_upload_urls() -> None:
    transport = CapturingTransport([
        {
            "code": 0,
            "data": {
                "batch_id": "batch-1",
                "file_urls": [
                    {"name": "a.pdf", "upload_url": "https://upload.test/a", "data_id": "a.pdf"}
                ],
            },
        }
    ])
    client = MinerUClient(api_key="mineru-key", base_url="https://mineru.test", transport=transport)

    batch = client.create_file_urls_batch(["a.pdf"])

    request, _, _ = transport.requests[-1]
    assert batch.batch_id == "batch-1"
    assert batch.file_urls == [MinerUFileUrl(file_name="a.pdf", upload_url="https://upload.test/a", data_id="a.pdf")]
    assert request.full_url == "https://mineru.test/api/v4/file-urls/batch"
    body = _json_request_body(transport)
    assert body["model_version"] == "vlm"
    assert body["files"][0]["name"] == "a.pdf"


def test_upload_file_uses_signed_url_without_bearer_header(tmp_path: Path) -> None:
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.7")
    transport = CapturingTransport([{}])
    client = MinerUClient(api_key="mineru-key", transport=transport)

    client.upload_file("https://upload.test/a", pdf)

    request, _, body = transport.requests[-1]
    assert request.full_url == "https://upload.test/a"
    assert request.get_method() == "PUT"
    assert "Authorization" not in request.headers
    assert body == b"%PDF-1.7"


def test_upload_file_prefers_requests_for_default_signed_upload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.7")
    calls: list[tuple[str, bytes, dict[str, str]]] = []
    client = MinerUClient(api_key="mineru-key")

    def fake_requests_upload(upload_url: str, data: bytes, headers: dict[str, str]) -> bool:
        calls.append((upload_url, data, headers))
        return True

    def forbidden_send(*args, **kwargs):
        raise AssertionError("default signed upload should not use urllib first")

    monkeypatch.setattr(client, "_upload_file_with_requests", fake_requests_upload)
    monkeypatch.setattr(client, "_send", forbidden_send)

    client.upload_file("https://upload.test/a", pdf)

    assert calls == [("https://upload.test/a", b"%PDF-1.7", {})]


def test_submit_local_file_accepts_ordered_url_list(tmp_path: Path) -> None:
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.7")
    transport = CapturingTransport([
        {
            "code": 0,
            "data": {
                "batch_id": "batch-1",
                "file_urls": ["https://upload.test/a"],
            },
        },
        {},
    ])
    client = MinerUClient(api_key="mineru-key", base_url="https://mineru.test", transport=transport)

    batch = client.submit_local_files([pdf])

    assert batch.batch_id == "batch-1"
    assert transport.requests[1][0].full_url == "https://upload.test/a"
    assert transport.requests[1][2] == b"%PDF-1.7"


def test_missing_api_key_raises_before_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MINERU_API_KEY", raising=False)
    monkeypatch.delenv("\ufeffMINERU_API_KEY", raising=False)
    transport = CapturingTransport()
    client = MinerUClient(api_key="", transport=transport)

    with pytest.raises(ValueError, match="MINERU_API_KEY"):
        client.create_extract_task("https://example.test/paper.pdf")
    assert transport.requests == []


def test_get_batch_result_uses_extract_results_endpoint() -> None:
    transport = CapturingTransport([{"code": 0, "data": {"status": "done"}}])
    client = MinerUClient(api_key="mineru-key", base_url="https://mineru.test", transport=transport)

    result = client.get_batch_result("batch-1")

    request, _, _ = transport.requests[-1]
    assert request.full_url == "https://mineru.test/api/v4/extract-results/batch/batch-1"
    assert request.get_method() == "GET"
    assert result["data"]["status"] == "done"


def test_response_status_reads_extract_result_states() -> None:
    assert _response_status({
        "data": {
            "extract_result": [
                {"state": "done"},
                {"state": "done"},
            ]
        }
    }) == "done"
    assert _response_status({
        "data": {
            "extract_result": [
                {"state": "done"},
                {"state": "failed"},
            ]
        }
    }) == "failed"


def test_mineru_api_error_redacts_signed_url_params() -> None:
    error = MinerUApiError(
        "upload failed",
        body="https://example.test/file.pdf?Expires=123&OSSAccessKeyId=abc&Signature=secret",
    )

    message = str(error)
    assert "Expires=<redacted>" in message
    assert "OSSAccessKeyId=<redacted>" in message
    assert "Signature=<redacted>" in message
    assert "secret" not in message
