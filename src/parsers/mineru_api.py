from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.runtime_env import load_project_env


DEFAULT_MINERU_BASE_URL = "https://mineru.net"
DEFAULT_MINERU_MODEL_VERSION = "vlm"

Transport = Callable[[Request, float], Any]


def _env_value(name: str, default: str = "") -> str:
    return (os.getenv(name) or os.getenv(f"\ufeff{name}") or default).strip()


def mineru_api_key() -> str:
    load_project_env()
    return _env_value("MINERU_API_KEY")


class MinerUApiError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None, body: str = "") -> None:
        self.status_code = status_code
        self.body = _redact_signed_url(body)
        detail = f"status_code={status_code}" if status_code is not None else "no status code"
        if self.body:
            detail += f", body={self.body[:500]}"
        super().__init__(f"{message} ({detail})")


@dataclass(frozen=True)
class MinerUFileUrl:
    file_name: str
    upload_url: str
    data_id: str = ""


@dataclass(frozen=True)
class MinerUUrlTask:
    task_id: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class MinerUBatchTask:
    batch_id: str
    file_urls: list[MinerUFileUrl]
    raw: dict[str, Any]


class MinerUClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model_version: str | None = None,
        timeout: float = 60.0,
        transport: Transport | None = None,
    ) -> None:
        load_project_env()
        self.api_key = (mineru_api_key() if api_key is None else api_key).strip()
        self.base_url = (
            base_url
            or _env_value("MINERU_API_BASE_URL", DEFAULT_MINERU_BASE_URL)
        ).rstrip("/")
        self.model_version = (
            model_version
            or _env_value("MINERU_MODEL_VERSION", DEFAULT_MINERU_MODEL_VERSION)
        ).strip()
        self.timeout = timeout
        self._uses_default_transport = transport is None
        self._transport = transport or _default_transport

    def create_extract_task(
        self,
        file_url: str,
        *,
        data_id: str | None = None,
        is_ocr: bool | None = True,
        enable_formula: bool | None = True,
        enable_table: bool | None = True,
        language: str | None = None,
        extra_payload: dict[str, Any] | None = None,
    ) -> MinerUUrlTask:
        """Submit a URL-based extraction task to /api/v4/extract/task."""
        payload = {
            "url": file_url,
            "model_version": self.model_version,
            "data_id": data_id,
            "is_ocr": is_ocr,
            "enable_formula": enable_formula,
            "enable_table": enable_table,
            "language": language,
        }
        if extra_payload:
            payload.update(extra_payload)
        raw = self._post_json("/api/v4/extract/task", payload)
        task_id = _find_first_string(raw, "task_id", "taskId", "id")
        if not task_id:
            raise MinerUApiError("MinerU extract task response did not include task_id", body=_safe_json(raw))
        return MinerUUrlTask(task_id=task_id, raw=raw)

    def get_extract_task_result(self, task_id: str) -> dict[str, Any]:
        """Fetch a single extraction task result."""
        if not task_id.strip():
            raise ValueError("task_id 不能为空")
        return self._get_json(f"/api/v4/extract/task/{task_id.strip()}")

    def create_file_urls_batch(
        self,
        file_names: list[str],
        *,
        is_ocr: bool | None = True,
        enable_formula: bool | None = True,
        enable_table: bool | None = True,
        extra_payload: dict[str, Any] | None = None,
    ) -> MinerUBatchTask:
        """Request signed upload URLs from /api/v4/file-urls/batch."""
        files = [
            {
                "name": Path(name).name,
                "is_ocr": is_ocr,
                "data_id": Path(name).name,
            }
            for name in file_names
        ]
        payload = {
            "model_version": self.model_version,
            "enable_formula": enable_formula,
            "enable_table": enable_table,
            "files": files,
        }
        if extra_payload:
            payload.update(extra_payload)
        raw = self._post_json("/api/v4/file-urls/batch", payload)
        return MinerUBatchTask(
            batch_id=_find_first_string(raw, "batch_id", "batchId", "id"),
            file_urls=_extract_file_urls(raw),
            raw=raw,
        )

    def upload_file(self, upload_url: str, file_path: str | Path) -> None:
        """Upload one local file to a MinerU signed URL."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        data = path.read_bytes()
        headers: dict[str, str] = {}
        if self._uses_default_transport:
            if self._upload_file_with_requests(upload_url, data, headers):
                return
            if self._upload_file_with_curl(upload_url, path):
                return
        last_error: MinerUApiError | None = None
        for attempt in range(2):
            request = Request(upload_url, data=data, headers=headers, method="PUT")
            try:
                self._send(request, expect_json=False)
                return
            except MinerUApiError as exc:
                last_error = exc
                if attempt == 0 and "network error" in str(exc).lower():
                    time.sleep(1.0)
                    continue
                if "network error" in str(exc).lower() and self._upload_file_with_requests(upload_url, data, headers):
                    return
                if "network error" in str(exc).lower() and self._upload_file_with_curl(upload_url, path):
                    return
                raise
        if last_error is not None:
            if "network error" in str(last_error).lower() and self._upload_file_with_requests(upload_url, data, headers):
                return
            if "network error" in str(last_error).lower() and self._upload_file_with_curl(upload_url, path):
                return
            raise last_error

    def _upload_file_with_requests(self, upload_url: str, data: bytes, headers: dict[str, str]) -> bool:
        try:
            import requests
        except ImportError:
            return False
        try:
            response = requests.put(upload_url, data=data, headers=headers, timeout=self.timeout)
        except requests.RequestException as exc:
            return False
        if response.status_code >= 400:
            raise MinerUApiError(
                "MinerU signed upload failed",
                status_code=response.status_code,
                body=response.text,
            )
        return True

    def _upload_file_with_curl(self, upload_url: str, file_path: Path) -> bool:
        curl = shutil.which("curl.exe") or shutil.which("curl")
        if not curl:
            return False
        command = [
            curl,
            "--fail",
            "--silent",
            "--show-error",
            "--location",
            "--request",
            "PUT",
            "--upload-file",
            str(file_path),
            upload_url,
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=self.timeout,
            check=False,
        )
        if completed.returncode == 0:
            return True
        raise MinerUApiError(
            "MinerU signed upload failed with curl",
            body=completed.stderr or completed.stdout,
        )

    def submit_local_files(
        self,
        file_paths: list[str | Path],
        *,
        is_ocr: bool | None = True,
        enable_formula: bool | None = True,
        enable_table: bool | None = True,
    ) -> MinerUBatchTask:
        """Request upload URLs and upload local PDFs as a batch."""
        paths = [Path(path) for path in file_paths]
        batch = self.create_file_urls_batch(
            [path.name for path in paths],
            is_ocr=is_ocr,
            enable_formula=enable_formula,
            enable_table=enable_table,
        )
        by_name = {item.file_name: item.upload_url for item in batch.file_urls}
        for index, path in enumerate(paths):
            upload_url = by_name.get(path.name)
            if not upload_url and len(batch.file_urls) == len(paths):
                upload_url = batch.file_urls[index].upload_url
            if not upload_url:
                raise MinerUApiError(
                    f"MinerU did not return an upload URL for {path.name}",
                    body=_safe_json(batch.raw),
                )
            self.upload_file(upload_url, path)
        return batch

    def get_batch_result(self, batch_id: str) -> dict[str, Any]:
        """Fetch a batch extraction result."""
        if not batch_id.strip():
            raise ValueError("batch_id 不能为空")
        return self._get_json(f"/api/v4/extract-results/batch/{batch_id.strip()}")

    def poll_extract_task_result(
        self,
        task_id: str,
        *,
        interval_seconds: float = 5.0,
        timeout_seconds: float = 600.0,
    ) -> dict[str, Any]:
        return self._poll(
            lambda: self.get_extract_task_result(task_id),
            interval_seconds=interval_seconds,
            timeout_seconds=timeout_seconds,
        )

    def poll_batch_result(
        self,
        batch_id: str,
        *,
        interval_seconds: float = 5.0,
        timeout_seconds: float = 600.0,
    ) -> dict[str, Any]:
        return self._poll(
            lambda: self.get_batch_result(batch_id),
            interval_seconds=interval_seconds,
            timeout_seconds=timeout_seconds,
        )

    def _poll(
        self,
        fetch: Callable[[], dict[str, Any]],
        *,
        interval_seconds: float,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        while True:
            result = fetch()
            status = _response_status(result)
            if status in {"done", "finished", "success", "completed", "failed", "error"}:
                return result
            if time.monotonic() >= deadline:
                raise TimeoutError(f"MinerU task polling timed out after {timeout_seconds:.0f}s")
            time.sleep(interval_seconds)

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(_drop_none(payload), ensure_ascii=False).encode("utf-8")
        request = Request(
            self._url(path),
            data=data,
            headers=self._json_headers(),
            method="POST",
        )
        return self._send(request)

    def _get_json(self, path: str) -> dict[str, Any]:
        request = Request(self._url(path), headers=self._auth_headers(), method="GET")
        return self._send(request)

    def _send(self, request: Request, *, expect_json: bool = True) -> dict[str, Any]:
        if "mineru.net" in request.full_url and not self.api_key and request.get_method() != "PUT":
            raise ValueError("MINERU_API_KEY 未设置，无法调用 MinerU API")
        try:
            with self._transport(request, self.timeout) as response:
                body_bytes = response.read()
                status_code = getattr(response, "status", None) or response.getcode()
        except HTTPError as exc:
            body = _decode_body(exc.read())
            raise MinerUApiError(
                "MinerU API request failed",
                status_code=exc.code,
                body=body,
            ) from exc
        except URLError as exc:
            raise MinerUApiError("MinerU API network error", body=str(exc.reason)) from exc

        body = _decode_body(body_bytes)
        if not expect_json:
            if status_code and status_code >= 400:
                raise MinerUApiError("MinerU signed upload failed", status_code=status_code, body=body)
            return {}
        parsed = _parse_json_body(body, status_code=status_code)
        if status_code and status_code >= 400:
            raise MinerUApiError("MinerU API request failed", status_code=status_code, body=body)
        _raise_on_api_error(parsed)
        return parsed

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def _auth_headers(self) -> dict[str, str]:
        if not self.api_key:
            raise ValueError("MINERU_API_KEY 未设置，无法调用 MinerU API")
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }

    def _json_headers(self) -> dict[str, str]:
        headers = self._auth_headers()
        headers["Content-Type"] = "application/json"
        return headers


def _drop_none(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {key: _drop_none(value) for key, value in payload.items() if value is not None}
    if isinstance(payload, list):
        return [_drop_none(value) for value in payload]
    return payload


def _default_transport(request: Request, timeout: float) -> Any:
    return urlopen(request, timeout=timeout)


def _decode_body(body: bytes | str | None) -> str:
    if body is None:
        return ""
    if isinstance(body, str):
        return body
    return body.decode("utf-8", errors="replace")


def _parse_json_body(body: str, *, status_code: int | None) -> dict[str, Any]:
    if not body.strip():
        return {}
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise MinerUApiError("MinerU API returned non-JSON response", status_code=status_code, body=body) from exc
    if not isinstance(parsed, dict):
        raise MinerUApiError("MinerU API returned unexpected JSON shape", status_code=status_code, body=body)
    return parsed


def _raise_on_api_error(parsed: dict[str, Any]) -> None:
    code = parsed.get("code")
    if code in (None, 0, "0", "success", "Success", "SUCCESS"):
        return
    message = parsed.get("msg") or parsed.get("message") or "MinerU API returned an error code"
    raise MinerUApiError(str(message), body=_safe_json(parsed))


def _safe_json(value: Any) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False)
    except TypeError:
        text = str(value)
    return _redact_signed_url(text)


def _redact_signed_url(text: str) -> str:
    text = re.sub(r"(OSSAccessKeyId=)[^&\" ]+", r"\1<redacted>", text)
    text = re.sub(r"(Signature=)[^&\" ]+", r"\1<redacted>", text)
    text = re.sub(r"(Expires=)[^&\" ]+", r"\1<redacted>", text)
    return text


def _find_first_string(value: Any, *keys: str) -> str:
    if isinstance(value, dict):
        for key in keys:
            item = value.get(key)
            if isinstance(item, str) and item.strip():
                return item.strip()
        for item in value.values():
            found = _find_first_string(item, *keys)
            if found:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_first_string(item, *keys)
            if found:
                return found
    return ""


def _extract_file_urls(raw: dict[str, Any]) -> list[MinerUFileUrl]:
    candidates: list[Any] = []
    data = raw.get("data")
    if isinstance(data, dict):
        for key in ("file_urls", "fileUrls", "files", "urls"):
            value = data.get(key)
            if isinstance(value, list):
                candidates = value
                break
    if not candidates:
        for key in ("file_urls", "fileUrls", "files", "urls"):
            value = raw.get(key)
            if isinstance(value, list):
                candidates = value
                break

    result: list[MinerUFileUrl] = []
    for item in candidates:
        if isinstance(item, str):
            result.append(MinerUFileUrl(file_name="", upload_url=item))
            continue
        if not isinstance(item, dict):
            continue
        upload_url = (
            item.get("upload_url")
            or item.get("uploadUrl")
            or item.get("url")
            or item.get("file_url")
            or item.get("fileUrl")
        )
        if not upload_url:
            continue
        result.append(
            MinerUFileUrl(
                file_name=str(item.get("name") or item.get("file_name") or item.get("fileName") or ""),
                upload_url=str(upload_url),
                data_id=str(item.get("data_id") or item.get("dataId") or ""),
            )
        )
    return result


def _response_status(raw: dict[str, Any]) -> str:
    data = raw.get("data") if isinstance(raw.get("data"), dict) else raw
    if isinstance(data, dict) and isinstance(data.get("extract_result"), list):
        states = [
            str(item.get("state") or "").strip().lower()
            for item in data["extract_result"]
            if isinstance(item, dict)
        ]
        if states and all(state in {"done", "failed", "error"} for state in states):
            return "failed" if any(state in {"failed", "error"} for state in states) else "done"
        if states:
            return ",".join(states)
    status = (
        data.get("status")
        or data.get("state")
        or data.get("task_status")
        or data.get("taskStatus")
        or ""
    )
    return str(status).strip().lower()
