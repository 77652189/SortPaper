from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.parsers.mineru_api import MinerUClient


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MinerU API smoke test for URL or local PDF extraction.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="Publicly accessible PDF URL for /api/v4/extract/task")
    source.add_argument("--file", type=Path, help="Local PDF path for /api/v4/file-urls/batch signed upload")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model-version", default=None)
    parser.add_argument("--poll", action="store_true")
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--poll-timeout", type=float, default=600.0)
    parser.add_argument("--out", type=Path, default=None, help="Optional path to save the full JSON response")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    client = MinerUClient(base_url=args.base_url, model_version=args.model_version)
    if args.url:
        task = client.create_extract_task(args.url)
        output: dict[str, object] = {"mode": "url", "task_id": task.task_id, "submit": task.raw}
        print(f"submitted url task_id={task.task_id}", file=sys.stderr)
        if args.poll:
            output["result"] = _poll_with_progress(
                lambda: client.get_extract_task_result(task.task_id),
                interval_seconds=args.poll_interval,
                timeout_seconds=args.poll_timeout,
            )
    else:
        batch = client.submit_local_files([args.file])
        output = {
            "mode": "file",
            "batch_id": batch.batch_id,
            "file_urls": [item.__dict__ for item in batch.file_urls],
            "submit": batch.raw,
        }
        print(f"submitted file batch_id={batch.batch_id} upload_urls={len(batch.file_urls)}", file=sys.stderr)
        if args.poll:
            output["result"] = _poll_with_progress(
                lambda: client.get_batch_result(batch.batch_id),
                interval_seconds=args.poll_interval,
                timeout_seconds=args.poll_timeout,
            )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved full response to {args.out}", file=sys.stderr)
    print(json.dumps(_summarize_output(output), ensure_ascii=False, indent=2))
    return 0


def _poll_with_progress(fetch, *, interval_seconds: float, timeout_seconds: float) -> dict[str, object]:
    deadline = time.monotonic() + timeout_seconds
    round_index = 0
    while True:
        round_index += 1
        result = fetch()
        status = _status(result)
        print(
            f"poll round={round_index} status={status or '<unknown>'} keys={','.join(result.keys())}",
            file=sys.stderr,
        )
        if status in {"done", "finished", "success", "completed", "failed", "error"}:
            return result
        if time.monotonic() >= deadline:
            raise TimeoutError(f"MinerU polling timed out after {timeout_seconds:.0f}s")
        time.sleep(interval_seconds)


def _status(raw: dict[str, object]) -> str:
    data = raw.get("data") if isinstance(raw.get("data"), dict) else raw
    if not isinstance(data, dict):
        return ""
    extract_result = data.get("extract_result")
    if isinstance(extract_result, list):
        states = [
            str(item.get("state") or "").strip().lower()
            for item in extract_result
            if isinstance(item, dict)
        ]
        if states and all(state in {"done", "failed", "error"} for state in states):
            return "failed" if any(state in {"failed", "error"} for state in states) else "done"
        if states:
            return ",".join(states)
    return str(
        data.get("status")
        or data.get("state")
        or data.get("task_status")
        or data.get("taskStatus")
        or ""
    ).strip().lower()


def _summarize_output(output: dict[str, object]) -> dict[str, object]:
    result = output.get("result")
    result_keys = sorted(result.keys()) if isinstance(result, dict) else []
    data = result.get("data") if isinstance(result, dict) and isinstance(result.get("data"), dict) else {}
    return {
        "mode": output.get("mode"),
        "task_id": output.get("task_id"),
        "batch_id": output.get("batch_id"),
        "status": _status(result) if isinstance(result, dict) else "",
        "result_keys": result_keys,
        "data_keys": sorted(data.keys()) if isinstance(data, dict) else [],
    }


if __name__ == "__main__":
    raise SystemExit(main())
