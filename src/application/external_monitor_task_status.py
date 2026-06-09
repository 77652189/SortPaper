from __future__ import annotations

import platform
import subprocess
from pathlib import Path
from typing import Any


TASK_NAME = "PaperSort External Paper Monitor"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_LOG_PATH = PROJECT_ROOT / "logs" / "external_monitor_task.log"
INSTALL_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "install_external_monitor_task.ps1"


def background_monitor_task_status() -> dict[str, Any]:
    status: dict[str, Any] = {
        "task_name": TASK_NAME,
        "installed": False,
        "platform": platform.system(),
        "next_run_time": "",
        "last_run_time": "",
        "last_result": "",
        "task_to_run": "",
        "hidden": False,
        "log_path": str(TASK_LOG_PATH),
        "log_exists": TASK_LOG_PATH.exists(),
        "latest_log_marker": _latest_log_marker(TASK_LOG_PATH),
        "commands": _task_commands(),
        "error": "",
    }
    if platform.system().lower() != "windows":
        status["error"] = "scheduled task status is only available on Windows"
        return status

    try:
        completed = subprocess.run(
            ["schtasks.exe", "/Query", "/TN", TASK_NAME, "/V", "/FO", "LIST"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )
    except Exception as exc:
        status["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        return status

    status["raw"] = completed.stdout
    if completed.returncode != 0:
        status["error"] = (completed.stderr or completed.stdout or "").strip()[:300]
        return status

    parsed = _parse_schtasks_list(completed.stdout)
    task_to_run = parsed.get("Task To Run", "")
    status.update({
        "installed": True,
        "next_run_time": parsed.get("Next Run Time", ""),
        "last_run_time": parsed.get("Last Run Time", ""),
        "last_result": parsed.get("Last Result", ""),
        "status": parsed.get("Status", ""),
        "task_to_run": task_to_run,
        "hidden": "wscript.exe" in task_to_run.lower() and "hidden.vbs" in task_to_run.lower(),
    })
    return status


def _parse_schtasks_list(output: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in output.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def _latest_log_marker(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    for line in reversed(lines):
        if line.startswith("====="):
            return line.strip("= ").strip()
    return ""


def _task_commands() -> dict[str, str]:
    task_name = TASK_NAME
    log_path = str(TASK_LOG_PATH)
    install_path = str(INSTALL_SCRIPT_PATH)
    return {
        "install_or_repair": f'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "{install_path}"',
        "run_now": f'schtasks.exe /Run /TN "{task_name}"',
        "query_status": f'schtasks.exe /Query /TN "{task_name}" /V /FO LIST',
        "tail_log": f'powershell.exe -NoProfile -Command "Get-Content -Path \'{log_path}\' -Tail 60"',
    }
