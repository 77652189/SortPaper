from __future__ import annotations

import os
from pathlib import Path


PROVIDER_OPTIONS = ["dashscope", "qwen3_local", "openai"]


def env_value(name: str, default: str = "") -> str:
    return (os.getenv(name) or os.getenv(f"\ufeff{name}") or default).strip()


def env_key_configured(name: str) -> bool:
    return bool(name and env_value(name))


def provider_collection_default(provider: str) -> str:
    return "papers_qwen3_local" if provider == "qwen3_local" else "papers"


def provider_dim_default(provider: str) -> int:
    return 3072 if provider == "openai" else 1024


def format_env_value(value: str) -> str:
    value = str(value)
    if not value:
        return ""
    if any(ch.isspace() for ch in value) or "#" in value or '"' in value:
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return value


def update_env_text(text: str, updates: dict[str, str]) -> str:
    lines = text.splitlines()
    seen: set[str] = set()
    rendered: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            rendered.append(line)
            continue
        key = line.split("=", 1)[0].strip()
        if key in updates:
            rendered.append(f"{key}={format_env_value(updates[key])}")
            seen.add(key)
        else:
            rendered.append(line)
    for key, value in updates.items():
        if key not in seen:
            rendered.append(f"{key}={format_env_value(value)}")
    return "\n".join(rendered).rstrip() + "\n"


def save_env_updates(updates: dict[str, str], *, env_path: Path) -> None:
    env_path.parent.mkdir(parents=True, exist_ok=True)
    current = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    env_path.write_text(update_env_text(current, updates), encoding="utf-8")
    for key, value in updates.items():
        os.environ[key] = value
