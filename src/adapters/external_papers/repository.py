from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from src.domain.external_papers import (
    DailyBrief,
    ExternalPaperCandidate,
    MonitorProfile,
    candidate_identity_keys,
    merge_candidate,
)


class JsonExternalImportRepository:
    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.candidates_path = self.root_dir / "candidates.json"
        self.monitors_path = self.root_dir / "monitors.json"
        self.daily_briefs_path = self.root_dir / "daily_briefs.json"
        self.metadata_path = self.root_dir / "metadata.json"
        self.runs_path = self.root_dir / "runs.jsonl"
        self.pdf_dir = self.root_dir / "pdfs"

    def list_candidates(self) -> list[ExternalPaperCandidate]:
        payload = self._read_json(self.candidates_path, default={"candidates": []})
        return [
            ExternalPaperCandidate.from_dict(item)
            for item in payload.get("candidates", [])
            if isinstance(item, dict)
        ]

    def get_candidate(self, candidate_id: str) -> ExternalPaperCandidate | None:
        for candidate in self.list_candidates():
            if candidate.candidate_id == candidate_id:
                return candidate
        return None

    def upsert_candidates(self, candidates: list[ExternalPaperCandidate]) -> dict[str, Any]:
        existing = self.list_candidates()
        by_id = {candidate.candidate_id: candidate for candidate in existing}
        identity_index: dict[str, str] = {}
        for candidate in existing:
            for key in candidate_identity_keys(candidate):
                identity_index[key] = candidate.candidate_id

        new_count = 0
        updated_count = 0
        stored_ids: list[str] = []
        for incoming in candidates:
            matched_id = ""
            for key in candidate_identity_keys(incoming):
                matched_id = identity_index.get(key, "")
                if matched_id:
                    break

            if matched_id and matched_id in by_id:
                merged, changed = merge_candidate(by_id[matched_id], incoming)
                by_id[matched_id] = merged
                if changed:
                    updated_count += 1
                stored_ids.append(matched_id)
                for key in candidate_identity_keys(merged):
                    identity_index[key] = matched_id
                continue

            by_id[incoming.candidate_id] = incoming
            for key in candidate_identity_keys(incoming):
                identity_index[key] = incoming.candidate_id
            new_count += 1
            stored_ids.append(incoming.candidate_id)

        ordered = sorted(by_id.values(), key=lambda item: (item.published_date or "", item.title), reverse=True)
        self._write_json(self.candidates_path, {"candidates": [item.to_dict() for item in ordered]})
        return {
            "new_count": new_count,
            "updated_count": updated_count,
            "candidate_ids": stored_ids,
            "total_count": len(ordered),
        }

    def update_candidate(self, candidate: ExternalPaperCandidate) -> None:
        candidates = self.list_candidates()
        replaced = False
        for index, item in enumerate(candidates):
            if item.candidate_id == candidate.candidate_id:
                candidates[index] = candidate
                replaced = True
                break
        if not replaced:
            candidates.append(candidate)
        self._write_json(self.candidates_path, {"candidates": [item.to_dict() for item in candidates]})

    def list_monitors(self) -> list[MonitorProfile]:
        payload = self._read_json(self.monitors_path, default={"monitors": []})
        return [
            MonitorProfile.from_dict(item)
            for item in payload.get("monitors", [])
            if isinstance(item, dict)
        ]

    def save_monitors(self, monitors: list[MonitorProfile]) -> None:
        self._write_json(self.monitors_path, {"monitors": [item.to_dict() for item in monitors]})

    def list_daily_briefs(self) -> list[DailyBrief]:
        payload = self._read_json(self.daily_briefs_path, default={"daily_briefs": []})
        return [
            DailyBrief.from_dict(item)
            for item in payload.get("daily_briefs", [])
            if isinstance(item, dict)
        ]

    def save_daily_brief(self, brief: DailyBrief) -> None:
        briefs = [item for item in self.list_daily_briefs() if item.brief_id != brief.brief_id]
        briefs.append(brief)
        ordered = sorted(briefs, key=lambda item: item.generated_at, reverse=True)
        self._write_json(self.daily_briefs_path, {"daily_briefs": [item.to_dict() for item in ordered]})

    def append_run_summary(self, summary: dict[str, Any]) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        with self.runs_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, ensure_ascii=False, sort_keys=True) + "\n")

    def list_run_summaries(self) -> list[dict[str, Any]]:
        if not self.runs_path.exists():
            return []
        summaries: list[dict[str, Any]] = []
        for line in self.runs_path.read_text(encoding="utf-8", errors="replace").splitlines():
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                summaries.append(value)
        return summaries

    def get_metadata(self) -> dict[str, Any]:
        return self._read_json(self.metadata_path, default={})

    def update_metadata(self, values: dict[str, Any]) -> dict[str, Any]:
        metadata = self.get_metadata()
        metadata.update(values)
        self._write_json(self.metadata_path, metadata)
        return metadata

    def _read_json(self, path: Path, *, default: dict[str, Any]) -> dict[str, Any]:
        if not path.exists():
            return default
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
        return value if isinstance(value, dict) else default

    def _write_json(self, path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_path, path)
