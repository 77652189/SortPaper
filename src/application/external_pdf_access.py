from __future__ import annotations

from typing import Any, Callable

from src.application.external_candidate_import import repository_root_dir
from src.application.external_source_search import emit_progress
from src.domain.external_papers import utc_now_iso
from src.ports.external_papers import ExternalPdfDownloader


ExternalPdfCheckProgress = Callable[[dict[str, Any]], None]
RepositoryFactory = Callable[[], Any]
PdfDownloaderFactory = Callable[[], ExternalPdfDownloader]


class ExternalPdfAccessService:
    """Application service for checking whether candidate PDFs can be downloaded."""

    def __init__(
        self,
        *,
        repository_factory: RepositoryFactory,
        pdf_downloader_factory: PdfDownloaderFactory,
    ) -> None:
        self.repository_factory = repository_factory
        self.pdf_downloader_factory = pdf_downloader_factory

    def check_candidate_pdf_access(
        self,
        candidate_ids: list[str],
        *,
        repository: Any | None = None,
        pdf_downloader: ExternalPdfDownloader | None = None,
        progress_callback: ExternalPdfCheckProgress | None = None,
    ) -> dict[str, Any]:
        repo = repository or self.repository_factory()
        downloader = pdf_downloader or self.pdf_downloader_factory()
        results: list[dict[str, Any]] = []
        verified = failed = missing_pdf_url = 0
        completed = 0
        total = len(candidate_ids)
        emit_progress(progress_callback, event="start", total=total, completed=0)
        for candidate_id in candidate_ids:
            candidate = repo.get_candidate(candidate_id)
            title = candidate.title if candidate is not None else candidate_id
            emit_progress(
                progress_callback,
                event="candidate_start",
                candidate_id=candidate_id,
                title=title,
                total=total,
                completed=completed,
            )
            if candidate is None:
                failed += 1
                item = {"candidate_id": candidate_id, "status": "failed", "error": "候选不存在"}
                results.append(item)
                completed += 1
                emit_pdf_check_progress(
                    progress_callback,
                    item,
                    title=title,
                    total=total,
                    completed=completed,
                    verified=verified,
                    failed=failed,
                    missing_pdf_url=missing_pdf_url,
                )
                continue
            if not candidate.pdf_url:
                missing_pdf_url += 1
                candidate.pdf_access_status = "unavailable"
                candidate.pdf_checked_at = utc_now_iso()
                candidate.pdf_access_error = "候选没有 PDF URL"
                candidate.updated_at = utc_now_iso()
                repo.update_candidate(candidate)
                item = {
                    "candidate_id": candidate_id,
                    "status": "missing_pdf_url",
                    "error": candidate.pdf_access_error,
                }
                results.append(item)
                completed += 1
                emit_pdf_check_progress(
                    progress_callback,
                    item,
                    title=title,
                    total=total,
                    completed=completed,
                    verified=verified,
                    failed=failed,
                    missing_pdf_url=missing_pdf_url,
                )
                continue

            try:
                candidate.pdf_access_status = "checking"
                candidate.pdf_access_error = ""
                candidate.updated_at = utc_now_iso()
                repo.update_candidate(candidate)

                pdf_bytes, pdf_path = downloader.download(candidate, root_dir=repository_root_dir(repo))
                candidate.pdf_access_status = "verified"
                candidate.pdf_checked_at = utc_now_iso()
                candidate.pdf_access_error = ""
                candidate.downloaded_pdf_path = str(pdf_path)
                candidate.updated_at = utc_now_iso()
                repo.update_candidate(candidate)
                verified += 1
                item = {
                    "candidate_id": candidate_id,
                    "status": "verified",
                    "path": str(pdf_path),
                    "size_bytes": len(pdf_bytes),
                }
                results.append(item)
            except Exception as exc:
                candidate.pdf_access_status = "unavailable"
                candidate.pdf_checked_at = utc_now_iso()
                candidate.pdf_access_error = f"{type(exc).__name__}: {str(exc)[:300]}"
                candidate.updated_at = utc_now_iso()
                repo.update_candidate(candidate)
                failed += 1
                item = {
                    "candidate_id": candidate_id,
                    "status": "failed",
                    "error": candidate.pdf_access_error,
                }
                results.append(item)
            completed += 1
            emit_pdf_check_progress(
                progress_callback,
                item,
                title=title,
                total=total,
                completed=completed,
                verified=verified,
                failed=failed,
                missing_pdf_url=missing_pdf_url,
            )
        emit_progress(
            progress_callback,
            event="done",
            total=total,
            completed=completed,
            verified=verified,
            failed=failed,
            missing_pdf_url=missing_pdf_url,
        )
        return {
            "verified": verified,
            "failed": failed,
            "missing_pdf_url": missing_pdf_url,
            "results": results,
        }


def emit_pdf_check_progress(
    callback: ExternalPdfCheckProgress | None,
    item: dict[str, Any],
    *,
    title: str,
    total: int,
    completed: int,
    verified: int,
    failed: int,
    missing_pdf_url: int,
) -> None:
    emit_progress(
        callback,
        event="candidate_done",
        candidate_id=item.get("candidate_id", ""),
        title=title,
        status=item.get("status", ""),
        error=item.get("error", ""),
        path=item.get("path", ""),
        size_bytes=item.get("size_bytes", 0),
        total=total,
        completed=completed,
        verified=verified,
        failed=failed,
        missing_pdf_url=missing_pdf_url,
    )
