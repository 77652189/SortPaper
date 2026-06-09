from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from src.application.settings import EXTERNAL_IMPORTS_DIR
from src.domain.external_papers import ExternalPaperCandidate, utc_now_iso
from src.ports.external_papers import ExternalPdfDownloader


PaperIdBuilder = Callable[[bytes], str]
IngestRunner = Callable[..., dict]
PaperCountLookup = Callable[[str], int]
RepositoryFactory = Callable[[], Any]
PdfDownloaderFactory = Callable[[], ExternalPdfDownloader]


class ExternalCandidateImportService:
    """Application service for importing selected external candidates into storage."""

    def __init__(
        self,
        *,
        repository_factory: RepositoryFactory,
        pdf_downloader_factory: PdfDownloaderFactory,
    ) -> None:
        self.repository_factory = repository_factory
        self.pdf_downloader_factory = pdf_downloader_factory

    def import_candidates(
        self,
        candidate_ids: list[str],
        *,
        repository: Any | None = None,
        paper_id_builder: PaperIdBuilder,
        ingest_runner: IngestRunner,
        paper_count_lookup: PaperCountLookup | None = None,
        pdf_downloader: ExternalPdfDownloader | None = None,
    ) -> dict[str, Any]:
        repo = repository or self.repository_factory()
        downloader = pdf_downloader or self.pdf_downloader_factory()
        results: list[dict[str, Any]] = []
        success = duplicate = failed = 0
        for candidate_id in candidate_ids:
            candidate = repo.get_candidate(candidate_id)
            if candidate is None:
                failed += 1
                results.append({"candidate_id": candidate_id, "status": "failed", "error": "候选不存在"})
                continue
            try:
                candidate.import_status = "importing"
                candidate.import_error = ""
                candidate.updated_at = utc_now_iso()
                repo.update_candidate(candidate)

                cached_pdf = read_validated_cached_pdf(candidate)
                if cached_pdf is None:
                    pdf_bytes, pdf_path = downloader.download(candidate, root_dir=repository_root_dir(repo))
                else:
                    pdf_bytes, pdf_path = cached_pdf
                candidate.pdf_access_status = "verified"
                candidate.pdf_checked_at = utc_now_iso()
                candidate.pdf_access_error = ""
                paper_id = paper_id_builder(pdf_bytes)
                candidate.imported_paper_id = paper_id
                candidate.downloaded_pdf_path = str(pdf_path)
                candidate.updated_at = utc_now_iso()
                if paper_count_lookup is not None and paper_count_lookup(paper_id) > 0:
                    candidate.import_status = "duplicate"
                    candidate.import_error = "该 PDF 已在向量库中"
                    duplicate += 1
                    repo.update_candidate(candidate)
                    results.append({"candidate_id": candidate_id, "status": "duplicate", "paper_id": paper_id})
                    continue

                ingest_result = ingest_runner(
                    pdf_bytes,
                    paper_id,
                    candidate_filename(candidate),
                    describe_figure_groups=True,
                )
                store_result = ingest_result.get("store_result", {}) or {}
                if store_result.get("duplicate"):
                    candidate.import_status = "duplicate"
                    candidate.import_error = f"已存在 {store_result.get('existing_count', '?')} 条记录"
                    duplicate += 1
                elif store_result.get("error"):
                    candidate.import_status = "failed"
                    candidate.import_error = str(store_result.get("error"))
                    failed += 1
                else:
                    candidate.import_status = "imported"
                    candidate.import_error = ""
                    success += 1
                candidate.updated_at = utc_now_iso()
                repo.update_candidate(candidate)
                results.append({
                    "candidate_id": candidate_id,
                    "status": candidate.import_status,
                    "paper_id": paper_id,
                    "store_result": store_result,
                })
            except Exception as exc:
                candidate.import_status = "failed"
                candidate.import_error = f"{type(exc).__name__}: {str(exc)[:300]}"
                candidate.updated_at = utc_now_iso()
                repo.update_candidate(candidate)
                failed += 1
                results.append({"candidate_id": candidate_id, "status": "failed", "error": candidate.import_error})
        return {"success": success, "duplicate": duplicate, "failed": failed, "results": results}


def candidate_filename(candidate: ExternalPaperCandidate) -> str:
    from src.adapters.external_papers.downloader import candidate_filename as adapter_candidate_filename

    return adapter_candidate_filename(candidate)


def repository_root_dir(repo: Any) -> Path:
    return Path(getattr(repo, "root_dir", EXTERNAL_IMPORTS_DIR))


def read_validated_cached_pdf(candidate: ExternalPaperCandidate) -> tuple[bytes, Path] | None:
    if candidate.pdf_access_status != "verified" or not candidate.downloaded_pdf_path:
        return None
    path = Path(candidate.downloaded_pdf_path)
    if not path.exists() or not path.is_file():
        return None
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if not data.startswith(b"%PDF"):
        return None
    return data, path
