from __future__ import annotations

import hashlib
import re
from pathlib import Path
from urllib.parse import urlparse
from xml.etree import ElementTree

import requests

from src.application.settings import (
    EXTERNAL_CONTACT_EMAIL,
    EXTERNAL_HTTP_TIMEOUT_SECONDS,
    EXTERNAL_IMPORTS_DIR,
    EXTERNAL_MAX_PDF_MB,
)
from src.domain.external_papers import ExternalPaperCandidate


class RequestsExternalPdfDownloader:
    def __init__(
        self,
        *,
        timeout: float = EXTERNAL_HTTP_TIMEOUT_SECONDS,
        max_pdf_mb: int = EXTERNAL_MAX_PDF_MB,
        contact_email: str = EXTERNAL_CONTACT_EMAIL,
    ) -> None:
        self.timeout = timeout
        self.max_pdf_mb = max_pdf_mb
        self.contact_email = contact_email

    def download(
        self,
        candidate: ExternalPaperCandidate,
        *,
        root_dir: str | Path = EXTERNAL_IMPORTS_DIR,
    ) -> tuple[bytes, Path]:
        if not candidate.pdf_url:
            raise ValueError("候选没有开放 PDF URL")

        max_bytes = self.max_pdf_mb * 1024 * 1024
        download_url = self._download_url(candidate)
        response = requests.get(
            download_url,
            timeout=self.timeout,
            stream=True,
            headers={
                "User-Agent": user_agent(self.contact_email),
                "Accept": "application/pdf, application/octet-stream, */*",
            },
            allow_redirects=True,
        )
        response.raise_for_status()

        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"PDF 超过大小限制 {self.max_pdf_mb} MB")

        pdf_bytes = b"".join(chunks)
        content_type = str(response.headers.get("Content-Type") or "").lower()
        if not pdf_bytes.startswith(b"%PDF") and "pdf" not in content_type:
            raise ValueError(f"下载内容不像 PDF: Content-Type={content_type or '?'}")

        pdf_dir = Path(root_dir) / "pdfs"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        path = pdf_dir / candidate_filename(candidate)
        path.write_bytes(pdf_bytes)
        return pdf_bytes, path

    def _download_url(self, candidate: ExternalPaperCandidate) -> str:
        if _should_resolve_with_ncbi_oa(candidate):
            resolved = _ncbi_oa_pdf_url(
                candidate.pmcid,
                timeout=self.timeout,
                contact_email=self.contact_email,
            )
            if resolved:
                return resolved
            raise ValueError(f"NCBI OA API 未提供 {candidate.pmcid} 的 PDF 直链")
        return candidate.pdf_url


def default_pdf_downloader() -> RequestsExternalPdfDownloader:
    return RequestsExternalPdfDownloader()


def candidate_filename(candidate: ExternalPaperCandidate) -> str:
    basis = candidate.doi or candidate.pmid or candidate.candidate_id
    title = re.sub(r"[^A-Za-z0-9._-]+", "_", candidate.title)[:80].strip("_") or candidate.candidate_id
    digest = hashlib.sha1(basis.encode("utf-8")).hexdigest()[:8]
    return f"{title}_{digest}.pdf"


def user_agent(contact_email: str = EXTERNAL_CONTACT_EMAIL) -> str:
    suffix = f" mailto:{contact_email}" if contact_email else ""
    return f"SortPaper/1.0 external-paper-importer{suffix}"


def _should_resolve_with_ncbi_oa(candidate: ExternalPaperCandidate) -> bool:
    if not candidate.pmcid:
        return False
    host = urlparse(candidate.pdf_url).netloc.lower()
    return "pmc.ncbi.nlm.nih.gov" in host or "europepmc.org" in host


def _ncbi_oa_pdf_url(pmcid: str, *, timeout: float, contact_email: str = "") -> str:
    response = requests.get(
        "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi",
        params={"id": pmcid},
        timeout=timeout,
        headers={
            "User-Agent": user_agent(contact_email),
            "Accept": "application/xml, text/xml",
        },
    )
    response.raise_for_status()
    try:
        root = ElementTree.fromstring(response.text)
    except ElementTree.ParseError as exc:
        raise ValueError(f"NCBI OA API 响应不是有效 XML: {exc}") from exc
    for link in root.findall(".//link"):
        if str(link.attrib.get("format") or "").lower() == "pdf":
            return _ftp_to_https(str(link.attrib.get("href") or ""))
    return ""


def _ftp_to_https(url: str) -> str:
    return re.sub(r"^ftp://ftp\.ncbi\.nlm\.nih\.gov/", "https://ftp.ncbi.nlm.nih.gov/", url.strip(), flags=re.I)
