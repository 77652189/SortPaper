from __future__ import annotations

from pathlib import Path

import pytest

from src.adapters.external_papers.downloader import RequestsExternalPdfDownloader
from src.domain.external_papers import make_candidate


class FakeResponse:
    def __init__(
        self,
        chunks: list[bytes],
        *,
        content_type: str = "application/pdf",
        text: str = "",
    ) -> None:
        self._chunks = chunks
        self.headers = {"Content-Type": content_type}
        self.text = text

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int):
        yield from self._chunks


def test_pdf_downloader_writes_candidate_pdf(tmp_path: Path, monkeypatch) -> None:
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return FakeResponse([b"%PDF-1.7\nbody"])

    monkeypatch.setattr("src.adapters.external_papers.downloader.requests.get", fake_get)
    candidate = make_candidate(
        source="europe_pmc",
        title="Download Me",
        doi="10.1/download",
        pdf_url="https://pdf.test/download.pdf",
    )

    pdf_bytes, path = RequestsExternalPdfDownloader().download(candidate, root_dir=tmp_path)

    assert pdf_bytes.startswith(b"%PDF")
    assert path.exists()
    assert path.parent == tmp_path / "pdfs"
    assert calls[0][0] == candidate.pdf_url


def test_pdf_downloader_rejects_oversized_pdf(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "src.adapters.external_papers.downloader.requests.get",
        lambda *args, **kwargs: FakeResponse([b"%PDF", b"x" * 20]),
    )
    candidate = make_candidate(
        source="europe_pmc",
        title="Too Big",
        doi="10.1/big",
        pdf_url="https://pdf.test/big.pdf",
    )

    with pytest.raises(ValueError, match="PDF"):
        RequestsExternalPdfDownloader(max_pdf_mb=0).download(candidate, root_dir=tmp_path)


def test_pdf_downloader_resolves_pmc_pdf_with_ncbi_oa_api(tmp_path: Path, monkeypatch) -> None:
    calls = []
    oa_xml = """
    <OA><records><record id="PMC1">
      <link format="pdf" href="ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_pdf/aa/bb/paper.pdf" />
    </record></records></OA>
    """

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        if str(url).endswith("oa.fcgi"):
            return FakeResponse([], content_type="text/xml", text=oa_xml)
        return FakeResponse([b"%PDF-1.7\nfrom pmc"])

    monkeypatch.setattr("src.adapters.external_papers.downloader.requests.get", fake_get)
    candidate = make_candidate(
        source="pubmed",
        title="PMC paper",
        pmcid="PMC1",
        pdf_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC1/pdf/",
    )

    pdf_bytes, path = RequestsExternalPdfDownloader().download(candidate, root_dir=tmp_path)

    assert pdf_bytes.startswith(b"%PDF")
    assert path.exists()
    assert calls[0][0] == "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
    assert calls[1][0] == "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_pdf/aa/bb/paper.pdf"


def test_pdf_downloader_reports_when_pmc_has_no_pdf_direct_link(tmp_path: Path, monkeypatch) -> None:
    calls = []
    oa_xml = """
    <OA><records><record id="PMC1">
      <link format="tgz" href="ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_package/aa/bb/PMC1.tar.gz" />
    </record></records></OA>
    """

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return FakeResponse([], content_type="text/xml", text=oa_xml)

    monkeypatch.setattr("src.adapters.external_papers.downloader.requests.get", fake_get)
    candidate = make_candidate(
        source="pubmed",
        title="PMC package only",
        pmcid="PMC1",
        pdf_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC1/pdf/",
    )

    with pytest.raises(ValueError, match="未提供 PMC1 的 PDF 直链"):
        RequestsExternalPdfDownloader().download(candidate, root_dir=tmp_path)

    assert len(calls) == 1
