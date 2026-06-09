from __future__ import annotations

import zipfile
from pathlib import Path

from src.parsers.layout_chunk import LayoutChunk
from src.parsers.mineru_vision import describe_mineru_figure_groups


def _chunk(
    *,
    chunk_id: str,
    group_id: str,
    img_path: str,
    role: str = "part",
    caption: str = "Figure 2. Example caption.",
) -> LayoutChunk:
    return LayoutChunk(
        content_type="image",
        raw_content="[Vision reparse needed]",
        page=1,
        bbox=(10.0, 20.0, 110.0, 120.0),
        column=0,
        order_in_page=0,
        chunk_id=chunk_id,
        metadata={
            "parser": "mineru_vlm",
            "mineru_type": "image",
            "img_path": img_path,
            "figure_group_id": group_id,
            "figure_label": "Figure 2",
            "figure_caption": caption,
            "figure_group_role": role,
            "vision_needed": True,
        },
    )


def test_describe_mineru_figure_groups_calls_vision_once_per_group(
    tmp_path: Path,
    monkeypatch,
) -> None:
    zip_path = tmp_path / "mineru.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("images/a.png", b"image-a")
        archive.writestr("images/b.png", b"image-b")

    chunks = [
        _chunk(chunk_id="a", group_id="figure_p1_001", img_path="images/a.png"),
        _chunk(chunk_id="b", group_id="figure_p1_001", img_path="images/b.png", role="caption_owner"),
    ]
    calls: list[dict] = []

    def fake_call(images, prompt, **kwargs):
        calls.append({"images": images, "prompt": prompt, "kwargs": kwargs})
        return "Grouped figure description."

    monkeypatch.setattr("src.parsers.mineru_vision.call_openai_vision_images", fake_call)

    result = describe_mineru_figure_groups(chunks, zip_path, model="vision-test")

    assert result is chunks
    assert len(calls) == 1
    assert len(calls[0]["images"]) == 2
    assert "Figure 2. Example caption." in calls[0]["prompt"]
    assert calls[0]["kwargs"]["model"] == "vision-test"
    for chunk in chunks:
        assert chunk.metadata["vision_group_attempted"] is True
        assert chunk.metadata["vision_group_succeeded"] is True
        assert chunk.metadata["vision_needed"] is False
        assert chunk.metadata["vision_group_description"] == "Grouped figure description."
        assert chunk.raw_content.startswith("Figure 2 group vision description")


def test_describe_mineru_figure_groups_marks_missing_images_without_calling_vision(
    tmp_path: Path,
    monkeypatch,
) -> None:
    zip_path = tmp_path / "mineru.zip"
    with zipfile.ZipFile(zip_path, "w"):
        pass
    chunk = _chunk(chunk_id="a", group_id="figure_p1_001", img_path="images/missing.png")

    def forbidden_call(*args, **kwargs):
        raise AssertionError("vision should not be called without images")

    monkeypatch.setattr("src.parsers.mineru_vision.call_openai_vision_images", forbidden_call)

    describe_mineru_figure_groups([chunk], zip_path)

    assert chunk.metadata["vision_group_attempted"] is False
    assert chunk.metadata["vision_group_succeeded"] is False
    assert chunk.metadata["vision_needed"] is True
    assert "vision_group_description" not in chunk.metadata


def test_describe_mineru_figure_groups_reuses_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    zip_path = tmp_path / "mineru.zip"
    cache_path = tmp_path / "figure_vision.json"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("images/a.png", b"image-a")

    calls: list[int] = []

    def fake_call(images, prompt, **kwargs):
        calls.append(len(images))
        return "Cached figure description."

    monkeypatch.setattr("src.parsers.mineru_vision.call_openai_vision_images", fake_call)
    first = [_chunk(chunk_id="a", group_id="figure_p1_001", img_path="images/a.png")]

    describe_mineru_figure_groups(first, zip_path, model="vision-test", cache_path=cache_path)

    assert calls == [1]
    assert cache_path.exists()
    assert first[0].metadata["vision_group_cache_hit"] is False

    def forbidden_call(*args, **kwargs):
        raise AssertionError("cached group should not call vision again")

    monkeypatch.setattr("src.parsers.mineru_vision.call_openai_vision_images", forbidden_call)
    second = [_chunk(chunk_id="a", group_id="figure_p1_001", img_path="images/a.png")]

    describe_mineru_figure_groups(second, zip_path, model="vision-test", cache_path=cache_path)

    assert second[0].metadata["vision_group_attempted"] is True
    assert second[0].metadata["vision_group_succeeded"] is True
    assert second[0].metadata["vision_group_cache_hit"] is True
    assert second[0].metadata["vision_group_description"] == "Cached figure description."
