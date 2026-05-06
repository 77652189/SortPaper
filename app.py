"""
SortPaper GUI — Streamlit 前端
功能：导入 PDF → 解析（预览 / 完整流水线）→ 查看结果
"""

from __future__ import annotations

import hashlib
import json
import logging
import tempfile
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ─── 页面配置 ──────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SortPaper",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── 常量 ──────────────────────────────────────────────────────────────────────

SAMPLE_DIR = Path("data/sample_papers")
FAISS_DIR = Path("faiss_index")
LOGO = "📄"

# ─── 工具函数 ──────────────────────────────────────────────────────────────────


def build_paper_id(pdf_bytes: bytes) -> str:
    return hashlib.sha256(pdf_bytes).hexdigest()[:16]


def type_badge(ctype: str) -> str:
    colors = {"text": "🔵", "table": "🟠", "image": "🟢"}
    return colors.get(ctype, "⚪")


def verdict_badge(passed: bool) -> str:
    return "✅" if passed else "❌"


@st.cache_data(show_spinner=False)
def run_preview(pdf_bytes: bytes) -> dict:
    """Quick preview: parsers + LayoutMerger.merge, no LLM calls."""
    import sys
    sys.path.insert(0, ".")
    from src.parsers.pymupdf_parser import PyMuPDFParser
    from src.parsers.table_parser import TableParser
    from src.parsers.layout_chunk import LayoutMerger

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = Path(tmp.name)

    try:
        text_chunks = PyMuPDFParser(tmp_path).parse()
        table_chunks = TableParser(tmp_path).parse()

        # 合并后再返回，预览模式也能看到 header 合并效果
        print(f"[preview] before merge: {len(text_chunks)} text + {len(table_chunks)} table = {len(text_chunks + table_chunks)} total")
        for c in (text_chunks + table_chunks)[:5]:
            print(f"  p{c.page} col{c.column} y{c.bbox[1]:.1f}-{c.bbox[3]:.1f} | {c.raw_content[:50]}")
        merged_chunks = LayoutMerger.merge(text_chunks + table_chunks)
        print(f"[preview] after merge: {len(merged_chunks)} chunks")

        # image metadata only (no VisionParser — avoids API cost)
        import fitz  # type: ignore
        doc = fitz.open(str(tmp_path))
        images = []
        for page_index in range(len(doc)):
            page = doc[page_index]
            img_list = page.get_images(full=True)
            for img_info in img_list:
                xref = img_info[0]
                bbox_list = page.get_image_rects(xref)
                bbox = tuple(bbox_list[0]) if bbox_list else None
                images.append({
                    "page": page_index + 1,
                    "xref": xref,
                    "width": img_info[2],
                    "height": img_info[3],
                    "bbox": bbox,
                })
        doc.close()
    finally:
        tmp_path.unlink(missing_ok=True)

    return {
        "text_chunks": [_chunk_to_dict(c) for c in text_chunks],
        "table_chunks": [_chunk_to_dict(c) for c in table_chunks],
        "merged_chunks": [_chunk_to_dict(c) for c in merged_chunks],
        "images": images,
        "verdicts": {},
        "mode": "preview",
    }


@st.cache_data(show_spinner=False)
def run_pipeline(pdf_bytes: bytes, paper_id: str) -> dict:
    """Full pipeline: parsers + judge + FAISS store."""
    import sys
    sys.path.insert(0, ".")
    from src.graph.pipeline_graph import build_graph

    out_dir = str(FAISS_DIR / paper_id)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = Path(tmp.name)

    try:
        pipeline = build_graph()
        final = pipeline.invoke(
            {
                "pdf_path": str(tmp_path),
                "paper_id": paper_id,
                "text_retries": 0,
                "table_retries": 0,
                "image_retries": 0,
                "status": "processing",
                "output_dir": out_dir,
            },
            config={"recursion_limit": 100},
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    all_verdicts: dict[str, dict] = {}
    for v in (
        final.get("text_verdicts", [])
        + final.get("table_verdicts", [])
        + final.get("image_verdicts", [])
    ):
        if v.get("chunk_id"):
            all_verdicts[v["chunk_id"]] = v

    merged = final.get("merged_chunks", [])

    # load records if FAISS was saved
    records: list[dict] = []
    records_path = Path(out_dir) / "records.json"
    if records_path.exists():
        records = json.loads(records_path.read_text(encoding="utf-8"))

    return {
        "text_chunks": [_chunk_to_dict(c) for c in final.get("text_chunks", [])],
        "table_chunks": [_chunk_to_dict(c) for c in final.get("table_chunks", [])],
        "image_chunks": [_chunk_to_dict(c) for c in final.get("image_chunks", [])],
        "merged_chunks": [_chunk_to_dict(c) for c in merged],
        "verdicts": all_verdicts,
        "status": final.get("status", "unknown"),
        "records": records,
        "out_dir": out_dir,
        "mode": "pipeline",
    }


def _chunk_to_dict(chunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "content_type": chunk.content_type,
        "page": chunk.page,
        "column": chunk.column,
        "raw_content": chunk.raw_content,
        "bbox": chunk.bbox,
        "metadata": chunk.metadata or {},
    }


def faiss_search(query: str, paper_id: str, top_k: int = 5) -> list[dict]:
    import sys
    sys.path.insert(0, ".")
    import numpy as np
    from src.store.faiss_store import FAISSStore

    index_dir = FAISS_DIR / paper_id
    if not (index_dir / "index.faiss").exists():
        return []

    store = FAISSStore()
    store.load(index_dir)
    embedding = store.embed(query)
    vec = np.array([embedding], dtype="float32")
    distances, indices = store.index.search(vec, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(store.records):
            continue
        rec = store.records[idx]
        results.append({**rec, "distance": float(dist)})
    return results


# ─── PDF 重建 ──────────────────────────────────────────────────────────────────


def render_reconstruction_tab(result: dict) -> None:
    """Chunk 重建可视化：空白画布上按 bbox 排列 chunk，每个框显示 global_order + 内容摘要。"""
    import fitz
    import textwrap
    from PIL import Image, ImageDraw, ImageFont

    merged = result.get("merged_chunks", [])
    if not merged:
        st.info("无 chunk 数据，无法重建")
        return

    color_mode = st.radio(
        "着色方式",
        ["按内容类型", "按栏位"],
        horizontal=True,
    )

    type_colors = {
        "text":  ("#2196F3", "#BBDEFB"),
        "table": ("#FF9800", "#FFE0B2"),
        "image": ("#4CAF50", "#C8E6C9"),
    }
    column_colors = {
        0: ("#1565C0", "#BBDEFB"),
        1: ("#C62828", "#FFCDD2"),
        2: ("#7B1FA2", "#E1BEE7"),
    }

    def get_color(chunk: dict) -> tuple[str, str]:
        if color_mode == "按内容类型":
            return type_colors.get(chunk.get("content_type", "text"), ("#888", "#ddd"))
        col = chunk.get("column", 0)
        return column_colors.get(col, ("#888", "#ddd"))

    pdf_file = st.session_state.get("_pdf_bytes")
    if not pdf_file:
        st.warning("PDF 文件数据丢失，请重新上传")
        return

    try:
        doc = fitz.open(stream=pdf_file, filetype="pdf")
        page_count = len(doc)
        pages = sorted(set(c["page"] for c in merged if 1 <= c["page"] <= page_count))
        if "recon_page" not in st.session_state:
            st.session_state.recon_page = pages[0]
        page_num = st.select_slider("选择页面", options=pages, key="recon_page")
        page_num = max(1, min(page_num, page_count))
        pw, ph = float(doc[page_num - 1].rect.width), float(doc[page_num - 1].rect.height)
        doc.close()
    except Exception as e:
        st.error(f"页面加载失败：{e}")
        return

    # 高分辨率画布（2x 缩放，保证字体清晰）
    scale = min(1560 / pw, 1800 / ph)
    canvas_w, canvas_h = int(pw * scale), int(ph * scale)
    img = Image.new("RGB", (canvas_w, canvas_h), (248, 249, 250))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("segoeui.ttf", 14)
        font_small = ImageFont.truetype("segoeui.ttf", 11)
    except OSError:
        font = ImageFont.load_default()
        font_small = font

    page_chunks = [c for c in merged if c["page"] == page_num]
    page_chunks.sort(key=lambda c: c.get("bbox", (0, 0, 0, 0))[1] if c.get("bbox") else 0)

    for chunk in page_chunks:
        bbox = chunk.get("bbox")
        if not bbox:
            continue
        x0, y0, x1, y1 = bbox
        px0, py0 = x0 * scale, y0 * scale
        px1, py1 = x1 * scale, y1 * scale
        box_w, box_h = px1 - px0, py1 - py0

        if box_w < 8 or box_h < 8:
            continue

        outline, fill = get_color(chunk)
        draw.rectangle([px0, py0, px1, py1], outline=outline, fill=fill + "30", width=2)

        # global_order 标签（深底白字）
        g_order = chunk.get("global_order", chunk.get("order_in_page", 0))
        label = str(g_order)
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        draw.rectangle([px0 + 2, py0 + 2, px0 + tw + 8, py0 + th + 6], fill=outline, width=0)
        draw.text((px0 + 5, py0 + 4), label, fill="white", font=font)

        # 内容摘要（多行自适应，尽量填满框的可用空间）
        content = chunk.get("raw_content", "").strip()
        ctype = chunk.get("content_type", "text")
        type_tag = {"text": "T", "table": "TBL", "image": "IMG"}.get(ctype, "?")

        if not content:
            # 无内容：只显示类型标签
            tag_text = f"[{type_tag}] (empty)"
            tbx = draw.textbbox((0, 0), tag_text, font=font_small)
            tw2, th2 = tbx[2] - tbx[0], tbx[3] - tbx[1]
            if box_h > th + th2 + 12:
                draw.text((px0 + 5, py0 + th + 10), tag_text, fill="#888", font=font_small)
            continue

        # 计算框内可用宽度 → 每行可容纳字符数
        text_area_left = px0 + 5
        text_area_width = box_w - 12
        avg_char_w = font_small.getlength("ABCDEFGH") / 8
        if avg_char_w > 0:
            chars_per_line = max(10, int(text_area_width / avg_char_w))
        else:
            chars_per_line = 60

        # 计算可用高度 → 可容纳行数（在 order 标签下方）
        lh = font_small.size + 4  # 行高
        available_h = box_h - th - 14  # th = order 标签高度
        max_lines = max(1, int(available_h / lh))

        # 多行换行，限制行数
        lines = textwrap.wrap(content, width=chars_per_line)
        display_lines = lines[:max_lines]

        if max_lines >= 2 and len(display_lines) >= 1:
            # 第一行带 [T] 标签，后续行继续显示内容
            lines_with_tag = [f"[{type_tag}] {display_lines[0]}"] + display_lines[1:]
            for i, line in enumerate(lines_with_tag):
                draw.text((text_area_left, py0 + th + 6 + i * lh), line, fill="#333", font=font_small)
        elif max_lines >= 1 and len(display_lines) >= 1:
            # 只能显示一行，带 [T] 标签
            line_text = f"[{type_tag}] {display_lines[0]}"
            tbx = draw.textbbox((0, 0), line_text, font=font_small)
            tw2, th2 = tbx[2] - tbx[0], tbx[3] - tbx[1]
            if box_h <= th + th2 + 10:
                # 框太小，只显示 [T]
                draw.text((text_area_left, py0 + th + 6), f"[{type_tag}]", fill="#555", font=font_small)
            else:
                draw.text((text_area_left, py0 + th + 6), line_text, fill="#333", font=font_small)

    st.image(img, use_container_width=True)

    # 图例
    if color_mode == "按内容类型":
        c1, c2, c3 = st.columns(3)
        for col, (ctype, (clr, _)) in zip([c1, c2, c3], type_colors.items()):
            col.markdown(
                f"<span style='display:inline-block;width:14px;height:14px;"                f"background:{clr};border-radius:3px;margin-right:6px;'></span>{ctype}",
                unsafe_allow_html=True,
            )
    else:
        c1, c2, c3 = st.columns(3)
        labels = {0: "左栏", 1: "右栏", 2: "通栏"}
        for col, (col_id, (clr, _)) in zip([c1, c2, c3], column_colors.items()):
            col.markdown(
                f"<span style='display:inline-block;width:14px;height:14px;"                f"background:{clr};border-radius:3px;margin-right:6px;'></span>{labels[col_id]}",
                unsafe_allow_html=True,
            )

    st.caption(f"共 {len(page_chunks)} 个 chunks | 页面 {page_num}")
def render_chunk_card(chunk: dict, verdict: dict | None = None, index: int = 0) -> None:
    ctype = chunk.get("content_type", "text")
    badge = type_badge(ctype)
    page = chunk.get("page", "?")
    cid = chunk.get("chunk_id", "")
    short_id = cid.split("_", 2)[-1] if "_" in cid else cid

    label = f"{badge} p{page} · {short_id}"
    if verdict:
        label = f"{verdict_badge(verdict['passed'])} {label}  score={verdict.get('score', 0):.2f}"

    with st.expander(label, expanded=False):
        content = chunk.get("raw_content", "")
        if ctype == "table":
            st.markdown(content)
        elif ctype == "image":
            st.markdown(f"**描述：** {content}")
        else:
            st.text(content[:1200] + ("…" if len(content) > 1200 else ""))

        col1, col2, col3 = st.columns(3)
        col1.caption(f"类型: {ctype}")
        col2.caption(f"页: {page}")
        col3.caption(f"列: {chunk.get('column', '?')}")

        if verdict:
            fb = verdict.get("feedback", "")
            if fb:
                st.caption(f"**Judge 意见：** {fb}")


def render_overview(result: dict) -> None:
    text_n = len(result.get("text_chunks", []))
    table_n = len(result.get("table_chunks", []))
    img_n = len(result.get("images", result.get("image_chunks", [])))
    mode = result.get("mode", "preview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔵 文本块", text_n)
    c2.metric("🟠 表格块", table_n)
    c3.metric("🟢 图片", img_n)

    if mode == "pipeline":
        verdicts = result.get("verdicts", {})
        passed = sum(1 for v in verdicts.values() if v["passed"])
        failed = sum(1 for v in verdicts.values() if not v["passed"])
        c4.metric("Judge 通过", f"{passed} / {passed + failed}")

        status = result.get("status", "")
        if status == "done":
            st.success("✅ 流水线完成，FAISS 已保存")
        elif status == "partial":
            st.warning("⚠️ 部分 chunk 判定失败，已降级存储通过部分")
        else:
            st.info(f"状态: {status}")
    else:
        c4.metric("模式", "快速预览")
        st.info("ℹ️ 快速预览模式：仅解析，不调用 LLM，无 Judge / FAISS 写入")


def render_text_tab(chunks: list[dict], verdicts: dict) -> None:
    if not chunks:
        st.info("无文本块")
        return
    st.caption(f"共 {len(chunks)} 个文本块")
    for i, c in enumerate(chunks):
        verdict = verdicts.get(c["chunk_id"])
        render_chunk_card(c, verdict, i)


def render_table_tab(chunks: list[dict], verdicts: dict) -> None:
    if not chunks:
        st.info("无表格块（已过滤所有误检伪表格）")
        return
    st.caption(f"共 {len(chunks)} 个表格块")
    for i, c in enumerate(chunks):
        verdict = verdicts.get(c["chunk_id"])
        render_chunk_card(c, verdict, i)


def render_image_tab(result: dict) -> None:
    mode = result.get("mode", "preview")
    if mode == "preview":
        images = result.get("images", [])
        if not images:
            st.info("未检测到嵌入图片")
            return
        st.caption(f"检测到 {len(images)} 张嵌入图片（快速预览模式不调用 VisionParser）")
        for img in images:
            st.markdown(
                f"- **第 {img['page']} 页** — xref={img['xref']} | "
                f"尺寸 {img['width']}×{img['height']} px"
            )
    else:
        chunks = result.get("image_chunks", [])
        verdicts = result.get("verdicts", {})
        if not chunks:
            st.info("无图片块")
            return
        st.caption(f"共 {len(chunks)} 个图片块（已由 VisionParser 生成描述）")
        for i, c in enumerate(chunks):
            verdict = verdicts.get(c["chunk_id"])
            render_chunk_card(c, verdict, i)


def render_search_tab(paper_id: str) -> None:
    index_path = FAISS_DIR / paper_id / "index.faiss"
    if not index_path.exists():
        st.info("FAISS 索引不存在，请先运行完整流水线")
        return

    query = st.text_input("输入检索关键词或问题", placeholder="e.g. lacto-N-tetraose biosynthesis yield")
    top_k = st.slider("返回条数", 1, 10, 5)
    if st.button("🔍 检索"):
        if not query.strip():
            st.warning("请输入查询内容")
            return
        with st.spinner("向量检索中…"):
            results = faiss_search(query, paper_id, top_k)
        if not results:
            st.warning("未找到相关结果")
            return
        st.caption(f"找到 {len(results)} 条结果")
        for r in results:
            dist = r.get("distance", 0)
            wtype = r.get("worker_type", "")
            content = r.get("content", "")
            meta = r.get("metadata", {})
            pg = meta.get("page", "?")
            with st.expander(f"📌 距离={dist:.4f} | {wtype} | p{pg}", expanded=True):
                st.text(content[:800])


# ─── 侧边栏 ────────────────────────────────────────────────────────────────────


def sidebar() -> tuple[bytes | None, str, str]:
    """返回 (pdf_bytes, paper_id, mode)。"""
    with st.sidebar:
        st.title(f"{LOGO} SortPaper")
        st.caption("学术论文解析与检索工具")
        st.divider()

        # ── 文件来源 ──
        source = st.radio("PDF 来源", ["上传文件", "示例论文"], horizontal=True)

        pdf_bytes: bytes | None = None
        filename = ""

        if source == "上传文件":
            uploaded = st.file_uploader("选择 PDF 文件", type="pdf", label_visibility="collapsed")
            if uploaded:
                pdf_bytes = uploaded.read()
                filename = uploaded.name
        else:
            samples = sorted(SAMPLE_DIR.glob("*.pdf")) if SAMPLE_DIR.exists() else []
            if not samples:
                st.warning("data/sample_papers/ 目录为空")
            else:
                choice = st.selectbox(
                    "选择示例论文",
                    options=samples,
                    format_func=lambda p: p.name,
                    label_visibility="collapsed",
                )
                if choice:
                    pdf_bytes = choice.read_bytes()
                    filename = choice.name

        if pdf_bytes:
            paper_id = build_paper_id(pdf_bytes)
            st.caption(f"📎 {filename}")
            st.caption(f"ID: `{paper_id}`")
        else:
            paper_id = ""

        st.divider()

        # ── 解析模式 ──
        st.subheader("解析模式")
        mode = st.radio(
            "模式",
            ["快速预览", "完整流水线"],
            captions=["仅解析，无 LLM 调用（免费）", "Judge + FAISS 写入（消耗 API 配额）"],
            label_visibility="collapsed",
        )

        st.divider()

        # ── 解析按钮 ──
        can_parse = pdf_bytes is not None
        parse_clicked = st.button(
            "🚀 开始解析",
            disabled=not can_parse,
            use_container_width=True,
            type="primary",
        )

        if not can_parse:
            st.caption("请先选择 PDF 文件")

    return pdf_bytes, paper_id, mode, parse_clicked  # type: ignore[return-value]


# ─── 主界面 ────────────────────────────────────────────────────────────────────


def main() -> None:
    pdf_bytes, paper_id, mode, parse_clicked = sidebar()

    # session state 管理
    if "result" not in st.session_state:
        st.session_state.result = None
    if "current_paper" not in st.session_state:
        st.session_state.current_paper = None

    # 切换论文时清除旧结果
    if paper_id and paper_id != st.session_state.current_paper:
        st.session_state.result = None
        st.session_state.current_paper = paper_id

    if parse_clicked and pdf_bytes:
        st.session_state._pdf_bytes = pdf_bytes  # 供 PDF 重建使用
        mode_key = "preview" if mode == "快速预览" else "pipeline"
        with st.spinner(f"解析中（{mode}）…"):
            try:
                if mode_key == "preview":
                    result = run_preview(pdf_bytes)
                else:
                    result = run_pipeline(pdf_bytes, paper_id)
                st.session_state.result = result
            except Exception as exc:
                st.error(f"解析失败：{exc}")
                logging.exception("parse error")
                return

    result = st.session_state.result

    if result is None:
        # 欢迎页
        st.markdown("## 欢迎使用 SortPaper 📄")
        st.markdown(
            """
            **使用步骤：**
            1. 在左侧选择或上传 PDF 文件
            2. 选择解析模式（快速预览 / 完整流水线）
            3. 点击 **🚀 开始解析**
            4. 在各标签页查看解析结果

            ---
            **快速预览**：仅调用本地解析器（PyMuPDF + pdfplumber），即时出结果，无需 API 配额。  
            **完整流水线**：调用 LLM Judge（qwen-max）评估质量，图片调用 VisionParser（qwen-vl-max）生成描述，最终写入 FAISS 向量索引，支持语义检索。
            """
        )
        return

    # 结果展示
    verdicts = result.get("verdicts", {})

    tabs = st.tabs(["📊 概览", "📝 文本块", "🖼️ 图片", "📋 表格", "📐 PDF 重建", "🔍 语义检索"])

    with tabs[0]:
        render_overview(result)

    with tabs[1]:
        # 预览模式展示 merged_chunks（含 header 合并），流水线模式展示原始 text_chunks
        chunks_to_show = result.get("merged_chunks") if result.get("mode") == "preview" else result.get("text_chunks", [])
        render_text_tab(chunks_to_show, verdicts)

    with tabs[2]:
        render_image_tab(result)

    with tabs[3]:
        render_table_tab(result.get("table_chunks", []), verdicts)

    with tabs[4]:
        render_reconstruction_tab(result)

    with tabs[5]:
        if paper_id:
            render_search_tab(paper_id)
        else:
            st.info("无论文 ID，无法检索")


if __name__ == "__main__":
    main()
