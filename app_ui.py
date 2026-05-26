"""
SortPaper UI — Streamlit 渲染组件。
不包含业务逻辑，不修改数据。
"""
from __future__ import annotations

import streamlit as st
from app_utils import qdrant_search, load_saved_list, load_saved_detail, delete_saved

def type_badge(ctype: str) -> str:
    colors = {"text": "🔵", "table": "🟠", "image": "🟢"}
    return colors.get(ctype, "⚪")


def verdict_badge(passed: bool) -> str:
    return "✅" if passed else "❌"



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
        ["按内容类型", "按栏位", "按表格解析状态"],
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
    table_debug_colors = {
        "manual": ("#D32F2F", "#FFCDD2"),
        "vision_pending": ("#F57C00", "#FFE0B2"),
        "vision_attempted": ("#8E24AA", "#E1BEE7"),
        "matched": ("#2E7D32", "#C8E6C9"),
        "table_no_region": ("#6D4C41", "#D7CCC8"),
        "other": ("#607D8B", "#CFD8DC"),
    }

    def get_color(chunk: dict) -> tuple[str, str]:
        if color_mode == "按内容类型":
            return type_colors.get(chunk.get("content_type", "text"), ("#888", "#ddd"))
        if color_mode == "按表格解析状态":
            if chunk.get("content_type") != "table":
                return ("#B0BEC5", "#ECEFF1")
            status = _table_debug_status(chunk.get("metadata", {}))
            return table_debug_colors.get(status, table_debug_colors["other"])
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
        if color_mode == "按表格解析状态" and ctype == "table":
            status_label = {
                "manual": "NO-LLM",
                "vision_pending": "VISION",
                "vision_attempted": "VISION-DONE",
                "matched": "OK",
                "table_no_region": "NO-REGION",
                "other": "TABLE",
            }.get(_table_debug_status(chunk.get("metadata", {})), "TABLE")
            type_tag = status_label

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

    st.image(img, width="stretch")

    # 图例
    if color_mode == "按内容类型":
        c1, c2, c3 = st.columns(3)
        for col, (ctype, (clr, _)) in zip([c1, c2, c3], type_colors.items()):
            col.markdown(
                f"<span style='display:inline-block;width:14px;height:14px;"                f"background:{clr};border-radius:3px;margin-right:6px;'></span>{ctype}",
                unsafe_allow_html=True,
            )
    else:
        if color_mode == "按表格解析状态":
            labels = {
                "matched": "已解析且匹配候选",
                "vision_pending": "结构质量需处理",
                "vision_attempted": "历史视觉兜底记录",
                "manual": "未解析候选，保留诊断",
                "table_no_region": "表格无候选",
                "other": "其他",
            }
            cols = st.columns(len(labels))
            for col, (key, label) in zip(cols, labels.items()):
                clr, _ = table_debug_colors[key]
                col.markdown(
                    f"<span style='display:inline-block;width:14px;height:14px;"
                    f"background:{clr};border-radius:3px;margin-right:6px;'></span>{label}",
                    unsafe_allow_html=True,
                )
        else:
            c1, c2, c3 = st.columns(3)
            labels = {0: "左栏", 1: "右栏", 2: "通栏"}
            for col, (col_id, (clr, _)) in zip([c1, c2, c3], column_colors.items()):
                col.markdown(
                    f"<span style='display:inline-block;width:14px;height:14px;"
                    f"background:{clr};border-radius:3px;margin-right:6px;'></span>{labels[col_id]}",
                    unsafe_allow_html=True,
                )

    if color_mode == "按表格解析状态":
        render_reconstruction_table_debug(page_chunks)

    st.caption(f"共 {len(page_chunks)} 个 chunks | 页面 {page_num}")


def _table_debug_status(metadata: dict) -> str:
    if metadata.get("table_region_unparsed") or metadata.get("structure_reparse_needed"):
        return "manual"
    if metadata.get("vision_fallback_attempted"):
        return "vision_attempted"
    if metadata.get("vision_fallback_needed"):
        return "vision_pending"
    if metadata.get("table_region_match") == "matched":
        return "matched"
    if metadata.get("parser") and not metadata.get("table_region"):
        return "table_no_region"
    return "other"


def _table_llm_action(metadata: dict) -> tuple[str, str]:
    if metadata.get("structure_reparse_needed") or metadata.get("vision_fallback_disabled"):
        return "structure_reparse", "结构质量低；走 LLM Judge/自动动作，不调用 Vision 读表"
    if metadata.get("table_region_unparsed") or metadata.get("manual_review_needed"):
        return "manual_review", "未解析候选：保留候选并记录诊断，不直接删除"
    if metadata.get("vision_fallback_attempted"):
        if metadata.get("vision_fallback_succeeded"):
            return "vision_called", "历史视觉兜底记录：修复成功"
        return "vision_called", "历史视觉兜底记录：未得到可用表格"
    if metadata.get("vision_fallback_needed"):
        return "vision_pending", "结构质量低：不调用 Vision 读表，由 Judge 决定策略"
    return "none", "未触发 LLM Judge"


def render_reconstruction_table_debug(page_chunks: list[dict]) -> None:
    table_chunks = [c for c in page_chunks if c.get("content_type") == "table"]
    if not table_chunks:
        st.info("当前页没有表格候选或表格块。")
        return

    rows = collect_table_debug_rows(table_chunks)
    unparsed = sum(1 for row in rows if row["unparsed"])
    quality_flagged = sum(1 for row in rows if row["vision_needed"])
    legacy_vision_attempted = sum(1 for row in rows if row["llm_action"] == "vision_called")
    no_judge = sum(1 for row in rows if row["llm_action"] in {"none"})
    matched = sum(1 for row in rows if row["match"] == "matched")

    with st.expander(
        f"当前页表格调试：{len(rows)} 个 | 已匹配 {matched} | 未解析 {unparsed} | "
        f"结构需处理 {quality_flagged} | 历史视觉记录 {legacy_vision_attempted} | 未触发 Judge {no_judge}",
        expanded=bool(unparsed or quality_flagged),
    ):
        for row in rows:
            status = row["llm_label"] if row["llm_action"] != "none" else "已解析"
            st.markdown(
                f"**p{row['page']} | {status} | {row['parser']} | "
                f"{row['region_band']} {row['region_score'] if row['region_score'] is not None else ''}**"
            )
            st.caption(f"bbox: {row['bbox']} | chunk_id: `{row['chunk_id']}`")
            render_table_debug_summary(row["metadata"], inline=True)
            st.divider()

def render_chunk_card(chunk: dict, verdict: dict | None = None, index: int = 0) -> None:
    ctype = chunk.get("content_type", "text")
    badge = type_badge(ctype)
    page = chunk.get("page", "?")
    cid = chunk.get("chunk_id", "")
    short_id = cid.split("_", 2)[-1] if "_" in cid else cid
    metadata = chunk.get("metadata", {})

    if ctype == "table":
        excluded = bool(metadata.get("excluded_from_storage") or (verdict and verdict.get("issue_type") == "false_positive"))
        unparsed = bool(metadata.get("table_region_unparsed"))
        if excluded:
            status_icon = "❌"
            status_text = "不入库候选"
        elif unparsed:
            status_icon = "⚠️"
            status_text = "未解析候选"
        else:
            status_icon = "✅"
            status_text = "可用表格"
        score = metadata.get("table_candidate_score")
        if score is None:
            score = (metadata.get("table_region") or {}).get("confidence")
        score_text = f" candidate={float(score):.2f}" if score is not None else ""
        judge_text = f" judge={verdict.get('score', 0):.2f}" if verdict else ""
        label = f"{status_icon} {badge} p{page} · {status_text} · {short_id}{score_text}{judge_text}"
    else:
        label = f"{badge} p{page} · {short_id}"
    if verdict and ctype != "table":
        label = f"{verdict_badge(verdict['passed'])} {label}  score={verdict.get('score', 0):.2f}"

    with st.expander(label, expanded=False):
        content = chunk.get("raw_content", "")
        if verdict and verdict.get("issue_type") == "false_positive":
            st.error("Judge 已判定该块不是表格；会标记为不入库，但仍保留候选记录。")
        if metadata.get("table_region_unparsed"):
            st.warning("检测到疑似表格区域，但结构解析器没有产出可用表格；保留候选并交给规则/LLM Judge 诊断。")
        if metadata.get("structure_reparse_needed"):
            st.warning("该表格结构质量偏低；不会调用 Vision 读表，会优先使用 LLM Judge 选择重试或 bbox 候选。")
        if metadata.get("vision_fallback_needed"):
            st.warning("该表格结构质量偏低；当前默认不会用 Vision 生成表格内容，会由 Judge/候选竞争处理。")
            reasons = metadata.get("vision_fallback_reasons", [])
            quality = metadata.get("structural_quality", {})
            if reasons or quality:
                reason_text = ", ".join(reasons) if reasons else "unknown"
                st.caption(
                    "quality flags: "
                    f"reasons={reason_text}; "
                    f"consistency={quality.get('consistency_score', '?')}; "
                    f"fill={quality.get('fill_rate', '?')}"
                )
        if metadata.get("vision_needed"):
            st.warning("预览模式仅标记图片位置，未调用 GPT Vision 生成描述。")
        if ctype == "table":
            st.markdown(content)
        elif ctype == "image":
            st.markdown(f"**描述：** {content}")
        else:
            st.text(content[:1200] + ("…" if len(content) > 1200 else ""))

        col1, col2, col3 = st.columns(3)
        col1.caption(f"类型: {ctype}  [{metadata.get('parser', chunk.get('parser', '?'))}]")
        col2.caption(f"页: {page}")
        col3.caption(f"列: {chunk.get('column', '?')}")

        if ctype == "table":
            render_table_debug_summary(metadata, inline=True)

        if verdict:
            fb = verdict.get("feedback", "")
            if fb:
                st.caption(f"**Judge 意见：** {fb}")


def render_table_debug_summary(metadata: dict, *, inline: bool = False) -> None:
    """Render compact table parsing diagnostics inside a chunk card."""
    region = metadata.get("table_region")
    attempts = metadata.get("region_extraction_attempts", [])
    quality = metadata.get("structural_quality", {})
    extraction = metadata.get("extraction_attempt", {})
    dedup_replaced = metadata.get("dedup_replaced", [])

    if not any([region, attempts, quality, extraction, dedup_replaced]):
        return

    debug_container = st.container() if inline else st.expander("表格解析调试", expanded=False)
    with debug_container:
        llm_action, llm_label = _table_llm_action(metadata)
        st.caption(f"LLM 动作: {llm_label}")
        render_table_judge_diagnostics(metadata)

        if region:
            c1, c2, c3 = st.columns(3)
            c1.metric("候选区域", region.get("band", "?"))
            c2.metric("证据分", f"{region.get('confidence', 0):.2f}")
            c3.metric("匹配状态", metadata.get("table_region_match", "?"))
            st.caption(f"region_id: `{region.get('region_id', '?')}`")
            st.caption(f"bbox: {region.get('bbox', '?')}")

            evidence = region.get("evidence", [])
            if evidence:
                st.markdown("**候选区域证据**")
                for item in evidence:
                    score = item.get("score", 0)
                    source = item.get("source", "?")
                    reason = item.get("reason", "")
                    st.caption(f"- {source}: {score:+.2f} | {reason}")

        if extraction:
            st.markdown("**成功解析**")
            st.caption(
                f"{extraction.get('parser', '?')} | "
                f"rows={extraction.get('rows', '?')} | "
                f"cols={extraction.get('cols', '?')} | "
                f"bbox={extraction.get('bbox', '?')}"
            )

        if attempts:
            st.markdown("**候选区解析尝试**")
            for attempt in attempts:
                ok = "成功" if attempt.get("succeeded") else "失败"
                parser = attempt.get("parser", "?")
                rows = attempt.get("rows", 0)
                cols = attempt.get("cols", 0)
                reason = attempt.get("failure_reason", "")
                line = f"- {parser}: {ok}"
                if attempt.get("succeeded"):
                    line += f" | {rows}x{cols}"
                if reason:
                    line += f" | {reason}"
                st.caption(line)

        if dedup_replaced:
            st.markdown("**去重移除的子集/重复表格**")
            for item in dedup_replaced:
                st.caption(
                    f"- {item.get('parser', '?')} | {item.get('reason', '?')}="
                    f"{item.get('score', '?')} | bbox={item.get('bbox', '?')}"
                )

        if quality:
            st.markdown("**结构质量**")
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("一致性", f"{quality.get('consistency_score', 0):.2f}")
            q2.metric("填充率", f"{quality.get('fill_rate', 0):.2f}")
            q3.metric("正文比例", f"{quality.get('prose_cell_ratio', 0):.2f}")
            q4.metric("质量标记", "需处理" if quality.get("fallback_to_vision") else "通过")
            reasons = quality.get("fallback_reasons", [])
            if reasons:
                st.caption("quality reasons: " + ", ".join(reasons))


def render_table_judge_diagnostics(metadata: dict) -> None:
    rule_category = metadata.get("rule_failure_category")
    rule_action = metadata.get("rule_recommended_action")
    llm_category = metadata.get("llm_failure_category")
    llm_action = metadata.get("llm_recommended_action")
    llm_error = metadata.get("llm_error")
    if not any([
        rule_category,
        rule_action,
        llm_category,
        llm_action,
        llm_error,
        metadata.get("auto_action"),
        metadata.get("bbox_candidates_tried"),
        metadata.get("excluded_from_storage"),
        metadata.get("storage_exclusion_reason"),
        metadata.get("unparseable_reason"),
    ]):
        return

    st.markdown("**Rule / LLM Judge**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rule", rule_category or "-")
    c2.metric("Action", rule_action or "-")
    c3.metric("Mode", metadata.get("llm_decision_mode", "off"))
    reasons = metadata.get("rule_reasons", [])
    if reasons:
        st.caption("rule reasons: " + " | ".join(str(item) for item in reasons))

    if llm_error:
        st.caption(f"LLM Judge error: {llm_error}")
    elif llm_category or llm_action:
        c4, c5, c6 = st.columns(3)
        c4.metric("LLM", llm_category or "-")
        c5.metric("LLM action", llm_action or "-")
        c6.metric("Agree", "yes" if metadata.get("rule_llm_agree") else "no")
        reason = metadata.get("llm_reason")
        if reason:
            st.caption(f"LLM reason: {reason}")

    if metadata.get("llm_decision_mode") == "shadow":
        st.caption("诊断报告模式：LLM Judge 仅记录建议，不改变解析结果。")
    elif metadata.get("auto_action_allowed"):
        st.caption("默认质量控制：该建议允许执行一次安全自动动作。")

    decision_reason = metadata.get("auto_decision_reason")
    if decision_reason:
        st.caption("auto decision: " + str(decision_reason))

    auto_fields = [
        f"action={metadata.get('auto_action') or '-'}",
        f"attempted={bool(metadata.get('auto_action_attempted'))}",
        f"succeeded={bool(metadata.get('auto_action_succeeded'))}",
    ]
    if metadata.get("excluded_from_storage"):
        auto_fields.append("excluded_from_storage=yes")
    if metadata.get("storage_exclusion_reason"):
        auto_fields.append(f"storage_reason={metadata.get('storage_exclusion_reason')}")
    if metadata.get("unparseable_reason"):
        auto_fields.append(f"unparseable={metadata.get('unparseable_reason')}")
    if metadata.get("auto_action") or metadata.get("auto_action_attempted"):
        st.caption("auto control: " + " | ".join(str(item) for item in auto_fields))

    if metadata.get("bbox_candidates_tried") or metadata.get("bbox_candidate_best"):
        st.caption(
            "bbox candidates: "
            f"tried={len(metadata.get('bbox_candidates_tried') or [])} | "
            f"adopted={bool(metadata.get('bbox_candidate_adopted'))} | "
            f"best={metadata.get('bbox_candidate_best') or {}} | "
            f"reason={metadata.get('bbox_candidate_reason') or '-'}"
        )

    repair_actions = metadata.get("table_repair_actions") or []
    if repair_actions:
        st.caption("table repair: " + " | ".join(str(item) for item in repair_actions))


def collect_table_debug_rows(chunks: list[dict], verdicts: dict | None = None) -> list[dict]:
    verdicts = verdicts or {}
    rows: list[dict] = []
    for chunk in chunks:
        if chunk.get("content_type") != "table":
            continue
        verdict = verdicts.get(chunk.get("chunk_id", ""), {})
        metadata = chunk.get("metadata", {})
        region = metadata.get("table_region") or {}
        quality = metadata.get("structural_quality") or {}
        attempts = metadata.get("region_extraction_attempts") or []
        failed_attempts = [a for a in attempts if not a.get("succeeded")]
        succeeded_attempts = [a for a in attempts if a.get("succeeded")]
        llm_action, llm_label = _table_llm_action(metadata)
        issue_type = verdict.get("issue_type", "")
        if issue_type == "false_positive":
            llm_action = "judge_false_positive"
            llm_label = "Judge 已判定不是表格：标记不入库，候选仍保留"
        storage_decision = {}
        try:
            from src.judge.table_judge import build_storage_decision
            storage_decision = build_storage_decision(
                content_type=chunk.get("content_type", ""),
                metadata=metadata,
            )
        except Exception:
            storage_decision = {}
        excluded = bool(
            issue_type == "false_positive"
            or storage_decision.get("excluded_from_storage", metadata.get("excluded_from_storage"))
        )
        storage_reason = (
            storage_decision.get("storage_exclusion_reason")
            if storage_decision
            else metadata.get("storage_exclusion_reason", "")
        )
        parsed = not bool(metadata.get("table_region_unparsed"))
        usable = bool(parsed and not excluded)
        if excluded:
            table_status = "不入库候选"
        elif not parsed:
            table_status = "未解析候选"
        elif metadata.get("vision_fallback_needed") or metadata.get("structure_reparse_needed"):
            table_status = "结构需处理"
        else:
            table_status = "可用表格"
        rows.append({
            "chunk_id": chunk.get("chunk_id", ""),
            "table_label": metadata.get("table_label") or metadata.get("caption") or "",
            "table_caption": metadata.get("table_caption") or metadata.get("caption") or "",
            "page": chunk.get("page", "?"),
            "parser": metadata.get("parser", "?"),
            "region_id": region.get("region_id", ""),
            "region_band": region.get("band", "none"),
            "region_score": region.get("confidence", None),
            "match": metadata.get("table_region_match", "none"),
            "unparsed": bool(metadata.get("table_region_unparsed")),
            "usable_table": usable,
            "table_status": table_status,
            "vision_needed": bool(metadata.get("vision_fallback_needed")),
            "vision_attempted": bool(metadata.get("vision_fallback_attempted")),
            "llm_action": llm_action,
            "llm_label": llm_label,
            "rule_category": metadata.get("rule_failure_category", ""),
            "rule_action": metadata.get("rule_recommended_action", ""),
            "llm_category": metadata.get("llm_failure_category", ""),
            "llm_recommended_action": metadata.get("llm_recommended_action", ""),
            "llm_decision_mode": metadata.get("llm_decision_mode", "off"),
            "safe_auto_action": bool(metadata.get("safe_auto_action")),
            "human_review_required": bool(metadata.get("human_review_required")),
            "excluded_from_storage": excluded,
            "storage_exclusion_reason": storage_reason or "",
            "table_candidate_score": metadata.get("table_candidate_score", None),
            "table_candidate_tier": metadata.get("table_candidate_tier", ""),
            "table_repair_actions": " | ".join(str(item) for item in (metadata.get("table_repair_actions") or [])),
            "judge_passed": verdict.get("passed", None),
            "judge_issue_type": issue_type,
            "rows": metadata.get("rows", 0),
            "cols": metadata.get("cols", 0),
            "consistency": quality.get("consistency_score", None),
            "fill_rate": quality.get("fill_rate", None),
            "prose_cell_ratio": quality.get("prose_cell_ratio", None),
            "success_attempts": len(succeeded_attempts),
            "failed_attempts": len(failed_attempts),
            "bbox": chunk.get("bbox"),
            "metadata": metadata,
            "raw_content": chunk.get("raw_content", ""),
        })
    return rows


def render_table_debug_tab(result: dict) -> None:
    chunks = [
        c for c in result.get("merged_chunks", [])
        if c.get("content_type") == "table"
    ]
    rows = collect_table_debug_rows(chunks)
    if not rows:
        st.info("暂无表格调试数据。请先运行预览或完整流水线。")
        return

    total = len(rows)
    usable_count = sum(1 for row in rows if row["usable_table"])
    unparsed = sum(1 for row in rows if row["unparsed"])
    non_table = sum(1 for row in rows if row["llm_action"] == "judge_false_positive")
    no_judge = sum(1 for row in rows if row["llm_action"] == "none")
    excluded_storage = sum(1 for row in rows if row["excluded_from_storage"])

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("候选总数", total)
    c2.metric("可用表格", usable_count)
    c3.metric("不入库候选", excluded_storage)
    c4.metric("Judge 非表格", non_table)
    c5.metric("未解析候选", unparsed)
    c6.metric("未触发 Judge", no_judge)

    pages = sorted({row["page"] for row in rows if row["page"] != "?"})
    page_options = ["全部"] + [str(page) for page in pages]
    col_filter1, col_filter2, col_filter3 = st.columns(3)
    page_filter = col_filter1.selectbox("页码", page_options, key="table_debug_page")
    status_filter = col_filter2.selectbox(
        "状态",
        ["全部", "可用表格", "不入库候选", "未解析候选", "结构质量需处理", "历史视觉兜底记录", "未触发 Judge", "无 region"],
        key="table_debug_status",
    )
    band_filter = col_filter3.selectbox(
        "候选强度",
        ["全部", "strong", "gray", "weak", "none"],
        key="table_debug_band",
    )

    filtered = rows
    if page_filter != "全部":
        filtered = [row for row in filtered if str(row["page"]) == page_filter]
    if status_filter == "可用表格":
        filtered = [row for row in filtered if row["usable_table"]]
    elif status_filter == "未解析候选":
        filtered = [row for row in filtered if row["unparsed"]]
    elif status_filter == "结构质量需处理":
        filtered = [row for row in filtered if row["vision_needed"]]
    elif status_filter == "不入库候选":
        filtered = [row for row in filtered if row["excluded_from_storage"]]
    elif status_filter == "历史视觉兜底记录":
        filtered = [row for row in filtered if row["llm_action"] == "vision_called"]
    elif status_filter == "未触发 Judge":
        filtered = [row for row in filtered if row["llm_action"] == "none"]
    elif status_filter == "无 region":
        filtered = [row for row in filtered if row["match"] == "none"]
    if band_filter != "全部":
        filtered = [row for row in filtered if row["region_band"] == band_filter]

    st.caption(f"当前显示 {len(filtered)} / {len(rows)} 条")
    if not filtered:
        return

    try:
        import pandas as pd

        table_rows = [
            {
                "table": row["table_label"],
                "caption": row["table_caption"],
                "page": row["page"],
                "parser": row["parser"],
                "region_band": row["region_band"],
                "score": row["region_score"],
                "match": row["match"],
                "status": row["table_status"],
                "usable": row["usable_table"],
                "unparsed": row["unparsed"],
                "vision": row["vision_needed"],
                "vision_attempted": row["vision_attempted"],
                "llm": row["llm_label"],
                "rule": row["rule_category"],
                "rule_action": row["rule_action"],
                "llm_judge": row["llm_category"],
                "llm_action": row["llm_recommended_action"],
                "llm_mode": row["llm_decision_mode"],
                "safe_auto": row["safe_auto_action"],
                "human_review": row["human_review_required"],
                "storage": "excluded" if row["excluded_from_storage"] else "storable",
                "storage_reason": row["storage_exclusion_reason"],
                "candidate_score": row["table_candidate_score"],
                "candidate_tier": row["table_candidate_tier"],
                "repair": row["table_repair_actions"],
                "rows": row["rows"],
                "cols": row["cols"],
                "consistency": row["consistency"],
                "fill": row["fill_rate"],
                "prose": row["prose_cell_ratio"],
                "failed_attempts": row["failed_attempts"],
                "bbox": row["bbox"],
            }
            for row in filtered
        ]
        st.dataframe(pd.DataFrame(table_rows), width="stretch", hide_index=True)
    except Exception:
        pass

    for idx, row in enumerate(filtered):
        title = (
            f"p{row['page']} | {row['parser']} | "
            f"{row['region_band']} {row['region_score'] if row['region_score'] is not None else ''} | "
            f"{row['table_status']}"
        )
        with st.expander(title, expanded=bool(row["unparsed"] or row["vision_needed"])):
            st.caption(f"chunk_id: `{row['chunk_id']}`")
            st.caption(f"bbox: {row['bbox']}")
            render_table_debug_summary(row["metadata"], inline=True)
            content = row["raw_content"]
            if content:
                st.markdown("**解析内容预览**")
                st.markdown(content[:2000])


def render_overview(result: dict) -> None:
    """纯展示，不包含任何会触发 st.rerun() 的按钮。"""
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
        quality = result.get("quality", {})
        has_quality = bool(quality.get("category"))

        if status == "done" and not has_quality:
            st.success("✅ 解析完成")
        elif status == "done" and has_quality:
            st.success("✅ 质量评估完成")
            st.caption(f"分类: {quality.get('category')} | 可信度: {quality.get('credibility', 0):.2f}")

        # 计时面板（展示，无按钮）
        quality = result.get("quality", {})
        timing = quality.get("timing", {})
        worker_timing = result.get("worker_timing", {})
        if timing or worker_timing:
            with st.expander("⏱️ 耗时统计", expanded=False):
                # 解析阶段
                wt_text = worker_timing.get("text", 0)
                wt_table = worker_timing.get("table", 0)
                wt_image = worker_timing.get("image", 0)
                parse_total = max(wt_text, wt_table, wt_image)  # Worker 并行，取最长
                st.caption("**解析阶段**（三个 Worker 并行）")
                pc1, pc2, pc3, pc4 = st.columns(4)
                pc1.metric("文本解析", f"{wt_text:.1f}s")
                pc2.metric("表格解析", f"{wt_table:.1f}s")
                pc3.metric("图片解析", f"{wt_image:.1f}s")
                pc4.metric("并行耗时", f"{parse_total:.1f}s")

                # Judge 阶段
                jt = result.get("judge_timing", {})
                if jt:
                    st.caption("**Judge 阶段**（text / table / image 分线并行）")
                    jc1, jc2, jc3, jc4 = st.columns(4)
                    jc1.metric("文本 Judge", f"{jt.get('text', 0):.1f}s")
                    jc2.metric("表格 Judge", f"{jt.get('table', 0):.1f}s")
                    jc3.metric("图片 Judge", f"{jt.get('image', 0):.1f}s")
                    jc4.metric("Judge 合计", f"{sum(jt.values()):.1f}s")

                # Merge 阶段
                mt = result.get("merge_timing", 0)
                if mt:
                    st.caption(f"**Merge 阶段**: {mt:.1f}s")

                # 描述生成阶段（表格/图片 DeepSeek 前缀，pipeline 结束后串行调用）
                dt = result.get("desc_timing", 0)
                if dt:
                    desc_n = sum(
                        1 for c in result.get("merged_chunks", [])
                        if c.get("content_type") in ("table", "image")
                    )
                    dc1, dc2 = st.columns(2)
                    dc1.metric("描述生成（表/图前缀）", f"{dt:.1f}s")
                    dc2.metric("  ", f"{desc_n} 个 chunk × DeepSeek 并行")

                # 图片子阶段（image_extract / image_describe）
                if worker_timing.get("image_describe", 0) > 0:
                    st.caption("图片解析详情：提取 {:.1f}s + VL描述 {:.1f}s".format(
                        worker_timing.get("image_extract", 0),
                        worker_timing.get("image_describe", 0),
                    ))

                # 总计
                total_elapsed = result.get("_elapsed", 0)
                if total_elapsed:
                    accounted = parse_total + sum(jt.values()) + mt + dt
                    if timing:
                        accounted += sum(timing.values())
                    st.caption(f"**总耗时 {total_elapsed:.0f} 秒** — 已统计 {accounted:.0f}s，其余为路由/通信开销")

                # 质量评估阶段
                t_cls = timing.get("classify", 0)
                t_map = timing.get("map", 0)
                t_red = timing.get("reduce", 0)
                if t_cls or t_map or t_red:
                    st.caption("**质量评估阶段**")
                    qc1, qc2, qc3, qc4 = st.columns(4)
                    qc1.metric("分类", f"{t_cls:.1f}s")
                    qc2.metric("Map", f"{t_map:.1f}s")
                    qc3.metric("Reduce", f"{t_red:.1f}s")
                    qc4.metric("合计", f"{t_cls + t_map + t_red:.1f}s")
    else:
        verdicts = result.get("verdicts", {})
        if verdicts:
            passed = sum(1 for v in verdicts.values() if v["passed"])
            total_v = len(verdicts)
            c4.metric("Judge 通过", f"{passed} / {total_v}")
        else:
            c4.metric("模式", "快速预览")
        st.info("ℹ️ 快速预览模式：PyMuPDF 文本 + pdfplumber 表格区域发现/解析；仅表格质量诊断会调用 LLM Judge，不执行文本 Judge、图片 VisionParser 或向量索引写入")

        # 计时面板（快速预览）
        worker_timing = result.get("worker_timing", {})
        judge_timing = result.get("judge_timing", {})
        if worker_timing or judge_timing:
            with st.expander("⏱️ 耗时统计", expanded=False):
                wt_text = worker_timing.get("text", 0)
                wt_table = worker_timing.get("table", 0)
                if wt_text or wt_table:
                    st.caption("**解析阶段**（文本+表格串行）")
                    pc1, pc2, pc3 = st.columns(3)
                    pc1.metric("文本解析", f"{wt_text:.1f}s")
                    pc2.metric("表格解析", f"{wt_table:.1f}s")
                    pc3.metric("合计", f"{wt_text + wt_table:.1f}s")

                jt_total = sum(judge_timing.values())
                if jt_total:
                    st.caption(f"**Judge 阶段**（仅表格，含 DeepSeek 调用）: {jt_total:.1f}s")

                total_elapsed = result.get("_elapsed", 0)
                if total_elapsed:
                    accounted = wt_text + wt_table + jt_total
                    st.caption(f"**总耗时 {total_elapsed:.0f} 秒** — 已统计 {accounted:.0f}s，其余为路由 / 网络开销")


def render_text_tab(chunks: list[dict], verdicts: dict) -> None:
    if not chunks:
        st.info("无文本块")
        return
    st.caption(f"共 {len(chunks)} 个文本块")
    for i, c in enumerate(chunks):
        verdict = verdicts.get(c["chunk_id"])
        render_chunk_card(c, verdict, i)



def render_table_tab(
    chunks: list[dict],
    verdicts: dict,
) -> None:
    # ── 表格内容块 ────────────────────────────────────────────────────────────
    if not chunks:
        st.info("未检测到表格（pdfplumber 未找到有效表格结构）")
        return

    rows = collect_table_debug_rows(chunks, verdicts)
    usable_count = sum(1 for row in rows if row["usable_table"])
    excluded_count = sum(1 for row in rows if row["excluded_from_storage"])
    false_positive = sum(1 for row in rows if row["llm_action"] == "judge_false_positive")
    quality_flagged = sum(1 for row in rows if row["llm_action"] == "vision_pending")
    no_judge = sum(1 for row in rows if row["llm_action"] == "none")
    manual_review = sum(1 for row in rows if row["llm_action"] == "manual_review")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("候选总数", len(chunks))
    c2.metric("可用表格", usable_count)
    c3.metric("不入库候选", excluded_count)
    c4.metric("Judge 非表格", false_positive)
    c5.metric("未触发 Judge", no_judge)

    if manual_review:
        st.warning(f"{manual_review} 个未解析候选会保留诊断记录；不会被物理删除。")

    _render_table_region_inventory(rows)

    status_filter = st.selectbox(
        "表格质量状态",
        ["可用表格", "全部", "不入库候选", "Judge 判定不是表格", "结构质量需处理", "未触发 Judge", "未解析候选"],
        key="table_tab_status_v2",
    )

    filtered_chunks = chunks
    if status_filter:
        row_by_id = {row["chunk_id"]: row for row in rows}

        def _keep(chunk: dict) -> bool:
            row = row_by_id.get(chunk.get("chunk_id", ""))
            if not row:
                return False
            if status_filter == "全部":
                return True
            if status_filter == "可用表格":
                return row["usable_table"]
            if status_filter == "不入库候选":
                return row["excluded_from_storage"]
            if status_filter == "Judge 判定不是表格":
                return row["llm_action"] == "judge_false_positive"
            if status_filter == "结构质量需处理":
                return row["llm_action"] == "vision_pending"
            if status_filter == "未触发 Judge":
                return row["llm_action"] == "none"
            if status_filter == "未解析候选":
                return row["llm_action"] == "manual_review"
            return True

        filtered_chunks = [chunk for chunk in chunks if _keep(chunk)]

    st.caption(f"当前显示 {len(filtered_chunks)} / {len(chunks)} 个表格块")
    for i, c in enumerate(filtered_chunks):
        verdict = verdicts.get(c["chunk_id"])
        render_chunk_card(c, verdict, i)


def _render_table_region_inventory(rows: list[dict]) -> None:
    if not rows:
        return
    try:
        import pandas as pd
    except Exception:
        return

    inventory_rows = [
        {
            "table": row.get("table_label", ""),
            "caption": row.get("table_caption", ""),
            "page": row.get("page", "?"),
            "bbox": row.get("bbox"),
            "score": row.get("region_score"),
            "band": row.get("region_band"),
            "status": row.get("table_status", ""),
            "storage": "excluded" if row.get("excluded_from_storage") else "storable",
            "parser": row.get("parser", "?"),
        }
        for row in rows
    ]
    with st.expander("表格区域清单", expanded=True):
        st.dataframe(pd.DataFrame(inventory_rows), width="stretch", hide_index=True)


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
    """语义检索标签页：当前论文内检索。"""
    try:
        from src.store.qdrant_store import QdrantStore
        store = QdrantStore()
        total = store.count
    except Exception:
        st.warning("Qdrant 服务未连接（localhost:6333），请先启动 Qdrant")
        return

    if total == 0:
        st.info("Qdrant 索引为空，请先解析论文并入库")
        return

    st.caption(f"当前索引共 {total} 条记录，当前论文: `{paper_id}`")

    sub_tabs = st.tabs(["🔍 手动检索", "🤖 Agent 检索"])

    with sub_tabs[0]:
        _render_manual_search(paper_id)

    with sub_tabs[1]:
        _render_agent_search(paper_id)


def render_standalone_search_tab() -> None:
    """独立检索标签页：不需要先解析论文，直接搜索 Qdrant 全库。"""
    try:
        from src.store.qdrant_store import QdrantStore
        store = QdrantStore()
        total = store.count
    except Exception:
        st.warning("Qdrant 服务未连接（localhost:6333），请先启动 Qdrant")
        return

    if total == 0:
        st.info("Qdrant 索引为空，请先解析论文并入库后再来检索。")
        return

    st.caption(f"全库共 {total} 条记录，跨论文检索")

    st_tabs = st.tabs(["🔍 手动检索", "🤖 Agent 检索"])

    with st_tabs[0]:
        _render_manual_search(paper_id=None)

    with st_tabs[1]:
        _render_agent_search(paper_id=None)


def _render_manual_search(paper_id: str | None = None) -> None:
    query = st.text_input("输入检索关键词或问题", placeholder="e.g. lacto-N-tetraose biosynthesis yield",
                          key="manual_query")
    top_k = st.slider("返回条数", 1, 10, 5, key="manual_top_k")
    use_quality_filter = st.checkbox(
        "仅检索质量解析后的可执行实验论文",
        value=False,
        help="过滤掉综述、专利和低行动价值 chunk；需要全库探索时可取消勾选",
        key="manual_quality_filter",
    )
    use_rerank = st.checkbox("启用 Rerank（qwen3-rerank 二次排序）", value=True,
                              help="先 Hybrid 召回，再用 Rerank 模型精排", key="manual_rerank")
    use_lexical_backfill = st.checkbox(
        "增强 chunk 召回（实验）",
        value=False,
        help="额外补入词面匹配候选，可能提升证据 chunk 召回，但首次检索会更慢",
        key="manual_lexical_backfill",
    )
    use_query_rewrite = st.checkbox(
        "查询改写（DeepSeek V4 Flash）",
        value=False,
        help="先把中文/口语问题标准化为英文科学检索表达，再进入向量与词面检索",
        key="manual_query_rewrite",
    )
    use_multi_query = st.checkbox(
        "多路召回（实验）",
        value=False,
        help="使用原问题、标准化问题和少量变体并行召回，再合并去重排序",
        key="manual_multi_query",
    )
    if st.button("🔍 检索", key="manual_search_btn"):
        if not query.strip():
            st.warning("请输入查询内容")
            return
        spinner_text = "Hybrid 检索 + Rerank 中…" if use_rerank else "Hybrid 检索中…"
        if use_lexical_backfill:
            spinner_text = spinner_text.replace("中…", " + 增强召回中…")
        filter_kwargs = {"is_actionable": True} if use_quality_filter else None
        with st.spinner(spinner_text):
            results = qdrant_search(
                query,
                paper_id,
                top_k,
                rerank=use_rerank,
                filter_kwargs=filter_kwargs,
                lexical_backfill=use_lexical_backfill,
                query_rewrite=use_query_rewrite,
                multi_query=use_multi_query,
            )
        if not results:
            st.warning("未找到相关结果")
            return
        st.caption(f"找到 {len(results)} 条结果")
        for r in results:
            score = r.get("score", 0)
            payload = r.get("payload", {})
            wtype = payload.get("content_type", payload.get("worker_type", ""))
            content = payload.get("content", "")
            pg = payload.get("page", "?")
            paper_title = payload.get("paper_title", "?")
            category = payload.get("category", "")
            credibility = payload.get("credibility")
            fermentation_relevance = payload.get("fermentation_relevance")
            products = payload.get("target_products") or []
            organisms = payload.get("organisms") or []
            reranked = "🔁 " if r.get("reranked") else ""
            paper_info = f" | {paper_title[:40]}" if paper_id is None else ""
            with st.expander(f"{reranked}📌 相似度={score:.4f} | {wtype} | p{pg}{paper_info}", expanded=True):
                quality_bits = []
                if category:
                    quality_bits.append(f"分类: {category_label(category)}")
                if credibility is not None:
                    quality_bits.append(f"可信度: {float(credibility):.2f}")
                if fermentation_relevance is not None:
                    quality_bits.append(f"发酵相关性: {float(fermentation_relevance):.2f}")
                if products:
                    quality_bits.append("产品: " + ", ".join(map(str, products[:4])))
                if organisms:
                    quality_bits.append("菌株/宿主: " + ", ".join(map(str, organisms[:4])))
                if quality_bits:
                    st.caption(" | ".join(quality_bits))
                if wtype == "table":
                    st.markdown(content)
                else:
                    st.text(content)


def _render_agent_search(paper_id: str | None = None) -> None:
    st.caption("Qwen 自主检索 + 综合建议。模型会自动选择关键词和过滤条件。" +
               ("（全库检索）" if paper_id is None else f"（限定论文: `{paper_id}`）"))
    question = st.text_area("输入你的问题", placeholder="e.g. HMO 发酵中乳糖补料导致乙酸积累，文献中有什么解决方案？",
                            key="agent_question", height=100)
    max_rounds = st.slider("最大检索轮次", 1, 3, 2, key="agent_rounds",
                            help="每轮可调用一次 search_literature，多轮可换角度再查")
    use_lexical_backfill = st.checkbox(
        "增强 chunk 召回（实验）",
        value=False,
        help="Agent 检索时额外补入词面匹配候选，可能提升证据 chunk 召回，但首次检索会更慢",
        key="agent_lexical_backfill",
    )
    use_query_rewrite = st.checkbox(
        "Agent 查询改写（DeepSeek V4 Flash）",
        value=False,
        help="对 Agent 生成的检索词再做一次领域标准化，减少口语表达和别名导致的漏召回",
        key="agent_query_rewrite",
    )
    use_multi_query = st.checkbox(
        "Agent 多路召回（实验）",
        value=False,
        help="每轮工具检索使用原检索词、标准化检索词和少量变体并行召回",
        key="agent_multi_query",
    )

    if st.button("🤖 Agent 检索", key="agent_search_btn", type="primary"):
        if not question.strip():
            st.warning("请输入问题")
            return

        from src.agent.literature_agent import LiteratureAgent

        agent = LiteratureAgent(
            lexical_backfill=use_lexical_backfill,
            use_query_rewrite=use_query_rewrite,
            multi_query_recall=use_multi_query,
        )
        with st.spinner("Agent 分析中（检索 + 综合建议）…"):
            result = agent.query(question, max_rounds=max_rounds)

        # 检索历史
        if result["search_history"]:
            with st.expander(f"🔍 检索过程（{len(result['search_history'])} 轮，共 {result['total_chunks']} 个 chunk）", expanded=False):
                for h in result["search_history"]:
                    st.caption(f"第 {h['round'] + 1} 轮: `{h['args'].get('query', '?')}`")
                    if h["args"].get("category"):
                        st.caption(f"  过滤: category={h['args']['category']}")
                    st.caption(f"  命中: {h['hits']} 条")

        # 综合建议
        st.markdown("### 🧠 综合建议")
        st.markdown(result["answer"])


# ─── 已保存结果 ────────────────────────────────────────────────────────────────


def category_label(cat: str) -> str:
    labels = {
        "fermentation_experiment": "🧪 发酵实验",
        "biosynthesis_review": "📖 合成生物学综述",
        "other": "📎 其他",
    }
    return labels.get(cat, cat)


def render_saved_tab() -> None:
    """已保存结果标签页：列表 + 详情。"""
    saved_list = load_saved_list()

    if not saved_list:
        st.info("暂无已保存的解析结果。运行完整流水线后点击\"💾 保存解析结果\"即可保存。")
        return

    # 如果当前选中项不在列表中（被删除），清除选中
    if st.session_state.selected_saved and st.session_state.selected_saved not in {
        s["file_stem"] for s in saved_list
    }:
        st.session_state.selected_saved = None

    # 列表 + 详情布局
    col_list, col_detail = st.columns([1, 2])

    with col_list:
        st.caption(f"共 {len(saved_list)} 条已保存结果")

        # 清空所有已保存
        if "confirm_clear_saved" not in st.session_state:
            st.session_state.confirm_clear_saved = False
        if not st.session_state.confirm_clear_saved:
            st.button("🗑️ 清空所有已保存", key="clear_all_saved", width="stretch",
                      on_click=lambda: setattr(st.session_state, "confirm_clear_saved", True))
        else:
            st.warning("⚠️ 将删除所有已保存结果（不含向量库数据）")
            c1, c2 = st.columns(2)
            if c1.button("✅ 确认", width="stretch"):
                for item in saved_list:
                    delete_saved(item["file_stem"])
                st.session_state.confirm_clear_saved = False
                st.session_state.selected_saved = None
                st.rerun()
            if c2.button("❌ 取消", width="stretch"):
                st.session_state.confirm_clear_saved = False
                st.rerun()

        for item in saved_list:
            title = item.get("paper_title", "?")
            cat = item.get("category", "?")
            cred = item.get("credibility", 0)
            passed = item.get("passed", 0)
            failed = item.get("failed", 0)
            saved_at = item.get("saved_at", "")[:16].replace("T", " ")

            # 卡片
            selected = st.session_state.selected_saved == item["file_stem"]
            card_style = "border: 2px solid #2196F3; border-radius: 8px; padding: 10px;" if selected else ""
            with st.container():
                if st.button(
                    f"{category_label(cat)}\n**{title[:60]}**\n可信度: {cred:.2f} | ✅{passed} ❌{failed}\n{saved_at}",
                    key=f"saved_{item['file_stem']}",
                    width="stretch",
                ):
                    st.session_state.selected_saved = item["file_stem"]
                    st.rerun()

    with col_detail:
        if st.session_state.selected_saved:
            detail = load_saved_detail(st.session_state.selected_saved)
            if detail:
                render_saved_detail(detail, st.session_state.selected_saved)
        else:
            st.info("← 选择一条已保存结果查看详情")


def render_saved_detail(detail: dict, file_stem: str) -> None:
    """已保存结果的详细视图。"""
    quality = detail.get("quality", {})
    stats = detail.get("stats", {})
    verdicts = detail.get("verdicts", {})
    chunks = detail.get("chunks", [])
    chunk_map = quality.get("chunk_map", {})

    st.subheader(quality.get("paper_title", file_stem))
    st.caption(f"保存时间: {detail.get('saved_at', '')[:19].replace('T', ' ')} | "
               f"文件: {detail.get('filename', file_stem)} | ID: `{detail.get('paper_id', '?')[:16]}`")

    if st.button("🗑️ 删除此结果", key=f"del_{file_stem}", type="secondary"):
        delete_saved(file_stem)
        st.session_state.selected_saved = None
        st.rerun()

    # 子标签页
    sub_tabs = st.tabs(["📊 概览", "📝 文本块", "📋 表格", "🖼️ 图片", "🔍 质量详情"])

    # ── 概览 ──
    with sub_tabs[0]:
        cat = quality.get("category", "?")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("分类", category_label(cat))
        c2.metric("可信度", f"{quality.get('credibility', 0):.2f}")
        c3.metric("发酵相关性", f"{quality.get('fermentation_relevance', 0):.2f}")
        c4.metric("自洽性", quality.get("classify_status", "?"))

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("总 Chunks", stats.get("total", 0))
        c6.metric("✅ 通过", stats.get("passed", 0))
        c7.metric("❌ 失败", stats.get("failed", 0))
        c8.metric("状态", stats.get("status", "?"))

        # 计时
        timing = quality.get("timing", {})
        if timing:
            st.divider()
            st.caption("⏱️ 质量评估耗时")
            t_cls = timing.get("classify", 0)
            t_map = timing.get("map", 0)
            t_red = timing.get("reduce", 0)
            tc1, tc2, tc3, tc4 = st.columns(4)
            tc1.metric("分类", f"{t_cls:.1f}s")
            tc2.metric("Map", f"{t_map:.1f}s")
            tc3.metric("Reduce", f"{t_red:.1f}s")
            tc4.metric("合计", f"{t_cls + t_map + t_red:.1f}s")

        st.divider()

        if quality.get("paper_summary"):
            st.markdown("**论文摘要**")
            st.info(quality["paper_summary"])

        c9, c10 = st.columns(2)
        with c9:
            if quality.get("target_products"):
                st.markdown("**目标产物**")
                for p in quality["target_products"]:
                    st.markdown(f"- {p}")
        with c10:
            if quality.get("organisms"):
                st.markdown("**相关菌株**")
                for o in quality["organisms"]:
                    st.markdown(f"- {o}")

        if quality.get("classify_reason"):
            st.markdown("**分类依据**")
            st.caption(quality["classify_reason"])

        if quality.get("is_actionable"):
            st.success("✅ 包含可指导发酵实验的信息")

        if quality.get("key_evidence"):
            st.markdown("**关键原文证据**")
            for i, ev in enumerate(quality["key_evidence"], 1):
                with st.expander(f"证据 {i}"):
                    st.text(ev)

    # ── 文本块 ──
    with sub_tabs[1]:
        text_chunks = [c for c in chunks if c.get("content_type") == "text"]
        if not text_chunks:
            st.info("无文本块")
        else:
            st.caption(f"共 {len(text_chunks)} 个文本块")
            for c in text_chunks:
                cid = c.get("chunk_id", "")
                verdict = verdicts.get(cid, {})
                passed = verdict.get("passed", True)
                score = verdict.get("score", 0)
                v_badge = "✅" if passed else "❌"
                preview = c.get("raw_content", "")[:80].replace("\n", " ")

                with st.expander(
                    f"{v_badge} p{c.get('page', '?')} · {preview}… | score={score:.2f}",
                    expanded=False,
                ):
                    st.text(c.get("raw_content", "")[:1200])
                    # Map 结果
                    cm = chunk_map.get(cid)
                    if cm:
                        kps = cm.get("key_points", [])
                        ents = cm.get("entities", [])
                        claim = cm.get("contains_claim", False)
                        claim_detail = cm.get("claim_detail", "")
                        if kps:
                            st.caption("**关键要点**")
                            for kp in kps:
                                st.markdown(f"- {kp}")
                        if ents:
                            st.caption(f"**实体**: {', '.join(ents)}")
                        if claim:
                            st.info(f"📌 **声明**: {claim_detail}" if claim_detail else "📌 包含实验/方法声明")
                    # Verdict
                    fb = verdict.get("feedback", "")
                    if fb:
                        st.caption(f"**Judge 意见**: {fb}")

    # ── 表格 ──
    with sub_tabs[2]:
        table_chunks = [c for c in chunks if c.get("content_type") == "table"]
        if not table_chunks:
            st.info("无表格块")
        else:
            st.caption(f"共 {len(table_chunks)} 个表格块")
            for c in table_chunks:
                cid = c.get("chunk_id", "")
                verdict = verdicts.get(cid, {})
                passed = verdict.get("passed", True)
                score = verdict.get("score", 0)
                v_badge = "✅" if passed else "❌"
                with st.expander(
                    f"{v_badge} 表格 p{c.get('page', '?')} | score={score:.2f}",
                    expanded=False,
                ):
                    st.markdown(c.get("raw_content", ""))
                    fb = verdict.get("feedback", "")
                    if fb:
                        st.caption(f"**Judge 意见**: {fb}")

    # ── 图片 ──
    with sub_tabs[3]:
        img_chunks = [c for c in chunks if c.get("content_type") == "image"]
        if not img_chunks:
            st.info("无图片块")
        else:
            st.caption(f"共 {len(img_chunks)} 个图片块")
            for c in img_chunks:
                cid = c.get("chunk_id", "")
                verdict = verdicts.get(cid, {})
                passed = verdict.get("passed", True)
                score = verdict.get("score", 0)
                v_badge = "✅" if passed else "❌"
                with st.expander(
                    f"{v_badge} 图片 p{c.get('page', '?')} | score={score:.2f}",
                    expanded=False,
                ):
                    st.text(c.get("raw_content", "")[:1200])
                    fb = verdict.get("feedback", "")
                    if fb:
                        st.caption(f"**Judge 意见**: {fb}")

    # ── 质量详情 ──
    with sub_tabs[4]:
        classify_result = quality.get("classify_result", {})
        verify_result = quality.get("verify_result", {})

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**分类结果 (Prompt 1)**")
            st.json(classify_result)
        with c2:
            st.markdown("**验证结果 (Prompt 2)**")
            st.json(verify_result)

        if quality.get("classify_reason"):
            st.divider()
            st.markdown("**判断依据**")
            st.info(quality["classify_reason"])



