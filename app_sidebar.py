"""
SortPaper Sidebar — 侧边栏组件。
"""
from __future__ import annotations

import logging

import streamlit as st

from app_config import (
    BATCH_ESTIMATED_MINUTES_PER_PAPER,
    BATCH_MAX_PAPER_WORKERS,
    DEFAULT_QDRANT_COLLECTION,
    DASHSCOPE_EMBEDDING_MODEL,
    EMBEDDING_API_KEY_ENV,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    OPENAI_EMBEDDING_BASE_URL,
    OPENAI_EMBEDDING_MODEL,
    QWEN3_LOCAL_EMBEDDING_MAX_LENGTH,
    QWEN3_LOCAL_EMBEDDING_MODEL_PATH,
    RERANK_DEFAULT_ENABLED,
    RESULTS_DIR,
    SAMPLE_DIR,
)
from app_modes import (
    LEGACY_MODES,
    MODE_LEGACY_AUTO_STORE,
    MODE_LEGACY_PIPELINE,
    MODE_LEGACY_PREVIEW,
    MODE_MINERU,
    MODE_MINERU_AUTO_STORE,
    MODE_MINERU_FULL,
    PRIMARY_MODES,
)
from app_utils import build_paper_id
from src.application import embedding_settings
from src.runtime_env import ENV_PATH

logger = logging.getLogger(__name__)
LOGO = "📄"
EMBEDDING_PROVIDER_OPTIONS = embedding_settings.PROVIDER_OPTIONS


def _env_value(name: str, default: str = "") -> str:
    return embedding_settings.env_value(name, default)


def _env_key_configured(name: str) -> bool:
    return embedding_settings.env_key_configured(name)


def _provider_collection_default(provider: str) -> str:
    return embedding_settings.provider_collection_default(provider)


def _provider_dim_default(provider: str) -> int:
    return embedding_settings.provider_dim_default(provider)


def _format_env_value(value: str) -> str:
    return embedding_settings.format_env_value(value)


def _update_env_text(text: str, updates: dict[str, str]) -> str:
    return embedding_settings.update_env_text(text, updates)


def _save_env_updates(updates: dict[str, str]) -> None:
    embedding_settings.save_env_updates(updates, env_path=ENV_PATH)


def render_parse_mode_controls() -> str:
    st.subheader("解析模式")
    show_legacy_modes = st.checkbox(
        "显示历史/兼容解析入口",
        value=False,
        help="默认只显示 MinerU 主路径；打开后可运行旧 parser 做展示、对照或兼容入库。",
        key="sidebar_show_legacy_modes",
    )
    mode_options = list(PRIMARY_MODES) + (list(LEGACY_MODES) if show_legacy_modes else [])
    mode_captions = {
        MODE_MINERU: "最快：文本/表格/图片占位，不调用图片 VL",
        MODE_MINERU_FULL: "完整：文本/表格 + Figure group 图片转文字",
        MODE_MINERU_AUTO_STORE: "完整解析后直接写入 Qdrant",
        MODE_LEGACY_PREVIEW: "自研旧路径：文本/表格预览，图片仅占位",
        MODE_LEGACY_PIPELINE: "自研旧路径：文本/表格/图片 + Judge；用于对照展示",
        MODE_LEGACY_AUTO_STORE: "兼容旧入库：自研解析 → Judge → Qdrant",
    }
    mode = st.radio(
        "模式",
        mode_options,
        captions=[mode_captions[item] for item in mode_options],
        label_visibility="collapsed",
        key="sidebar_parse_mode",
    )

    if mode == MODE_MINERU:
        st.success("快速预览：调用 MinerU 解析 PDF，保留页码、bbox、表格和 Figure group。")
    elif mode == MODE_MINERU_FULL:
        st.success("完整解析：调用 MinerU，并按 Figure group 调用 VL 生成图片文字描述。")
    elif mode == MODE_MINERU_AUTO_STORE:
        st.success("一键入库：完整解析后直接写入 Qdrant。")
    elif mode == MODE_LEGACY_PREVIEW:
        st.info("历史展示路径：文本/表格预览，图片只生成占位信息。")
    elif mode == MODE_LEGACY_PIPELINE:
        st.info("历史展示路径：文本/表格/图片 + Judge，用于对照展示。")
    elif mode == MODE_LEGACY_AUTO_STORE:
        st.info(f"兼容旧入库路径；向量模型：`{EMBEDDING_PROVIDER}:{EMBEDDING_MODEL}`。")
    return mode


def sidebar(workspace: str = "import") -> dict:
    result: dict = {
        "pdf_bytes": None,
        "paper_id": "",
        "mode": st.session_state.get("sidebar_parse_mode", MODE_MINERU),
        "parse_clicked": False,
        "filename": "",
        "batch": None,
    }

    with st.sidebar:
        st.title(f"{LOGO} SortPaper")
        st.caption("学术论文解析与检索工具")
        st.divider()

        if workspace == "online":
            st.caption("当前工作区：在线获取论文")
            st.subheader("在线获取")
            st.caption("首页：每日简报")
            st.caption("同级入口：手动搜索、自动定时搜索、候选库")
        elif workspace == "import":
            st.caption("当前工作区：导入论文")
            result["mode"] = render_parse_mode_controls()
            st.divider()
            st.subheader("库状态")
            try:
                from src.application import vector_library

                total = vector_library.total_count()
                papers = vector_library.list_papers()
                pending = sum(1 for paper in papers if paper.get("enrichment_status", "done") == "pending")
                st.caption(f"Collection: `{DEFAULT_QDRANT_COLLECTION}`")
                st.caption(f"记录 {total} 条；论文 {len(papers)} 篇；待质量分析 {pending} 篇")
            except Exception:
                st.warning("Qdrant 未连接")
            st.divider()
            _render_embedding_settings_ui()
        elif workspace == "search":
            st.caption("当前工作区：检索召回")
            st.subheader("召回上下文")
            current = st.session_state.get("current_paper") or ""
            st.caption(f"当前论文过滤：`{current}`" if current else "默认全库检索")
            st.caption(f"Rerank 默认：{'开启' if RERANK_DEFAULT_ENABLED else '关闭'}")

    return result


def _render_embedding_settings_ui() -> None:
    provider = EMBEDDING_PROVIDER if EMBEDDING_PROVIDER in EMBEDDING_PROVIDER_OPTIONS else "dashscope"
    with st.expander("Embedding / Rerank 设置", expanded=False):
        current_key_label = EMBEDDING_API_KEY_ENV or "无需 API key"
        key_state = "已配置" if _env_key_configured(EMBEDDING_API_KEY_ENV) else "未配置"
        if not EMBEDDING_API_KEY_ENV:
            key_state = "本地模型无需配置"
        st.caption(
            f"当前生效：`{EMBEDDING_PROVIDER}` / `{EMBEDDING_MODEL}` / "
            f"dim={EMBEDDING_DIM} / collection=`{DEFAULT_QDRANT_COLLECTION}` / "
            f"{current_key_label}: {key_state}"
        )

        selected_provider = st.selectbox(
            "Provider",
            EMBEDDING_PROVIDER_OPTIONS,
            index=EMBEDDING_PROVIDER_OPTIONS.index(provider),
            key="embedding_provider_editor",
        )
        collection = st.text_input(
            "Qdrant collection",
            value=_env_value("SORTPAPER_QDRANT_COLLECTION", _provider_collection_default(selected_provider)),
            help="切换 embedding provider 后需要清空或重建 collection，避免向量空间混用。",
            key="embedding_collection_editor",
        )
        dim = st.number_input(
            "Embedding dim",
            min_value=1,
            max_value=8192,
            value=int(_env_value("SORTPAPER_EMBEDDING_DIM", str(_provider_dim_default(selected_provider))) or 1024),
            step=1,
            key="embedding_dim_editor",
        )

        updates: dict[str, str] = {
            "SORTPAPER_EMBEDDING_PROVIDER": selected_provider,
            "SORTPAPER_EMBEDDING_DIM": str(int(dim)),
            "SORTPAPER_QDRANT_COLLECTION": collection.strip() or _provider_collection_default(selected_provider),
            "SORTPAPER_RERANK_DEFAULT_ENABLED": "true" if RERANK_DEFAULT_ENABLED else "false",
        }

        if selected_provider == "dashscope":
            api_key = st.text_input(
                "DASHSCOPE_API_KEY",
                value="",
                type="password",
                placeholder="留空则保留现有 .env 值",
                key="dashscope_api_key_editor",
            )
            model = st.text_input(
                "DASHSCOPE_EMBEDDING_MODEL",
                value=_env_value("DASHSCOPE_EMBEDDING_MODEL", DASHSCOPE_EMBEDDING_MODEL),
                key="dashscope_embedding_model_editor",
            )
            rerank_default = st.checkbox(
                "默认启用 Rerank",
                value=RERANK_DEFAULT_ENABLED,
                help="Rerank 仍依赖 DashScope key；如果欠费或无 key，请关闭。",
                key="dashscope_rerank_default_editor",
            )
            updates["SORTPAPER_EMBEDDING_API_KEY_ENV"] = "DASHSCOPE_API_KEY"
            updates["DASHSCOPE_EMBEDDING_MODEL"] = model.strip() or "text-embedding-v3"
            updates["SORTPAPER_RERANK_DEFAULT_ENABLED"] = "true" if rerank_default else "false"
            if api_key.strip():
                updates["DASHSCOPE_API_KEY"] = api_key.strip()
        elif selected_provider == "openai":
            api_key = st.text_input(
                "OPENAI_API_KEY",
                value="",
                type="password",
                placeholder="留空则保留现有 .env 值",
                key="openai_api_key_editor",
            )
            model = st.text_input(
                "OPENAI_EMBEDDING_MODEL",
                value=_env_value("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL),
                key="openai_embedding_model_editor",
            )
            base_url = st.text_input(
                "OPENAI_EMBEDDING_BASE_URL",
                value=_env_value("OPENAI_EMBEDDING_BASE_URL", OPENAI_EMBEDDING_BASE_URL),
                key="openai_embedding_base_url_editor",
            )
            rerank_default = st.checkbox(
                "默认启用 DashScope Rerank",
                value=False,
                help="OpenAI embedding 不自带 rerank；打开后仍需要 DASHSCOPE_API_KEY。",
                key="openai_rerank_default_editor",
            )
            updates["SORTPAPER_EMBEDDING_API_KEY_ENV"] = "OPENAI_API_KEY"
            updates["OPENAI_EMBEDDING_MODEL"] = model.strip() or "text-embedding-3-large"
            updates["OPENAI_EMBEDDING_BASE_URL"] = base_url.strip() or "https://api.openai.com/v1"
            updates["SORTPAPER_RERANK_DEFAULT_ENABLED"] = "true" if rerank_default else "false"
            if api_key.strip():
                updates["OPENAI_API_KEY"] = api_key.strip()
        else:
            model_path = st.text_input(
                "QWEN3_LOCAL_EMBEDDING_MODEL_PATH",
                value=_env_value("QWEN3_LOCAL_EMBEDDING_MODEL_PATH", QWEN3_LOCAL_EMBEDDING_MODEL_PATH),
                key="qwen3_local_model_path_editor",
            )
            max_length = st.number_input(
                "QWEN3_LOCAL_EMBEDDING_MAX_LENGTH",
                min_value=128,
                max_value=32768,
                value=int(_env_value("QWEN3_LOCAL_EMBEDDING_MAX_LENGTH", str(QWEN3_LOCAL_EMBEDDING_MAX_LENGTH)) or 8192),
                step=128,
                key="qwen3_local_max_length_editor",
            )
            rerank_default = st.checkbox(
                "默认启用 DashScope Rerank",
                value=RERANK_DEFAULT_ENABLED,
                help="本地 Qwen3 负责 embedding；Rerank 仍使用 DashScope qwen3-rerank，需要 DASHSCOPE_API_KEY。",
                key="qwen3_local_rerank_default_editor",
            )
            st.info("本地 Qwen3 不需要 API key；切换到本地模型后，请清空或重建当前 collection。")
            updates["SORTPAPER_EMBEDDING_API_KEY_ENV"] = ""
            updates["QWEN3_LOCAL_EMBEDDING_MODEL_PATH"] = model_path.strip() or "data/models/Qwen3-Embedding-0.6B"
            updates["QWEN3_LOCAL_EMBEDDING_MAX_LENGTH"] = str(int(max_length))
            updates["SORTPAPER_RERANK_DEFAULT_ENABLED"] = "true" if rerank_default else "false"

        if st.button("保存 Embedding 设置", use_container_width=True, key="save_embedding_settings"):
            _save_env_updates(updates)
            st.success("已写入 .env。由于模型配置在启动时加载，请重启 Streamlit 后生效。")


def legacy_sidebar() -> dict:
    """返回 {
        pdf_bytes, paper_id, mode, parse_clicked, filename,
        batch: {files: list[UploadedFile], pending: bool, mode: str} | None,
    }。"""
    result: dict = {
        "pdf_bytes": None, "paper_id": "", "mode": MODE_MINERU,
        "parse_clicked": False, "filename": "",
        "batch": None,
    }

    with st.sidebar:
        st.title(f"{LOGO} SortPaper")
        st.caption("学术论文解析与检索工具")
        _render_embedding_settings_ui()
        st.divider()

        # ── 文件来源 ──
        source = st.radio("PDF 来源", ["上传文件", "示例论文"], horizontal=True)

        pdf_bytes: bytes | None = None
        filename = ""
        uploaded_files: list = []

        if source == "上传文件":
            uploaded = st.file_uploader(
                "选择 PDF 文件",
                type="pdf",
                accept_multiple_files=True,
                label_visibility="collapsed",
                key="pdf_uploader",
            )
            uploaded_files = list(uploaded) if uploaded else []
            if uploaded_files:
                st.caption(f"已选择 {len(uploaded_files)} 个 PDF 文件")
        else:
            samples = sorted(SAMPLE_DIR.glob("*.pdf")) if SAMPLE_DIR.exists() else []
            if not samples:
                st.warning("data/sample_papers/ 目录为空")
            else:
                choice = st.selectbox(
                    "选择示例论文", options=samples,
                    format_func=lambda p: p.name, label_visibility="collapsed",
                )
                if choice:
                    pdf_bytes = choice.read_bytes()
                    filename = choice.name

        is_batch = len(uploaded_files) > 1
        if len(uploaded_files) == 1:
            pdf_bytes = uploaded_files[0].getvalue()
            filename = uploaded_files[0].name

        if pdf_bytes:
            paper_id = build_paper_id(pdf_bytes)
            st.caption(f"📎 {filename}")
            st.caption(f"ID: `{paper_id}`")
        elif is_batch:
            paper_id = "batch"
            filename = f"批量 {len(uploaded_files)} 篇"
            effective_batches = (len(uploaded_files) + BATCH_MAX_PAPER_WORKERS - 1) // BATCH_MAX_PAPER_WORKERS
            est_min = effective_batches * BATCH_ESTIMATED_MINUTES_PER_PAPER
            st.caption(f"📎 批量 {len(uploaded_files)} 篇")
            st.caption(f"预计耗时约 {est_min} 分钟（论文并发 {BATCH_MAX_PAPER_WORKERS}）")
        else:
            paper_id = ""

        result["pdf_bytes"] = pdf_bytes
        result["paper_id"] = paper_id
        result["filename"] = filename

        st.divider()

        # ── 解析模式 ──
        st.subheader("解析模式")
        show_legacy_modes = st.checkbox(
            "显示历史/兼容解析入口",
            value=False,
            help="默认只显示 MinerU 主路径；打开后可运行旧 parser 做展示、对照或兼容入库。",
        )
        mode_options = list(PRIMARY_MODES) + (list(LEGACY_MODES) if show_legacy_modes else [])
        mode_captions = {
            MODE_MINERU: "最快：文本/表格/图片占位，不调用图片 VL",
            MODE_MINERU_FULL: "完整：文本/表格 + Figure group 图片转文字",
            MODE_MINERU_AUTO_STORE: "完整解析后直接写入 Qdrant",
            MODE_LEGACY_PREVIEW: "自研旧路径：文本/表格预览，图片仅占位",
            MODE_LEGACY_PIPELINE: "自研旧路径：文本/表格/图片 + Judge；用于对照展示",
            MODE_LEGACY_AUTO_STORE: "兼容旧入库：自研解析 → Judge → Qdrant",
        }
        mode = st.radio(
            "模式",
            mode_options,
            captions=[mode_captions[item] for item in mode_options],
            label_visibility="collapsed",
        )
        result["mode"] = mode

        if mode == MODE_MINERU:
            st.success(
                "快速预览：调用 MinerU 解析 PDF，保留页码、bbox、表格和 Figure group，但图片只保留占位与坐标，不调用图片转文字。"
            )
        elif mode == MODE_MINERU_FULL:
            st.success(
                "完整解析：调用 MinerU 解析 PDF，并按 Figure group 调用 VL 把图片转成检索友好的文字描述。"
            )
        elif mode == MODE_MINERU_AUTO_STORE:
            st.success(
                "一键入库：执行 MinerU 完整解析（含 Figure group 图片转文字），生成 LayoutChunk 后立即写入 Qdrant。"
            )
        elif mode == MODE_LEGACY_PREVIEW:
            st.info(
                "历史展示路径：快速解析文本/表格；仅表格质量诊断会调用 LLM Judge，图片只生成占位信息，不写入向量库。"
            )
        elif mode == MODE_LEGACY_PIPELINE:
            st.info(
                "历史展示路径：单篇通常 **1 ~ 2 分钟**；会解析文本、表格和图片并运行 LLM Judge，但不自动入库。"
            )
        elif mode == MODE_LEGACY_AUTO_STORE:
            st.info(
                "兼容入口：沿用自研解析链路完成 Judge 与 Qdrant 入库；后续会逐步迁移到 MinerU 主路径。\n"
                f"已入库论文会跳过；向量模型：`{EMBEDDING_PROVIDER}:{EMBEDDING_MODEL}`。"
            )

        st.divider()

        selected_count = len(uploaded_files) if source == "上传文件" else int(pdf_bytes is not None)
        can_parse = selected_count > 0
        button_label = "🚀 开始批量处理" if selected_count > 1 else "🚀 开始解析"
        parse_clicked = st.button(
            button_label, disabled=not can_parse,
            use_container_width=True, type="primary",
        )
        if not can_parse:
            st.caption("请先选择 PDF 文件")

        result["parse_clicked"] = parse_clicked
        if is_batch:
            result["batch"] = {
                "files": uploaded_files,
                "pending": parse_clicked,
                "mode": mode,
            }

        st.divider()
        _render_vector_library_ui()

    return result


def _render_batch_import() -> list:
    """渲染批量导入 UI，返回用户上传的 UploadedFile 列表。"""
    uploaded = st.file_uploader(
        "拖拽或选择 PDF 文件", type="pdf",
        accept_multiple_files=True, label_visibility="collapsed",
        key="batch_uploader",
    )
    if uploaded:
        st.caption(f"已选择 {len(uploaded)} 个 PDF 文件")
    else:
        st.caption("支持多选或拖拽 PDF 文件")
    return list(uploaded) if uploaded else []


def render_vector_library_workspace() -> None:
    _render_vector_library_ui()


def _render_vector_library_ui() -> None:
    """展示向量库文献列表 + 删除/清空功能。"""
    from src.application import vector_library

    st.subheader("向量库管理")

    try:
        total = vector_library.total_count()
    except Exception:
        st.warning("Qdrant 未连接")
        return

    st.caption(f"共 {total} 条记录")

    papers = vector_library.list_papers()
    if not papers:
        st.caption("暂无已录入文献")
    else:
        pending_papers = [p for p in papers if p.get("enrichment_status", "done") == "pending"]
        done_count = len(papers) - len(pending_papers)
        st.caption(f"论文 {len(papers)} 篇；待质量分析 {len(pending_papers)} 篇；已完成 {done_count} 篇")
        if st.button(
            "📊 一键质量分析全部待分析论文",
            disabled=not pending_papers,
            use_container_width=True,
            type="primary",
            key="enrich_all_pending_papers",
        ):
            progress = st.progress(0, "准备质量分析…")
            status_box = st.empty()
            total_pending = len(pending_papers)

            def update_progress(idx: int, _total: int, paper: dict, success: int, fail: int) -> None:
                pid = paper["paper_id"]
                title = paper.get("paper_title", pid)
                progress.progress((idx - 1) / total_pending, f"质量分析 {idx}/{total_pending}: {title[:36]}")
                status_box.caption(f"已完成 {idx}/{total_pending}；成功 {success}，失败 {fail}")

            batch_result = vector_library.enrich_pending_papers(
                pending_papers,
                progress_callback=update_progress,
            )
            success_n = batch_result["success_n"]
            fail_n = batch_result["fail_n"]
            failures = batch_result["failures"]
            progress.progress(1.0, "质量分析完成")
            if failures:
                st.warning(f"批量质量分析完成：成功 {success_n}，失败 {fail_n}")
                with st.expander("查看失败详情", expanded=False):
                    for item in failures[:20]:
                        st.caption(item)
                    if len(failures) > 20:
                        st.caption(f"其余 {len(failures) - 20} 条失败未展开显示")
            else:
                st.success(f"批量质量分析完成：成功 {success_n} 篇")
            st.rerun()

        for p in papers:
            pid = p["paper_id"]
            with st.expander(f"📄 {p['paper_title'][:50]}", expanded=False):
                st.caption(f"Hash: `{pid}`")
                st.caption(f"Chunks: {p['chunk_count']}")
                status = p.get("enrichment_status", "done")
                status_note = p.get("enrichment_status_note", "")
                if status == "pending":
                    label = "质量分析: 待分析"
                    if status_note:
                        label += f"（{status_note}）"
                else:
                    label = "质量分析: 已完成"
                st.caption(label)
                eval_label = "📊 一键质量评估" if status == "pending" else "🔁 重新质量评估"
                if st.button(eval_label, key=f"enrich_sidebar_{pid}", use_container_width=True):
                    with st.spinner("正在从 Qdrant 读取 chunks 并执行 Map-Reduce…"):
                        result = vector_library.enrich_paper(pid)
                    if result.get("error"):
                        st.error(f"质量评估失败：{result.get('error')}")
                    else:
                        st.success(
                            f"质量评估完成：{result.get('category', '?')} "
                            f"/ credibility={result.get('credibility', '?')}"
                        )
                        st.rerun()
                if st.button("🗑️ 删除此文献", key=f"del_{pid}", use_container_width=True):
                    try:
                        n = vector_library.delete_paper(pid)
                        st.success(f"已删除 {n} 条记录")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"删除失败：{exc}")

    st.divider()
    # 清空按钮
    if "confirm_clear" not in st.session_state:
        st.session_state.confirm_clear = False

    if not st.session_state.confirm_clear:
        if st.button("🗑️ 清空全部向量库", use_container_width=True, type="secondary"):
            st.session_state.confirm_clear = True
            st.rerun()
    else:
        st.warning("⚠️ 将删除 Qdrant 中所有论文数据，不可恢复")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ 确认清空", use_container_width=True, type="primary"):
                try:
                    before = vector_library.clear_library()
                    st.session_state.confirm_clear = False
                    st.success(f"已清空（删除 {before} 条记录）")
                    st.rerun()
                except Exception as exc:
                    st.session_state.confirm_clear = False
                    st.error(f"清空失败：{exc}")
        with col2:
            if st.button("❌ 取消", use_container_width=True):
                st.session_state.confirm_clear = False
                st.rerun()


