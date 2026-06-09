from __future__ import annotations


def test_update_env_text_updates_existing_and_appends_new_keys() -> None:
    from app_sidebar import _update_env_text

    original = "\n".join(
        [
            "# SortPaper",
            "SORTPAPER_EMBEDDING_PROVIDER=dashscope",
            "DASHSCOPE_API_KEY=keep-me",
        ]
    )

    updated = _update_env_text(
        original,
        {
            "SORTPAPER_EMBEDDING_PROVIDER": "qwen3_local",
            "SORTPAPER_QDRANT_COLLECTION": "papers_qwen3_local",
        },
    )

    assert "SORTPAPER_EMBEDDING_PROVIDER=qwen3_local" in updated
    assert "DASHSCOPE_API_KEY=keep-me" in updated
    assert "SORTPAPER_QDRANT_COLLECTION=papers_qwen3_local" in updated
    assert updated.endswith("\n")


def test_update_env_text_quotes_values_that_need_it() -> None:
    from app_sidebar import _update_env_text

    updated = _update_env_text(
        "",
        {"QWEN3_LOCAL_EMBEDDING_MODEL_PATH": "data/models/Qwen 3 Embedding"},
    )

    assert 'QWEN3_LOCAL_EMBEDDING_MODEL_PATH="data/models/Qwen 3 Embedding"' in updated


def test_update_env_text_can_clear_key_env_without_touching_secret() -> None:
    from app_sidebar import _update_env_text

    updated = _update_env_text(
        "SORTPAPER_EMBEDDING_API_KEY_ENV=DASHSCOPE_API_KEY\nDASHSCOPE_API_KEY=secret\n",
        {"SORTPAPER_EMBEDDING_API_KEY_ENV": ""},
    )

    assert "SORTPAPER_EMBEDDING_API_KEY_ENV=\n" in updated
    assert "DASHSCOPE_API_KEY=secret" in updated
