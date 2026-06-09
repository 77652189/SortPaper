from __future__ import annotations

from src.application import embedding_settings


def test_provider_defaults_are_centralized() -> None:
    assert embedding_settings.provider_collection_default("qwen3_local") == "papers_qwen3_local"
    assert embedding_settings.provider_collection_default("dashscope") == "papers"
    assert embedding_settings.provider_dim_default("openai") == 3072
    assert embedding_settings.provider_dim_default("dashscope") == 1024


def test_env_value_accepts_bom_prefixed_key(monkeypatch) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setenv("\ufeffDASHSCOPE_API_KEY", "secret")

    assert embedding_settings.env_value("DASHSCOPE_API_KEY") == "secret"
    assert embedding_settings.env_key_configured("DASHSCOPE_API_KEY") is True


def test_update_env_text_preserves_comments_quotes_and_allows_clear() -> None:
    updated = embedding_settings.update_env_text(
        "# SortPaper\nSORTPAPER_EMBEDDING_API_KEY_ENV=DASHSCOPE_API_KEY\n",
        {
            "SORTPAPER_EMBEDDING_API_KEY_ENV": "",
            "QWEN3_LOCAL_EMBEDDING_MODEL_PATH": "data/models/Qwen 3",
        },
    )

    assert updated.startswith("# SortPaper\n")
    assert "SORTPAPER_EMBEDDING_API_KEY_ENV=\n" in updated
    assert 'QWEN3_LOCAL_EMBEDDING_MODEL_PATH="data/models/Qwen 3"' in updated


def test_save_env_updates_writes_file_and_updates_process_env(tmp_path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    monkeypatch.delenv("SORTPAPER_EMBEDDING_PROVIDER", raising=False)

    embedding_settings.save_env_updates(
        {"SORTPAPER_EMBEDDING_PROVIDER": "qwen3_local"},
        env_path=env_path,
    )

    assert env_path.read_text(encoding="utf-8") == "SORTPAPER_EMBEDDING_PROVIDER=qwen3_local\n"
    assert embedding_settings.env_value("SORTPAPER_EMBEDDING_PROVIDER") == "qwen3_local"
