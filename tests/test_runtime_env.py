from __future__ import annotations


def test_load_project_env_uses_configured_path_not_cwd(monkeypatch, tmp_path) -> None:
    from src import runtime_env

    env_path = tmp_path / ".env"
    env_path.write_text("SORTPAPER_TEST_ENV_LOADED=from-dotenv\n", encoding="utf-8")
    other_cwd = tmp_path / "other"
    other_cwd.mkdir()

    monkeypatch.delenv("SORTPAPER_TEST_ENV_LOADED", raising=False)
    monkeypatch.setattr(runtime_env, "ENV_PATH", env_path)
    monkeypatch.chdir(other_cwd)

    assert runtime_env.load_project_env() is True
    assert runtime_env.load_project_env() is True
    assert runtime_env.PROJECT_ROOT.name == "SortPaper"
    assert runtime_env.ENV_PATH == env_path
    assert __import__("os").environ["SORTPAPER_TEST_ENV_LOADED"] == "from-dotenv"


def test_load_project_env_does_not_override_existing_env(monkeypatch, tmp_path) -> None:
    from src import runtime_env

    env_path = tmp_path / ".env"
    env_path.write_text("SORTPAPER_TEST_ENV_LOADED=from-dotenv\n", encoding="utf-8")

    monkeypatch.setenv("SORTPAPER_TEST_ENV_LOADED", "from-shell")
    monkeypatch.setattr(runtime_env, "ENV_PATH", env_path)

    assert runtime_env.load_project_env() is True
    assert __import__("os").environ["SORTPAPER_TEST_ENV_LOADED"] == "from-shell"
