from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_ui_modules_do_not_import_low_level_agent_domain_or_store() -> None:
    forbidden_prefixes = (
        "src.agent",
        "src.domain",
        "src.store",
    )
    for filename in ("app_ui.py", "app_sidebar.py"):
        modules = _imported_modules(PROJECT_ROOT / filename)
        forbidden = [
            module
            for module in modules
            if module.startswith(forbidden_prefixes)
        ]

        assert forbidden == []


def test_app_pipeline_facade_is_only_used_by_main_app_shell() -> None:
    ui_modules = {
        filename: _imported_modules(PROJECT_ROOT / filename)
        for filename in ("app_ui.py", "app_sidebar.py", "app_utils.py", "app_external_ui.py")
    }

    assert all("app_pipeline" not in modules for modules in ui_modules.values())
    assert "app_pipeline" in _imported_modules(PROJECT_ROOT / "app.py")


def test_external_ui_does_not_wire_ingest_dependencies_directly() -> None:
    modules = _imported_modules(PROJECT_ROOT / "app_external_ui.py")

    assert "app_utils" not in modules
    assert "src.application.vector_library" not in modules


def test_application_services_do_not_import_legacy_agent_directly() -> None:
    for path in (PROJECT_ROOT / "src" / "application").glob("*.py"):
        modules = _imported_modules(path)
        forbidden = [
            module
            for module in modules
            if module.startswith("src.agent")
        ]

        assert forbidden == [], path.name


def test_paper_pipeline_does_not_wire_qdrant_quality_adapter_directly() -> None:
    modules = _imported_modules(PROJECT_ROOT / "src" / "application" / "paper_pipeline.py")

    assert "src.adapters.store.qdrant" not in modules


def test_application_services_do_not_import_qdrant_store_adapter_directly() -> None:
    for path in (PROJECT_ROOT / "src" / "application").glob("*.py"):
        modules = _imported_modules(path)

        assert "src.adapters.store.qdrant" not in modules, path.name


def test_application_services_do_not_import_root_app_facades() -> None:
    forbidden = {"app_pipeline", "app_utils", "app_ui", "app_sidebar", "app_external_ui"}
    for path in (PROJECT_ROOT / "src" / "application").glob("*.py"):
        modules = _imported_modules(path)

        assert sorted(forbidden.intersection(modules)) == [], path.name
