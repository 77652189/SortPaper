from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _tree(path: str) -> ast.Module:
    return ast.parse((PROJECT_ROOT / path).read_text(encoding="utf-8-sig"))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def _called_names(node: ast.AST) -> set[str]:
    calls: set[str] = set()
    for item in ast.walk(node):
        if not isinstance(item, ast.Call):
            continue
        if isinstance(item.func, ast.Name):
            calls.add(item.func.id)
        elif isinstance(item.func, ast.Attribute):
            calls.add(item.func.attr)
    return calls


def _workspace_branch_calls(fn: ast.FunctionDef, workspace: str) -> set[str]:
    for node in ast.walk(fn):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not isinstance(test, ast.Compare):
            continue
        if not isinstance(test.left, ast.Name) or test.left.id != "workspace":
            continue
        comparators = test.comparators
        if comparators and isinstance(comparators[0], ast.Constant) and comparators[0].value == workspace:
            return _called_names(ast.Module(body=node.body, type_ignores=[]))
    raise AssertionError(f"workspace branch {workspace} not found")


def test_workbench_defaults_to_online_daily_brief() -> None:
    tree = _tree("app_workbench.py")
    defaults_assign = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "WORKBENCH_DEFAULT_STATE"
    )
    assert isinstance(defaults_assign.value, ast.Dict)
    defaults = {
        key.value: value.value
        for key, value in zip(defaults_assign.value.keys, defaults_assign.value.values)
        if isinstance(key, ast.Constant) and isinstance(value, ast.Constant)
    }

    assert defaults["workspace"] == "online"
    assert defaults["online_view"] == "daily_brief"
    assert defaults["import_view"] == "local_pdf"
    assert defaults["search_view"] == "manual"


def test_main_routes_by_workspace_without_global_tabs() -> None:
    main_fn = _function(_tree("app.py"), "main")
    calls = _called_names(main_fn)

    assert "_render_workspace_nav" in calls
    assert "sidebar" in calls
    assert "_render_online_workspace" in calls
    assert "_render_import_workspace" in calls
    assert "_render_search_workspace" in calls
    assert "tabs" not in calls
    assert "render_external_import_tab" not in calls
    assert "render_standalone_search_tab" not in calls


def test_sidebar_is_contextual_not_fixed_import_or_library() -> None:
    sidebar_fn = _function(_tree("app_sidebar.py"), "sidebar")
    calls = _called_names(sidebar_fn)

    assert "file_uploader" not in calls
    assert "_render_vector_library_ui" not in calls
    assert "render_parse_mode_controls" in calls


def test_embedding_settings_only_render_in_import_workspace_sidebar() -> None:
    sidebar_fn = _function(_tree("app_sidebar.py"), "sidebar")

    assert "_render_embedding_settings_ui" not in _workspace_branch_calls(sidebar_fn, "online")
    assert "_render_embedding_settings_ui" in _workspace_branch_calls(sidebar_fn, "import")
    assert "_render_embedding_settings_ui" not in _workspace_branch_calls(sidebar_fn, "search")


def test_external_import_exposes_workbench_subviews() -> None:
    tree = _tree("app_external_ui.py")
    function_names = {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }

    assert {
        "render_daily_brief_view",
        "render_candidate_library_view",
        "render_manual_discovery_view",
        "render_topics_view",
    }.issubset(function_names)


def test_top_level_workspaces_are_three_user_tasks() -> None:
    tree = _tree("app.py")
    labels_assign = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "WORKSPACE_LABELS" for target in node.targets)
    )
    assert isinstance(labels_assign.value, ast.Dict)
    keys = [key.value for key in labels_assign.value.keys if isinstance(key, ast.Constant)]

    assert keys == ["online", "import", "search"]


def test_search_workspace_has_no_saved_results_view() -> None:
    tree = _tree("app.py")
    labels_assign = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "SEARCH_VIEW_LABELS" for target in node.targets)
    )
    assert isinstance(labels_assign.value, ast.Dict)
    keys = [key.value for key in labels_assign.value.keys if isinstance(key, ast.Constant)]
    search_fn = _function(tree, "_render_search_workspace")
    calls = _called_names(search_fn)

    assert keys == ["manual", "agent"]
    assert "render_saved_tab" not in calls
