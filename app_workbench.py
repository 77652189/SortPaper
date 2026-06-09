from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any


WORKBENCH_DEFAULT_STATE: dict[str, Any] = {
    "workspace": "online",
    "online_view": "daily_brief",
    "import_view": "local_pdf",
    "search_view": "manual",
    "library_selected_paper": "",
    "result": None,
    "current_paper": None,
    "orig_filename": "",
    "pending_save_filename": "",
    "save_success_msg": "",
    "batch_results": [],
    "batch_preview_results": {},
}


def ensure_workspace_state(
    session_state: MutableMapping[str, Any],
    *,
    workspace_labels: dict[str, str],
    online_view_labels: dict[str, str],
    import_view_labels: dict[str, str],
    search_view_labels: dict[str, str],
) -> None:
    """Keep workbench navigation state valid without binding to Streamlit UI code."""

    for key, value in WORKBENCH_DEFAULT_STATE.items():
        if key not in session_state:
            session_state[key] = value

    if session_state.get("workspace") in {"library", "analysis"}:
        session_state["workspace"] = "import"
    elif session_state.get("workspace") not in workspace_labels:
        session_state["workspace"] = "online"

    if session_state.get("import_view") not in import_view_labels:
        session_state["import_view"] = "local_pdf"

    if session_state.get("online_view") not in online_view_labels:
        legacy_import_view = session_state.get("import_view")
        session_state["online_view"] = (
            legacy_import_view
            if legacy_import_view in online_view_labels
            else "daily_brief"
        )

    if session_state.get("search_view") not in search_view_labels:
        session_state["search_view"] = "manual"
