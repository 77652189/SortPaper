from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run due external paper monitors for PaperSort.")
    parser.add_argument("--json", action="store_true", help="Print the run summary as JSON.")
    args = parser.parse_args()

    from src.application import external_imports

    summary = external_imports.run_due_monitors(now=datetime.now(timezone.utc))
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(
            "PaperSort external monitors | "
            f"run_count={summary.get('run_count', 0)} | "
            f"latest_fetch_at={summary.get('latest_fetch_at', '')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
