"""Run with --check in CI to detect stale contract exports."""

import json
import sys
from pathlib import Path

from mlmom.backend import app as backend
from mlmom.research_mock import app as research


def main():
    root = Path(__file__).resolve().parents[2] / "contracts"
    checking = "--check" in sys.argv
    for name, app in [("research", research), ("backend", backend)]:
        content = json.dumps(app.openapi(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        path = root / f"{name}.openapi.json"
        if checking:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                raise SystemExit(f"Stale contract: {path}; run npm run contracts")
        else:
            root.mkdir(exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
