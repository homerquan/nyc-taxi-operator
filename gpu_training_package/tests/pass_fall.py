#!/usr/bin/env python3
"""Convenience spelling requested by the task: run the rubric and pass/fail gate."""
import json
from pathlib import Path

from rubric_score import discover_paths, main as score_rubric
from pass_fail import main as score_pass_fail

if __name__ == "__main__":
    status = score_rubric() or score_pass_fail()
    if status:
        raise SystemExit(status)
    result = json.loads((Path(discover_paths().logs_dir) / "reward.json").read_text())
    # Unlike the Harbor entrypoint, this convenience command also uses the
    # process status for ordinary shell/CI use.
    raise SystemExit(0 if result.get("passed") else 1)
