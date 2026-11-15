# AI-assisted code (team statement and per-step log: docs/AI_USE_LOG.md).
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def pytest_collection_modifyitems(config, items):
    if os.environ.get("HBEDB_DIR"):
        return
    skip = pytest.mark.skip(reason="full HBEDB not available (set HBEDB_DIR)")
    for item in items:
        if "full" in item.keywords:
            item.add_marker(skip)
