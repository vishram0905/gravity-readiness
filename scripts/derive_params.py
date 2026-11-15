# AI-assisted code (team statement and per-step log: docs/AI_USE_LOG.md).
"""B3: derive parameters from the full HBEDB (needs HBEDB_DIR) and write params/*.json.

    HBEDB_DIR=... python scripts/derive_params.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from readiness import config as C  # noqa: E402
from readiness import params as P  # noqa: E402
from readiness import signals as S  # noqa: E402


def main() -> int:
    if not C.full_data():
        print("Set HBEDB_DIR to the full HBEDB to derive parameters.", file=sys.stderr)
        return 1
    trials = S.build_trial_table()
    data = P.derive_data_params(trials)
    P.PARAMS_FILE.write_text(json.dumps(data, indent=1))
    lit = P.derive_q95_lit()
    P.LIT_FILE.write_text(json.dumps(lit))
    for k, v in data.items():
        print(k, v)
    print("q95_lit at b = 85:", {k: round(float(__import__('numpy').interp(85, lit['grid'], v)), 4)
                                for k, v in lit.items() if k != 'grid'})
    return 0


if __name__ == "__main__":
    sys.exit(main())
