"""Run ShiftGate on the three synthetic delivery scenarios.

    PYTHONPATH must include the Kuration SDK src tree, or set KURATION_SRC.
"""

from __future__ import annotations

import os
import sys

_SRC = os.environ.get(
    "KURATION_SRC",
    "/tmp/kuration-sdk/kuration-sdk-private-release/src",
)
if _SRC and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from pathlib import Path

from shiftgate.gate import evaluate, render_scorecard


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "out/shiftgate")
    expected = {"accept": "accept", "repair": "repair", "reject": "reject"}
    for scenario, want in expected.items():
        scorecard = evaluate(scenario, root / scenario)
        print(render_scorecard(scorecard))
        print(f"scorecard: {root / scenario / 'scorecard.json'}")
        print()
        got = scorecard["decision"]
        if got != want:
            raise SystemExit(f"{scenario}: expected {want}, got {got}")
        if want == "reject" and scorecard["keep_shards"]:
            raise SystemExit("reject wrote a keep shard")
        if want != "reject" and len(scorecard["keep_shards"]) != 2:
            raise SystemExit(f"{scenario}: expected a W shard and an S shard")
    print("shiftgate: accept, repair, and reject matched the delivery contract")


if __name__ == "__main__":
    main()
