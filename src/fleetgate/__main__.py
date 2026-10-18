"""Curate the synthetic fleet and check the hour budget.

KURATION_SRC points at the SDK src tree when the package is not installed.
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

from fleetgate.curate import deliver, render
from fleetgate.logs import JUMP_WINDOW


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "out/fleetgate")
    scorecard = deliver(root)
    print(render(scorecard))
    print(f"scorecard: {root / 'scorecard.json'}")
    if scorecard["delivered_contact_rate"] < 0.9:
        raise SystemExit("delivered hours are not contact-rich")
    if scorecard["chronological_prefix_contact_rate"] >= scorecard["delivered_contact_rate"]:
        raise SystemExit("curation did not beat taking the start of each shift")
    if scorecard["rare_event_coverage"]["contact_rate"] <= scorecard["pool_contact_rate"]:
        raise SystemExit("rare-event coverage did not raise the contact rate")
    jumped = {item["source"] for item in scorecard["jump_windows"]}
    if set(JUMP_WINDOW) - jumped:
        raise SystemExit(f"expected a jump on {sorted(JUMP_WINDOW)}, saw {sorted(jumped)}")
    print("fleetgate: contact hours kept, transit and the camera teleport left out")


if __name__ == "__main__":
    main()
