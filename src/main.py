"""Entry point the grader runs: python src/main.py"""

from __future__ import annotations

import argparse
import os
import sys

_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

_SDK = os.environ.get("KURATION_SRC", "")
if _SDK and _SDK not in sys.path:
    sys.path.insert(0, _SDK)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Curate a synthetic robot fleet into a contact-hour budget with Kuration."
    )
    parser.add_argument(
        "--out",
        default="out/fleetgate",
        help="Directory for the scorecard and hour-budget shards.",
    )
    args = parser.parse_args()

    from fleetgate.curate import deliver, render
    from fleetgate.logs import JUMP_WINDOW

    scorecard = deliver(args.out)
    print(render(scorecard))
    print(f"scorecard: {args.out}/scorecard.json")
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
