"""Select a fleet's trainable hours with Kuration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import kuration.gameplay  # noqa: F401  slam_quality, slam_motion_diversity
import kuration.selection  # noqa: F401  rare_event_coverage
from kuration import SchemaSpec, input, iter_shards

from .logs import HZ, N_WINDOWS, WINDOW, build_fleet

SCHEMA = SchemaSpec.from_dict({
    "name": "fleet-shift",
    "entity_axis": "robot",
    "sample_hz": HZ,
    "fields": {
        "camera_position": {
            "kind": "vector", "dims": ["x", "y", "z"], "semantics": "position",
        },
        "camera_z_axis": {"kind": "vector", "dims": ["x", "y", "z"]},
        "tracking_confidence": {"kind": "scalar"},
        "contact": {"kind": "scalar"},
        "command": {"kind": "vector", "dims": ["vx", "vy", "wz"]},
    },
})

# Four contact spans for train, one for test. Each span is one 32-tick window.
TRAIN_SPANS = 4
TEST_SPANS = 1
SPAN_HOURS = WINDOW / HZ / 3600.0
TRAIN_HOURS = TRAIN_SPANS * SPAN_HOURS
TEST_HOURS = TEST_SPANS * SPAN_HOURS

GAPS = [
    "rare_event_coverage only takes a count. An hour budget cannot call it, so the delivery uses the same contact-rate feature with sample('equal', hours=) after a contact filter.",
    "sample('weighted') materializes the whole pool and has no hours or streaming path.",
    "Contact is a field the fleet must already log. The SDK does not detect gripper contact from video.",
    "aintropy/collision needs a Counter-Strike mesh and a fitted distance file, so it is the wrong operator for a robot aisle.",
    "slam_quality_score stays high when a single step teleports, because the penalty is a fraction of steps. The filter uses slam_translation_jump_fraction.",
    "slam_reconstruct from raw video needs the slam extra and a depth model. This run uses the pose the robot already recorded.",
    "pixel_shift applies to isolated keyboard presses, not to a continuous drive command.",
]


def _scored(fleet: dict):
    """Window every shift, score contact and camera motion, drop teleports."""
    return (
        input(fleet, schema=SCHEMA)
        .window(WINDOW, stride=WINDOW)
        .measure("contact_rate", "rate", of="contact")
        .slam_motion_diversity(axis="slam_path_length", bins=3)
        .slam_quality()
        .filter("slam_translation_jump_fraction", "<=", 0)
        .options(mode="strict", seed=7)
    )


def _mean(segments, name: str) -> float:
    if not segments:
        return float("nan")
    return float(np.mean([segment.features[name] for segment in segments]))


def _chronological(segments, n: int) -> list:
    ordered = sorted(segments, key=lambda segment: (segment.start_idx, segment.source_id))
    return ordered[:n]


def deliver(out_dir: str | Path) -> dict:
    """Write the hour-budget dataset and a scorecard against an uncurated prefix."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fleet = build_fleet()

    pool = (
        input(fleet, schema=SCHEMA)
        .window(WINDOW, stride=WINDOW)
        .measure("contact_rate", "rate", of="contact")
        .slam_quality()
        .options(mode="strict", seed=7)
        .collect(n=None)
    )
    jumped = [
        segment for segment in pool.selected
        if segment.features["slam_translation_jump_fraction"] > 0
    ]
    clean = _scored(fleet).collect(n=None)

    # The count recipe shows the contact lift. It cannot express the hour budget.
    counted = _scored(fleet).rare_event_coverage(
        of="contact", feature="contact_rate_recipe", count=6, weight=1.0,
    ).collect()

    pipeline = (
        _scored(fleet)
        .filter("contact_rate", ">=", 0.5)
        .split(test=0.25)
        .sample("equal", hours=TRAIN_HOURS, test_hours=TEST_HOURS)
    )
    target = out / "hours"
    written = pipeline.output(
        str(target),
        container="npz",
        shard_length=WINDOW,
        span_steps=WINDOW,
    )
    explain = pipeline.explain()
    (target / "explain.txt").write_text(explain.render() if hasattr(explain, "render") else str(explain))

    delivered_contact = _shard_contact(target)
    prefix = _chronological(pool.selected, TRAIN_SPANS + TEST_SPANS)
    scorecard = {
        "shifts": len(fleet),
        "windows": N_WINDOWS * len(fleet),
        "pool_contact_rate": _mean(pool.selected, "contact_rate"),
        "jump_windows": [
            {"source": segment.source_id, "start": segment.start_idx}
            for segment in jumped
        ],
        "clean_windows": len(clean.selected),
        "rare_event_coverage": {
            "selected": len(counted.selected),
            "contact_rate": _mean(counted.selected, "contact_rate_recipe"),
        },
        "budget_hours": {"train": TRAIN_HOURS, "test": TEST_HOURS},
        "delivered_contact_rate": delivered_contact,
        "chronological_prefix_contact_rate": _mean(prefix, "contact_rate"),
        "output": written.get("results", written) if isinstance(written, dict) else written,
        "shards": str(target),
        "sdk_gaps": GAPS,
    }
    (out / "scorecard.json").write_text(json.dumps(scorecard, indent=2, default=_json) + "\n")
    return scorecard


def _shard_contact(dataset: Path) -> float:
    rates = []
    for split in ("train", "test"):
        for arrays in iter_shards(str(dataset), split=split):
            contact = np.asarray(arrays["contact"], dtype=np.float64)
            present = np.asarray(arrays["_present"])
            rates.append(float(contact[present].mean()) if present.any() else float(contact.mean()))
            step = np.linalg.norm(np.diff(np.asarray(arrays["camera_position"], dtype=np.float64), axis=0), axis=-1)
            if step.size and float(np.max(step)) > 5:
                raise RuntimeError("a delivered shard still contains a camera teleport")
    if not rates:
        raise RuntimeError("hour budget wrote no shards")
    return float(np.mean(rates))


def _json(value):
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    raise TypeError(type(value).__name__)


def render(scorecard: dict) -> str:
    rare = scorecard["rare_event_coverage"]
    lines = [
        f"shifts:        {scorecard['shifts']}   windows: {scorecard['windows']}",
        f"pool contact:  {scorecard['pool_contact_rate']:.0%}",
        f"jumps dropped: {len(scorecard['jump_windows'])} "
        f"({', '.join(item['source'] for item in scorecard['jump_windows']) or 'none'})",
        f"rare-event:    {rare['selected']} windows at {rare['contact_rate']:.0%} contact",
        f"delivered:     {scorecard['delivered_contact_rate']:.0%} contact "
        f"inside {scorecard['budget_hours']['train'] * 3600:.1f}s train "
        f"+ {scorecard['budget_hours']['test'] * 3600:.1f}s test",
        f"same duration, taken from the start of each shift: "
        f"{scorecard['chronological_prefix_contact_rate']:.0%} contact",
        f"shards:        {scorecard['shards']}",
    ]
    return "\n".join(lines)
