"""Score a gameplay drop against a reference action contract using Kuration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import kuration.gameplay  # noqa: F401  registers pixel-shift, the scorer, and SLAM
from kuration import FrameRef, OutputSpec, SchemaSpec, VideoMetadata, input

from .corpus import FRAME_H, FRAME_W, STRIDE, WINDOW, gameplay_drop, trajectory_drop
from kuration.gameplay.pixel_shift.diamond_action_decode import ACTION_WIDTH

ACTION_COLUMNS = [f"a{i}" for i in range(ACTION_WIDTH)]
CONTROLS = ("W", "S")
SCORER = "aintropy/action-state-consistency"

# A window is on-contract when its peak response stays inside this multiple of
# the reference median magnitude. Same band the paper's CS:GO/CS2 gap exceeded.
MAG_LO = 0.75
MAG_HI = 1.35
# Share of measured windows that may violate the contract and still be accepted.
ACCEPT_ANOMALY = 0.15
# Below this in-band share there is nothing worth a repair shard.
REPAIR_IN_BAND = 0.25
SYMMETRY_TOL = 0.15

GAMEPLAY_SCHEMA = SchemaSpec.from_dict({
    "name": "shiftgate-gameplay",
    "sample_hz": 1.0,
    "fields": {
        "action": {"kind": "vector", "dims": ACTION_COLUMNS},
        "pressed": {"kind": "scalar"},
    },
})

TRAJECTORY_SCHEMA = SchemaSpec.from_dict({
    "name": "shiftgate-trajectory",
    "entity_axis": "camera",
    "sample_hz": 10.0,
    "fields": {
        "camera_position": {"kind": "vector", "dims": ["x", "y", "z"], "semantics": "position"},
        "camera_z_axis": {"kind": "vector", "dims": ["x", "y", "z"]},
        "tracking_confidence": {"kind": "scalar"},
    },
})

SDK_GAPS = [
    "No corpus-diff operator: reference and candidate are two pipelines, and the ratio table is caller-side.",
    "Baselines are caller-supplied. load_baselines_from_reference() expects a dataset-specific JSON this drop does not ship.",
    "pixel_shift reports raw pixels. FOV, resolution, and mouse sensitivity are not normalized, so a 2x magnitude gap can be a capture setting.",
    "Momentum after key release is not a pipeline column. The scorer sees per-press shift, not the decay curve from the paper.",
    "S/W symmetry is reduced here from the two shift means. The SDK does not emit a symmetry feature.",
    "aintropy/collision requires a map mesh plus a fitted dist JSON (trimesh/rtree). A drop with no geometry cannot be collision-oversampled.",
    "slam_reconstruct from raw video needs the slam extra and a depth model. This gate scores trajectories the drop already has.",
    "Per-window anomaly counts do not roll up to the clip verdict CONSISTENT/INCONSISTENT. That audit is dataset-mode only, after shards exist.",
    "The action-state scorer has no bounded runtime. explain() warns that it materializes the whole candidate pool.",
    "slam_quality_score stays high when one step in a window is impossible, because the penalty is a step fraction. slam_translation_jump_fraction is the usable flag.",
    "sample('equal') is count-based. Rare-event oversampling exists as rare_event_coverage, but collision rate is not a field until the mesh operator runs.",
    "The gate predicts the axis that tracked playability in the CS:GO/CS2 study. It does not certify playability.",
]


class MemoryVideo:
    """MediaProvider over in-memory RGB frames. No codec, same contract as a real file."""

    capabilities = {"streaming": True, "variable_frame_rate": True}

    def __init__(self, clips: dict[str, list[np.ndarray]]):
        self.clips = clips

    def probe(self, uri: str) -> VideoMetadata:
        frame = self.clips[uri][0]
        height, width = frame.shape[:2]
        count = len(self.clips[uri])
        return VideoMetadata(uri, width, height, count, float(count - 1), 1.0)

    def iter_refs(self, uri: str, *, source_id: str | None = None):
        for index in range(len(self.clips[uri])):
            yield FrameRef(uri, source_id or uri, index, float(index))

    def decode_batch(self, refs) -> np.ndarray:
        return np.stack([self.clips[ref.uri][ref.frame_index] for ref in refs])


def _measured(segments, control: str) -> np.ndarray:
    prefix = control.lower()
    return np.asarray([
        segment.features[f"{prefix}_shift_mean"]
        for segment in segments
        if segment.features.get(f"{prefix}_shift_measured", 0.0) >= 1.0
    ], dtype=np.float64)


def _median(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.median(values))


def _contract_band(shifts: np.ndarray) -> dict:
    """Magnitude band around the reference median, with the reference sign."""
    if shifts.size == 0:
        raise RuntimeError("reference drop produced no measured presses")
    center = float(np.median(shifts))
    sign = 1 if center >= 0 else -1
    magnitude = abs(center)
    return {
        "expected_sign": sign,
        "min": magnitude * MAG_LO,
        "max": magnitude * MAG_HI,
    }


def baselines_from_reference(segments) -> dict:
    """Freeze per-control, single-tap bands. Cell key ``1`` is a one-frame press."""
    baselines = {}
    for control in CONTROLS:
        baselines[control] = {"1": _contract_band(_measured(segments, control))}
    return baselines


def gameplay_pipeline(rows: dict, frames: dict, *, baselines: dict | None):
    """Window each clip, measure W/S pixel shift, and keep windows that contain a press.

    The video URI is the source id. ``MemoryVideo`` probes that same key.
    """
    pipeline = (
        input(rows, schema=GAMEPLAY_SCHEMA,
              fields={"action": ACTION_COLUMNS, "pressed": "pressed"}, time="t")
        .join_video({source: source for source in rows}, provider=MemoryVideo(frames), on="frame_index")
        .window(WINDOW, stride=STRIDE)
        .pixel_shift(controls=list(CONTROLS))
        .measure("press_rate", "rate", of="pressed")
        .filter("press_rate", ">=", 1.0 / WINDOW)
        .options(mode="strict", seed=7)
    )
    if baselines is not None:
        pipeline = pipeline.score(
            SCORER,
            out="action_state_anomaly",
            baselines=baselines,
            controls=list(CONTROLS),
        )
    return pipeline


def trajectory_pipeline(sources: dict):
    return (
        input(sources, schema=TRAJECTORY_SCHEMA)
        .window(8, stride=4)
        .slam_motion_diversity(axis="slam_turn_degrees", bins=4)
        .slam_quality()
        .options(mode="strict", seed=7)
    )


def _ratio(candidate: float, reference: float) -> float:
    if not np.isfinite(candidate) or not np.isfinite(reference) or reference == 0:
        return float("nan")
    return float(candidate / reference)


def _action_decision(reference_segments, candidate_segments, baselines: dict) -> dict:
    ref_w = _measured(reference_segments, "W")
    ref_s = _measured(reference_segments, "S")
    cand_w = _measured(candidate_segments, "W")
    cand_s = _measured(candidate_segments, "S")
    ref_sym = _ratio(abs(_median(ref_s)), abs(_median(ref_w)))
    cand_sym = _ratio(abs(_median(cand_s)), abs(_median(cand_w)))
    anomalies = np.asarray(
        [segment.features["action_state_anomaly"] for segment in candidate_segments],
        dtype=np.float64,
    )
    measured = len(candidate_segments)
    in_band = int(np.sum(anomalies <= 0)) if measured else 0
    anomaly_frac = float(1.0 - in_band / measured) if measured else 1.0
    ratios = {
        "W": _ratio(abs(_median(cand_w)), abs(_median(ref_w))),
        "S": _ratio(abs(_median(cand_s)), abs(_median(ref_s))),
    }
    symmetry_gap = abs(cand_sym - ref_sym) if np.isfinite(cand_sym) and np.isfinite(ref_sym) else float("inf")
    ratios_ok = all(np.isfinite(value) and MAG_LO <= value <= MAG_HI for value in ratios.values())
    if measured and anomaly_frac <= ACCEPT_ANOMALY and ratios_ok and symmetry_gap <= SYMMETRY_TOL:
        label = "accept"
    elif measured and (in_band / measured) >= REPAIR_IN_BAND:
        label = "repair"
    else:
        label = "reject"
    return {
        "label": label,
        "measured_windows": measured,
        "in_band_windows": in_band,
        "anomaly_fraction": anomaly_frac,
        "magnitude_ratio": ratios,
        "symmetry_s_over_w": {"reference": ref_sym, "candidate": cand_sym},
        "baselines": baselines,
        "shift_median": {
            "reference": {"W": _median(ref_w), "S": _median(ref_s)},
            "candidate": {"W": _median(cand_w), "S": _median(cand_s)},
        },
    }


def _spatial_decision(reference_segments, candidate_segments) -> dict:
    def column(segments, name: str) -> np.ndarray:
        return np.asarray([segment.features[name] for segment in segments], dtype=np.float64)

    ref_turn = column(reference_segments, "slam_turn_degrees")
    cand_turn = column(candidate_segments, "slam_turn_degrees")
    ref_path = column(reference_segments, "slam_path_length")
    cand_path = column(candidate_segments, "slam_path_length")
    quality = column(candidate_segments, "slam_quality_score")
    jumps = column(candidate_segments, "slam_translation_jump_fraction")
    quality_ok = float(np.mean(quality >= 0.5)) if quality.size else 0.0
    max_jump = float(np.max(jumps)) if jumps.size else 0.0
    turn_ratio = _ratio(_median(cand_turn), _median(ref_turn))
    path_ratio = _ratio(_median(cand_path), _median(ref_path))
    # One impossible step barely moves slam_quality_score: the penalty is the
    # fraction of steps in the window. The jump fraction is the flag that fires.
    if max_jump > 0 or quality_ok < 0.5:
        label = "unstable"
    elif not np.isfinite(turn_ratio) or turn_ratio < 0.5:
        label = "narrow"
    else:
        label = "ok"
    return {
        "label": label,
        "windows": len(candidate_segments),
        "quality_pass_fraction": quality_ok,
        "max_translation_jump_fraction": max_jump,
        "turn_ratio": turn_ratio,
        "path_ratio": path_ratio,
        "turn_median": {"reference": _median(ref_turn), "candidate": _median(cand_turn)},
        "path_median": {"reference": _median(ref_path), "candidate": _median(cand_path)},
    }


def _overall(action: dict, spatial: dict) -> str:
    if action["label"] == "reject":
        return "reject"
    if action["label"] == "repair" or spatial["label"] != "ok":
        return "repair"
    return "accept"


def _keep_shard(rows, frames, baselines: dict, out_dir: Path, *, control: str, n: int) -> dict | None:
    """Write the in-contract windows for one control as a reproducible NPZ shard."""
    if n < 1:
        return None
    bins = 2 if n >= 2 else 1
    prefix = control.lower()
    pipeline = (
        gameplay_pipeline(rows, frames, baselines=baselines)
        .filter("action_state_anomaly", "<=", 0)
        .filter(f"{prefix}_shift_measured", ">=", 1)
        .bin(f"{prefix}_shift_mean", bins=bins)
    )
    if n >= 3:
        pipeline = pipeline.split(test=1 / 3)
    pipeline = pipeline.sample("equal", n=n)
    spec = OutputSpec.from_dict({
        "shard_length": WINDOW,
        "container": "npz",
        "modalities": {
            "frames": {"dtype": "uint8", "shape": [FRAME_H, FRAME_W, 3], "source": "media:frame"},
            "actions": {"dtype": "float64", "shape": [ACTION_WIDTH], "source": "field:action"},
        },
    })
    target = out_dir / f"keep-{prefix}"
    written = pipeline.output(str(target), spec=spec)
    explain = pipeline.explain()
    explain_text = explain.render() if hasattr(explain, "render") else str(explain)
    (target / "explain.txt").write_text(explain_text)
    return {
        "control": control,
        "path": str(target),
        "requested": n,
        "written": getattr(written, "results", None),
    }


def _control_keep_count(segments, control: str) -> int:
    prefix = control.lower()
    return sum(
        1
        for segment in segments
        if segment.features.get("action_state_anomaly", 1) <= 0
        and segment.features.get(f"{prefix}_shift_measured", 0) >= 1
    )


def evaluate(scenario: str, out_dir: str | Path) -> dict:
    """Run reference, candidate, and trajectory pipelines. Write the scorecard."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ref_rows, ref_frames = gameplay_drop("reference")
    cand_rows, cand_frames = gameplay_drop(scenario)
    reference = gameplay_pipeline(ref_rows, ref_frames, baselines=None).collect(n=None)
    baselines = baselines_from_reference(reference.selected)
    candidate = gameplay_pipeline(cand_rows, cand_frames, baselines=baselines).collect(n=None)
    action = _action_decision(reference.selected, candidate.selected, baselines)

    ref_traj = trajectory_pipeline(trajectory_drop("reference")).collect(n=None)
    cand_traj = trajectory_pipeline(trajectory_drop(scenario)).collect(n=None)
    spatial = _spatial_decision(ref_traj.selected, cand_traj.selected)
    decision = _overall(action, spatial)

    keeps = []
    if decision != "reject":
        for control in CONTROLS:
            count = _control_keep_count(candidate.selected, control)
            # On accept, keep the on-contract windows. On repair, keep only those too.
            shard = _keep_shard(cand_rows, cand_frames, baselines, out, control=control, n=count)
            if shard is not None:
                keeps.append(shard)

    scorecard = {
        "scenario": scenario,
        "decision": decision,
        "action_contract": action,
        "spatial_coverage": spatial,
        "keep_shards": keeps,
        "sdk_gaps": SDK_GAPS,
    }
    (out / "scorecard.json").write_text(json.dumps(scorecard, indent=2) + "\n")
    return scorecard


def render_scorecard(scorecard: dict) -> str:
    action = scorecard["action_contract"]
    spatial = scorecard["spatial_coverage"]
    ratios = action["magnitude_ratio"]
    shifts = action["shift_median"]
    lines = [
        f"scenario:  {scorecard['scenario']}",
        f"decision:  {scorecard['decision']}",
        f"action:    {action['label']}  "
        f"in-band {action['in_band_windows']}/{action['measured_windows']}  "
        f"anomaly {action['anomaly_fraction']:.0%}",
        f"W shift:   reference {shifts['reference']['W']:.3f}  "
        f"candidate {shifts['candidate']['W']:.3f}  ratio {ratios['W']:.2f}",
        f"S shift:   reference {shifts['reference']['S']:.3f}  "
        f"candidate {shifts['candidate']['S']:.3f}  ratio {ratios['S']:.2f}",
        f"S/W:       reference {action['symmetry_s_over_w']['reference']:.2f}  "
        f"candidate {action['symmetry_s_over_w']['candidate']:.2f}",
        f"spatial:   {spatial['label']}  "
        f"turn ratio {spatial['turn_ratio']:.2f}  "
        f"quality pass {spatial['quality_pass_fraction']:.0%}  "
        f"max jump {spatial['max_translation_jump_fraction']:.2f}",
        f"shards:    {len(scorecard['keep_shards'])}",
    ]
    return "\n".join(lines)
