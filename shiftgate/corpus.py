"""Synthetic DIAMOND-format clips whose forward motion is a known zoom.

Pixel-shift measures the radial optical-flow at an isolated W or S release.
Zooming in yields a positive shift; zooming out yields a negative one. A
reference corpus uses a strong, symmetric pair. Candidate scenarios weaken,
flip, or keep that pair so the gate has an accept, a repair, and a reject.
"""

from __future__ import annotations

import math

import numpy as np

from kuration import Tick
from kuration.gameplay.pixel_shift.diamond_action_decode import (
    ACTION_WIDTH,
    KEY_NAMES,
    MOUSE_X_POSSIBLES,
    MOUSE_Y_POSSIBLES,
    MX_OFF,
    MY_OFF,
)

FRAME_H = 48
FRAME_W = 64
N_FRAMES = 16
PRESS_W = 3
PRESS_S = 11
WINDOW = 8
STRIDE = 8

# Scales probed against Farneback. 1.10 is a salient forward expansion;
# 0.90 is a salient contraction; 1.02 is a weak expansion the contract rejects.
HEALTHY_W = 1.10
HEALTHY_S = 0.90
WEAK_W = 1.02
WEAK_S = 0.98
INVERTED_W = 0.90
INVERTED_S = 1.10


def action_vector(*, key: str | None = None) -> np.ndarray:
    """One 51-dim DIAMOND action. Mouse bins stay strict one-hot at rest."""
    vec = np.zeros(ACTION_WIDTH, dtype=np.float64)
    if key is not None:
        vec[KEY_NAMES.index(key)] = 1.0
    vec[MX_OFF + int(np.argmin(np.abs(MOUSE_X_POSSIBLES - 0.0)))] = 1.0
    vec[MY_OFF + int(np.argmin(np.abs(MOUSE_Y_POSSIBLES - 0.0)))] = 1.0
    return vec


def _frame(base: np.ndarray, scale: float) -> np.ndarray:
    ch = max(4, int(round(FRAME_H / scale)))
    cw = max(4, int(round(FRAME_W / scale)))
    y0 = (base.shape[0] - ch) // 2
    x0 = (base.shape[1] - cw) // 2
    patch = base[y0 : y0 + ch, x0 : x0 + cw]
    ys = np.linspace(0, ch - 1, FRAME_H).astype(np.int64)
    xs = np.linspace(0, cw - 1, FRAME_W).astype(np.int64)
    return np.ascontiguousarray(patch[ys][:, xs])


def make_clip(seed: int, w_scale: float, s_scale: float) -> tuple[list[dict], list[np.ndarray]]:
    """Return row dicts and RGB frames for one clip with an isolated W and S press."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(FRAME_H * 4, FRAME_W * 4, 3), dtype=np.uint8)
    idle = _frame(base, 1.0)
    frames = [idle.copy() for _ in range(N_FRAMES)]
    frames[PRESS_W + 1] = _frame(base, w_scale)
    frames[PRESS_S + 1] = _frame(base, s_scale)

    rows = []
    for t in range(N_FRAMES):
        key = "W" if t == PRESS_W else "S" if t == PRESS_S else None
        vec = action_vector(key=key)
        row = {
            "t": t,
            "pressed": 1.0 if key else 0.0,
            **{f"a{i}": float(v) for i, v in enumerate(vec)},
        }
        rows.append(row)
    return rows, frames


def gameplay_drop(kind: str) -> tuple[dict[str, list[dict]], dict[str, list[np.ndarray]]]:
    """Build a named drop. ``reference`` is the known-playable contract source."""
    specs = {
        "reference": [("healthy", HEALTHY_W, HEALTHY_S)] * 6,
        "accept": [("healthy", HEALTHY_W, HEALTHY_S)] * 6,
        "repair": (
            [("healthy", HEALTHY_W, HEALTHY_S)] * 4
            + [("weak", WEAK_W, WEAK_S)] * 2
            + [("inverted", INVERTED_W, INVERTED_S)] * 2
        ),
        "reject": [("weak", WEAK_W, WEAK_S)] * 3 + [("inverted", INVERTED_W, INVERTED_S)] * 3,
    }
    if kind not in specs:
        raise ValueError(f"unknown drop {kind!r}")
    rows: dict[str, list[dict]] = {}
    frames: dict[str, list[np.ndarray]] = {}
    kind_seed = {"reference": 1, "accept": 2, "repair": 3, "reject": 4}[kind]
    for index, (label, w_scale, s_scale) in enumerate(specs[kind]):
        source = f"{kind}-{label}-{index:02d}"
        rows[source], frames[source] = make_clip(10_000 + kind_seed * 100 + index * 17, w_scale, s_scale)
    return rows, frames


def camera_path(*, speed: float, turn_rate: float, confidence: float, teleport_at: int | None = None) -> list[Tick]:
    """One camera trajectory. A teleport is an impossible step SLAM quality should flag."""
    ticks = []
    for step in range(32):
        angle = turn_rate * step
        x = speed * step
        if teleport_at is not None and step >= teleport_at:
            x += 40.0
        ticks.append(Tick(time=step / 10.0, entities={0: {
            "camera_position": (x, math.sin(angle), 0.05 * math.cos(angle)),
            "camera_z_axis": (math.cos(angle), math.sin(angle), 0.0),
            "tracking_confidence": confidence if step not in {14, 15} else confidence / 2,
        }}))
    return ticks


def trajectory_drop(kind: str) -> dict[str, list[Tick]]:
    """Reference paths cover turn-rate bins. Reject adds a teleport and a camped path."""
    diverse = {
        f"{kind}-slow": camera_path(speed=0.03, turn_rate=0.03, confidence=0.98),
        f"{kind}-fast": camera_path(speed=0.12, turn_rate=0.25, confidence=0.92),
        f"{kind}-arc": camera_path(speed=0.07, turn_rate=0.12, confidence=0.95),
    }
    if kind == "reject":
        return {
            f"{kind}-camp-a": camera_path(speed=0.01, turn_rate=0.01, confidence=0.97),
            f"{kind}-camp-b": camera_path(speed=0.01, turn_rate=0.012, confidence=0.97),
            f"{kind}-teleport": camera_path(speed=0.05, turn_rate=0.04, confidence=0.9, teleport_at=16),
        }
    return diverse
