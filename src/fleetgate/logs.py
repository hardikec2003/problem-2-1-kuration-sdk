"""Synthetic fleet logs: long transit, rare contact, and one camera teleport."""

from __future__ import annotations

import math

from kuration import Tick

HZ = 10.0
WINDOW = 32
N_WINDOWS = 8
# Speeds are meters per tick. Contact windows use different speeds so path length
# actually spans more than one bin.
CONTACT = {
    "picker": {4: 0.05, 6: 0.28},
    "docker": {4: 0.12, 6: 0.42},
    "aisle": {5: 0.20},
    "glitch": {3: 0.15, 6: 0.10},
}
JUMP_WINDOW = {"glitch": 3}


def build_fleet() -> dict[str, list[Tick]]:
    """One shift per robot. Contact is a logged 0/1 field, the way a fleet already records it."""
    return {name: _shift(name, contacts, JUMP_WINDOW.get(name)) for name, contacts in CONTACT.items()}


def _shift(name: str, contacts: dict[int, float], jump_window: int | None) -> list[Tick]:
    ticks: list[Tick] = []
    for window in range(N_WINDOWS):
        speed = contacts.get(window, 0.04)
        turn = 0.08 if window in contacts else 0.015
        contact = 1.0 if window in contacts else 0.0
        for step_in_window in range(WINDOW):
            step = window * WINDOW + step_in_window
            angle = turn * step
            x = speed * step
            if jump_window == window and step_in_window >= WINDOW // 2:
                x += 40.0
            ticks.append(Tick(time=step / HZ, entities={0: {
                "camera_position": (x, math.sin(angle), 0.0),
                "camera_z_axis": (math.cos(angle), math.sin(angle), 0.0),
                "tracking_confidence": 0.95,
                "contact": contact,
                "command": (0.0 if contact else speed, 0.0, 1.0 if contact else 0.0),
            }}))
    return ticks
