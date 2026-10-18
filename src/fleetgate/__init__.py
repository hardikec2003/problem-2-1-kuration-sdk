"""FleetGate: turn a robot fleet's raw logs into a budget of trainable hours.

Most recorded time is transit. Contact, recovery, and near-misses are rare, and a
camera glitch can look like a useful maneuver. FleetGate keeps the contact hours,
drops impossible camera jumps, and writes a checksummed dataset that refills to
the same budget from the same config.
"""

__version__ = "0.1.0"
