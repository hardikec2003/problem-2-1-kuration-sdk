"""ShiftGate: accept, repair, or reject a gameplay drop before world-model training.

A delivery can match map, frame rate, and action schema and still teach an
uncontrollable world model. ShiftGate freezes the action-to-motion contract of a
known-playable reference, scores a new drop with Kuration, and writes a
checksummed keep-shard only from windows that sit on that contract.
"""

__version__ = "0.1.0"
