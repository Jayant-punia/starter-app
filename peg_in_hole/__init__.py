"""peg_in_hole — MuJoCo peg insertion research baseline.

Run:
    python -m peg_in_hole
    python peg_in_hole.py
"""

from .config import DEFAULT_CONFIG, PegInHoleConfig
from .simulate import load_sim, main, run

__all__ = [
    "DEFAULT_CONFIG",
    "PegInHoleConfig",
    "load_sim",
    "main",
    "run",
]

__version__ = "0.1.0"
