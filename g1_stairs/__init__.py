"""Shared MuJoCo environment for Unitree G1 23-DOF stair climbing.

Import ``g1_stairs.tasks`` to register every G1-Stairs-* task.
"""

from pathlib import Path

PACKAGE_ROOT: Path = Path(__file__).resolve().parent
REPO_ROOT: Path = PACKAGE_ROOT.parent
