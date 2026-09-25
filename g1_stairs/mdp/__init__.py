"""Shared MDP term library: rewards, observations, curriculums, terminations.

Derived from unitree_rl_mjlab src/tasks/velocity/mdp (Apache-2.0). Also re-exports
mjlab.envs.mdp. Put terms that several variants may use here; keep terms used by
one variant in that variant's folder.
"""

from mjlab.envs.mdp import *  # noqa: F401, F403

from .curriculums import *  # noqa: F403
from .observations import *  # noqa: F403
from .rewards import *  # noqa: F403
from .terminations import *  # noqa: F403
