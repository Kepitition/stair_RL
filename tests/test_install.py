"""The install must give a CUDA torch and the pinned simulator stack (spec §4)."""

from importlib.metadata import version

import torch

import g1_stairs


def test_torch_is_a_cuda_build():
  # CPU torch means requirements.txt was bypassed or torchvision was unpinned (spec §2).
  assert torch.version.cuda is not None, (
    f"torch {torch.__version__} is a CPU-only build; reinstall with "
    "`pip install -r requirements.txt`"
  )


def test_pinned_versions():
  assert torch.__version__.startswith("2.9.0")
  assert version("torchvision").startswith("0.24.0")
  assert version("mjlab") == "1.2.0"
  assert version("mujoco") == "3.5.0"
  assert version("mujoco-warp") == "3.5.0"
  assert version("warp-lang") == "1.12.0"
  assert version("rsl-rl-lib") == "5.0.1"


def test_repo_root_points_at_the_checkout():
  assert (g1_stairs.REPO_ROOT / "pyproject.toml").is_file()
  assert g1_stairs.PACKAGE_ROOT == g1_stairs.REPO_ROOT / "g1_stairs"
