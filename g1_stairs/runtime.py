"""Helpers shared by scripts/*.py: log paths, rendering backend, viewer, dummy policies."""

import os
import sys
from pathlib import Path

import torch

from g1_stairs import REPO_ROOT

LOG_ROOT: Path = REPO_ROOT / "logs" / "rsl_rl"


def configure_utf8_output() -> None:
  """Write stdout/stderr as UTF-8, so logs redirected to a file don't crash on Windows.

  Redirected output on Windows defaults to cp1252, which cannot encode characters
  that tyro's help and mjlab's startup tables print (e.g. "→", "←", "×").
  """
  for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
      stream.reconfigure(encoding="utf-8", errors="replace")


def configure_render_backend() -> None:
  """EGL for headless rendering on Linux. Windows and macOS keep MuJoCo's default.

  Upstream unitree_rl_mjlab sets MUJOCO_GL=egl on every platform, which breaks
  rendering on Windows. A value the user already set always wins.
  """
  if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")


def default_viewer() -> str:
  """'native' (MuJoCo window) on Windows or with a display, else 'viser' (browser)."""
  if sys.platform == "win32":
    return "native"
  has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
  return "native" if has_display else "viser"


def resolve_device(device: str | None) -> str:
  return device or ("cuda:0" if torch.cuda.is_available() else "cpu")


def zero_policy(env):
  """Policy that always outputs 0: the robot holds its default pose."""
  shape = env.unwrapped.action_space.shape
  device = env.unwrapped.device

  def policy(obs):
    del obs
    return torch.zeros(shape, device=device)

  return policy


def random_policy(env):
  """Policy that outputs uniform noise in [-1, 1]."""
  shape = env.unwrapped.action_space.shape
  device = env.unwrapped.device

  def policy(obs):
    del obs
    return 2 * torch.rand(shape, device=device) - 1

  return policy


def run_viewer(env, policy, viewer: str) -> None:
  from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

  if viewer == "native":
    NativeMujocoViewer(env, policy).run()
  elif viewer == "viser":
    ViserPlayViewer(env, policy).run()
  else:
    raise ValueError(f"Unknown viewer {viewer!r}; use 'native' or 'viser'.")
