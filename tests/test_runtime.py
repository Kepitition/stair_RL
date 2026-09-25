"""Script helpers: Windows-safe rendering, viewer choice, cwd-independent paths (spec §7)."""

import os
import subprocess
import sys

import pytest

from g1_stairs import REPO_ROOT, runtime


@pytest.fixture
def no_mujoco_gl(monkeypatch):
  """Unset MUJOCO_GL, and restore its original state after the test.

  setenv first so monkeypatch records the original value: configure_render_backend()
  sets MUJOCO_GL outside monkeypatch, and a leaked "egl" breaks every later mjlab
  import on Windows.
  """
  monkeypatch.setenv("MUJOCO_GL", "set-by-test")
  monkeypatch.delenv("MUJOCO_GL")


def test_render_backend_is_not_forced_on_windows(monkeypatch, no_mujoco_gl):
  monkeypatch.setattr(sys, "platform", "win32")
  runtime.configure_render_backend()
  assert "MUJOCO_GL" not in os.environ


def test_render_backend_is_egl_on_linux(monkeypatch, no_mujoco_gl):
  monkeypatch.setattr(sys, "platform", "linux")
  runtime.configure_render_backend()
  assert os.environ["MUJOCO_GL"] == "egl"


def test_render_backend_respects_user_choice(monkeypatch):
  monkeypatch.setattr(sys, "platform", "linux")
  monkeypatch.setenv("MUJOCO_GL", "osmesa")
  runtime.configure_render_backend()
  assert os.environ["MUJOCO_GL"] == "osmesa"


def test_default_viewer(monkeypatch):
  monkeypatch.setattr(sys, "platform", "win32")
  monkeypatch.delenv("DISPLAY", raising=False)
  assert runtime.default_viewer() == "native"
  monkeypatch.setattr(sys, "platform", "linux")
  monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
  assert runtime.default_viewer() == "viser"
  monkeypatch.setenv("DISPLAY", ":0")
  assert runtime.default_viewer() == "native"


def test_resolve_device():
  assert runtime.resolve_device("cpu") == "cpu"
  assert runtime.resolve_device(None) in ("cuda:0", "cpu")


def test_logs_live_in_the_repo():
  assert runtime.LOG_ROOT == REPO_ROOT / "logs" / "rsl_rl"


def test_scripts_survive_redirected_output():
  # On Windows, piped/redirected stdout defaults to cp1252; tyro help and mjlab's
  # startup tables print characters like "→" and "←" that cp1252 cannot encode.
  env = {k: v for k, v in os.environ.items() if k not in ("PYTHONUTF8", "PYTHONIOENCODING")}
  result = subprocess.run(
    [sys.executable, str(REPO_ROOT / "scripts" / "train.py"), "G1-Stairs-Baseline", "--help"],
    capture_output=True,
    text=True,
    encoding="utf-8",
    env=env,
    timeout=300,
  )
  assert result.returncode == 0, result.stderr[-2000:]
  assert "max-iterations" in result.stdout


def test_list_envs_script_from_another_directory(tmp_path):
  result = subprocess.run(
    [sys.executable, str(REPO_ROOT / "scripts" / "list_envs.py")],
    cwd=tmp_path,
    capture_output=True,
    text=True,
    timeout=300,
  )
  assert result.returncode == 0, result.stderr
  assert "G1-Stairs-Baseline" in result.stdout
