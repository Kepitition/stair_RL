# Derived from unitree_rl_mjlab scripts/list_envs.py (Apache-2.0).
"""List registered tasks. Usage: python scripts/list_envs.py [--keyword G1-Stairs]"""

import mjlab
import tyro
from mjlab.tasks.registry import list_tasks
from prettytable import PrettyTable

import g1_stairs.tasks  # noqa: F401  (registers the G1-Stairs-* tasks)
from g1_stairs.runtime import configure_utf8_output


def list_environments(keyword: str = "G1-Stairs") -> int:
  """Print registered task IDs containing ``keyword`` (use "" for all)."""
  table = PrettyTable(["#", "Task ID"])
  table.title = "Registered tasks"
  table.align["Task ID"] = "l"
  tasks = [t for t in list_tasks() if keyword.lower() in t.lower()]
  for i, task in enumerate(tasks, start=1):
    table.add_row([i, task])
  print(table)
  return len(tasks)


def main() -> None:
  configure_utf8_output()
  tyro.cli(list_environments, config=mjlab.TYRO_FLAGS)


if __name__ == "__main__":
  main()
