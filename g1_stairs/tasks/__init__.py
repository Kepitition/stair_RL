"""Task registry.

Every subpackage of g1_stairs.tasks is imported here, and each one registers its
own task IDs. To add a variant, copy tasks/baseline/ to tasks/<name>/ and rename
its task ID; no other file needs editing.
"""

from mjlab.utils.lab_api.tasks.importer import import_packages

import_packages(__name__, blacklist_pkgs=[])
