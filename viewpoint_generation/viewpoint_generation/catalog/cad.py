"""The catalog's one route to CAD loading.

Thumbnail rendering is the only thing in the catalog that reads geometry, and
it does so through this module alone, so moving the STEP loader elsewhere is a
one-line change here.
"""

from viewpoint_generation.step_loader import load_step  # noqa: F401
