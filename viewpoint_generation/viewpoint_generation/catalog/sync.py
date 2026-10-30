"""Compatibility entry point: `python -m viewpoint_generation.catalog.sync`.

The command line moved to `catalog.cli`; this name is kept because operator
notes and the .env example refer to it.
"""

import sys

from viewpoint_generation.catalog.cli import main

if __name__ == '__main__':
    sys.exit(main())
