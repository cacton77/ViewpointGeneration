"""Executable entry point for the 3DEXPERIENCE parts catalog node.

Mirrors the other node scripts in this package: the implementation lives in
the library (`viewpoint_generation.catalog.ros_node`), and this module only
provides the console-script entry point declared in setup.py.
"""

from viewpoint_generation.catalog.ros_node import main

if __name__ == '__main__':
    main()
