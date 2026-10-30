"""PLM backends: the protocol (`base`), the registry, and implementations.

The catalog core imports only `base` and `registry`; concrete backends are
loaded by name, so their optional dependencies are needed only when used.
"""
