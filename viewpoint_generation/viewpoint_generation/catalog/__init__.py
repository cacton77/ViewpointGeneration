"""3DEXPERIENCE parts catalog subsystem.

Keeps a local SQLite index of engineering items published on a 3DEXPERIENCE
(ENOVIA) tenant, caches their STEP files and thumbnails on disk, and exchanges
inspection plans and results with the PLM record.

Modules:
    config  -- environment-driven connection/sync/path configuration
    schema  -- SQLite table definitions and the CatalogDB access layer
    client  -- authenticated REST session against the 3DX tenant (DXClient)
    sync    -- catalog synchronization daemon (CatalogSync)
    jms     -- optional event-driven sync via a JMS/STOMP bridge

Import from the submodules directly, e.g.::

    from viewpoint_generation.catalog.sync import CatalogSync

Nothing is re-exported here on purpose: `sync` doubles as the
`python -m viewpoint_generation.catalog.sync` CLI entry point, and importing it
from this package's `__init__` would load the module twice under runpy.
"""

