"""PLM parts catalog: a local index of parts held in PLM systems.

Keeps a SQLite index of the parts each configured backend reports, caches
their STEP files and thumbnails, and exchanges inspection plans and results
with the PLM record.

Layout (the shape the code will have as the standalone `plm_bridge` package):

    core/       -- Catalog (reconciliation, plans, runs), CatalogDB, paths,
                   CoreConfig, the plan contract. Imports only backends.base.
    backends/   -- the PLMBackend protocol (base), the registry, a fake, and
                   dx3/ (3DEXPERIENCE: client, config, backend, JMS events)
    factory     -- builds a Catalog from the environment via the registry
    cli         -- `python -m viewpoint_generation.catalog.cli`
    thumbnails  -- local preview rendering (through cad, the loader shim)
    ros_node    -- the ROS 2 node (services, topics, the picker)

Import from the submodules directly; nothing is re-exported here.
"""
