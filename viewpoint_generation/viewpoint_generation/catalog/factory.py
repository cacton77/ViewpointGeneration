"""Assemble a `Catalog` from the environment: the core plus its backends.

The only place the core meets the registry, so the core itself never names a
concrete backend.
"""

import logging

from viewpoint_generation.catalog.backends import registry
from viewpoint_generation.catalog.core.catalog import Catalog
from viewpoint_generation.catalog.core.config import CoreConfig
from viewpoint_generation.catalog.core.paths import CatalogPaths

logger = logging.getLogger(__name__)


def build_catalog(config=None, backends=None, paths=None):
    """A Catalog hosting the configured backends.

    Args:
        config: A `CoreConfig`; from the environment when omitted.
        backends: Mapping name -> backend, to bypass the registry (tests).
        paths: A `CatalogPaths`; derived from `config.root` when omitted.

    Returns:
        tuple: (Catalog, list of error messages from backends that failed
        to start). A broken backend never stops the others.
    """
    config = config or CoreConfig.from_env()
    errors = []
    if backends is None:
        backends, errors = registry.create_all(config.backend_names())
        for message in errors:
            logger.error(message)
    return Catalog(backends, config, paths or CatalogPaths.from_env(config.root)), errors
