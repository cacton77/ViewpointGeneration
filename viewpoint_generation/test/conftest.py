"""Shared fixtures for the catalog characterization tests.

Nothing here touches the network or ROS: the fake client keeps an in-memory
tenant, and every catalog lives under pytest's tmp_path.
"""

from pathlib import Path

import pytest

from catalog_fakes import FakeDXClient, results_for
from viewpoint_generation import plan_io
from viewpoint_generation.catalog.config import CatalogPaths, SyncConfig
from viewpoint_generation.catalog.sync import CatalogSync


@pytest.fixture
def fake_client():
    return FakeDXClient()


@pytest.fixture
def catalog_paths(tmp_path):
    root = tmp_path / 'catalog'
    return CatalogPaths(db_path=root / 'catalog.db', step_dir=root / 'steps',
                        thumb_dir=root / 'thumbnails', plan_dir=root / 'plans',
                        result_dir=root / 'results')


@pytest.fixture
def sync_config():
    return SyncConfig(sync_interval=0, bookmark_scope='*', use_bookmark_scope=True,
                      page_size=100, max_items=1000, fetch_thumbnails=True,
                      render_thumbnails=False, auto_upload_plan_stages='ordered')


@pytest.fixture
def catalog(fake_client, catalog_paths, sync_config):
    return CatalogSync(fake_client, catalog_paths.db_path, sync_config, catalog_paths)


@pytest.fixture
def write_plan_file():
    """Write a results dict for a stage into a directory, as the pipeline does."""
    def _write(directory, stage, **kwargs):
        path, message = plan_io.write_plan(directory, results_for(stage, **kwargs),
                                           stage=stage)
        assert path is not None, message
        return Path(path)
    return _write
