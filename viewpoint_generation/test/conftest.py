"""Shared fixtures for the catalog tests.

Nothing here touches the network or ROS. `catalog` is the multi-backend core
hosting one dx3 backend whose REST client is `FakeDXClient`, an in-memory
tenant, so the 3DX scoping and paging behaviour stays pinned. Every catalog
lives under pytest's tmp_path.
"""

from pathlib import Path

import pytest

from catalog_fakes import FakeDXClient, results_for
from viewpoint_generation import plan_io
from viewpoint_generation.catalog.backends.dx3.backend import DX3Backend
from viewpoint_generation.catalog.backends.dx3.config import DX3Settings
from viewpoint_generation.catalog.core.catalog import Catalog
from viewpoint_generation.catalog.core.config import CoreConfig
from viewpoint_generation.catalog.core.paths import CatalogPaths


@pytest.fixture
def fake_client():
    return FakeDXClient()


@pytest.fixture
def catalog_paths(tmp_path):
    return CatalogPaths.from_env(tmp_path / 'catalog')


@pytest.fixture
def core_config(tmp_path):
    return CoreConfig(backends='dx3', sync_interval=0, render_thumbnails=False,
                      auto_upload_plan_stages='ordered', root=str(tmp_path / 'catalog'))


@pytest.fixture
def dx3_settings():
    return DX3Settings(bookmark_scope='*', use_bookmark_scope=True, page_size=100,
                       max_items=1000, fetch_thumbnails=True, document_links=False)


@pytest.fixture
def dx3(fake_client, dx3_settings):
    return DX3Backend(fake_client.config, dx3_settings, client=fake_client)


@pytest.fixture
def catalog(dx3, core_config, catalog_paths):
    return Catalog({'dx3': dx3}, core_config, catalog_paths)


@pytest.fixture
def plan_dir():
    """The directory the catalog hands the producer for a dx3 part's plans."""
    def _plan_dir(catalog, part_id, backend='dx3'):
        return catalog.plan_dir_for(catalog.db.get_part(backend, part_id))
    return _plan_dir


@pytest.fixture
def write_plan_file():
    """Write a results dict for a stage into a directory, as the pipeline does."""
    def _write(directory, stage, **kwargs):
        path, message = plan_io.write_plan(directory, results_for(stage, **kwargs),
                                           stage=stage)
        assert path is not None, message
        return Path(path)
    return _write


@pytest.fixture
def record():
    """Record a plan the way the VG node does: describe it, then RecordPlan."""
    def _record(catalog, part_id, path, backend='dx3', **kwargs):
        description = plan_io.describe_plan(path)
        return catalog.record_plan(
            backend, part_id, path, stage=description['stage'],
            stage_rank=description['stage_rank'],
            summary_json=description['summary_json'],
            summary_text=description['summary_text'],
            change_token=description['change_token'],
            generated_at=description['generated_at'], **kwargs)
    return _record
