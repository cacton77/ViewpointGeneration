"""The catalog core over the in-memory backend: capabilities and identity."""

import json

import pytest

from catalog_fakes import results_for

from viewpoint_generation import plan_io
from viewpoint_generation.catalog.backends.base import Capability
from viewpoint_generation.catalog.backends.fake import ALL_CAPABILITIES, FakeBackend
from viewpoint_generation.catalog.core.catalog import Catalog
from viewpoint_generation.catalog.core.config import CoreConfig
from viewpoint_generation.catalog.core.paths import CatalogPaths, storage_key


@pytest.fixture
def make_catalog(tmp_path):
    def _make(*backends, root='catalog', **config):
        core = CoreConfig(root=str(tmp_path / root), render_thumbnails=False, **config)
        return Catalog({b.name: b for b in backends}, core, CatalogPaths.from_env(tmp_path / root))
    return _make


def _record(catalog, backend, part_id, stage='ordered', upload=None):
    """Write and record a plan as the VG node does, embedding the catalog's context."""
    part = catalog.db.get_part(backend, part_id)
    results = results_for(stage)
    results['plm_context'] = json.loads(catalog.plm_context_json(part))
    path, _ = plan_io.write_plan(catalog.plan_dir_for(part), results, stage=stage)
    description = plan_io.describe_plan(path)
    return catalog.record_plan(backend, part_id, path, stage=stage,
                               stage_rank=description['stage_rank'],
                               summary_text=description['summary_text'], upload=upload)


def test_two_backends_keep_separate_identities(make_catalog):
    a, b = FakeBackend('a'), FakeBackend('b')
    a.add_part('P1', title='From A')
    b.add_part('P1', title='From B')        # the same part_id in another PLM
    catalog = make_catalog(a, b)
    results = catalog.run_full_sync()
    assert [(r.backend, r.added) for r in results] == [('a', 1), ('b', 1)]
    assert catalog.db.get_part('a', 'P1')['title'] == 'From A'
    assert catalog.db.get_part('b', 'P1')['title'] == 'From B'
    assert storage_key('a', 'P1') != storage_key('b', 'P1')
    [only_b] = catalog.run_full_sync('b')
    assert only_b.backend == 'b' and only_b.unchanged == 1


def test_part_ids_never_become_paths(make_catalog):
    backend = FakeBackend()
    backend.add_part('doc/123/w/456/e/789/p/JHD')
    catalog = make_catalog(backend)
    catalog.run_full_sync()
    path, _ = catalog.fetch_geometry('fake', 'doc/123/w/456/e/789/p/JHD')
    key = storage_key('fake', 'doc/123/w/456/e/789/p/JHD')
    assert path.parent == catalog.paths.geometry_dir(key)
    assert '/' not in key and 'doc' not in str(path.relative_to(catalog.paths.root))
    plan_id, _ = _record(catalog, 'fake', 'doc/123/w/456/e/789/p/JHD')
    assert catalog.resolved_plan_path(catalog.db.get_plan(plan_id)).parent == \
        catalog.paths.plans_for(key)


def test_without_attach_plans_stay_local(make_catalog):
    backend = FakeBackend(capabilities=ALL_CAPABILITIES - {Capability.ATTACH})
    backend.add_part('P')
    catalog = make_catalog(backend)
    catalog.run_full_sync()
    plan_id, message = _record(catalog, 'fake', 'P')  # 'ordered' auto-uploads by default
    assert 'stays on this cell' in message
    assert catalog.db.get_plan(plan_id)['upload_status'] == 'local'
    assert backend.attachments == []


def test_without_geometry_the_part_is_marked_and_hand_placed_steps_work(make_catalog):
    backend = FakeBackend(capabilities=ALL_CAPABILITIES - {Capability.GEOMETRY})
    backend.add_part('P')
    catalog = make_catalog(backend)
    catalog.run_full_sync()
    part = catalog.db.get_part('fake', 'P')
    assert part['geometry_available'] == 0
    assert catalog.fetch_geometry('fake', 'P')[0] is None
    directory = catalog.paths.geometry_dir(part['storage_key'])
    directory.mkdir(parents=True)
    (directory / 'mine.stp').write_text('ISO-10303-21;')
    assert catalog.fetch_geometry('fake', 'P')[0] == directory / 'mine.stp'


def test_thumbnails_come_from_the_backend_only_when_declared(make_catalog):
    with_thumbs = FakeBackend('with')
    with_thumbs.add_part('P')
    with_thumbs.thumbnails['P'] = b'\x89PNG'
    without = FakeBackend('without', capabilities=ALL_CAPABILITIES - {Capability.THUMBNAILS})
    without.add_part('P')
    without.thumbnails['P'] = b'\x89PNG'
    catalog = make_catalog(with_thumbs, without)
    catalog.run_full_sync()
    assert catalog.resolved_thumb_path(catalog.db.get_part('with', 'P')) is not None
    assert catalog.resolved_thumb_path(catalog.db.get_part('without', 'P')) is None


def test_truncated_listing_archives_nothing(make_catalog):
    backend = FakeBackend()
    for i in range(4):
        backend.add_part(f'P{i}')
    catalog = make_catalog(backend)
    catalog.run_full_sync()
    backend.scope_budget = 2
    [result] = catalog.run_full_sync()
    assert result.archived == 0 and 'truncated' in result.error_details[0]


def test_maturity_filter_applies_to_every_backend(make_catalog):
    backend = FakeBackend()
    backend.add_part('R', maturity='RELEASED')
    backend.add_part('W', maturity='IN_WORK')
    catalog = make_catalog(backend, maturity_filter='RELEASED')
    catalog.run_full_sync()
    assert set(catalog.db.get_all_parts('fake')) == {'R'}


def test_events_drive_single_part_syncs(make_catalog):
    backend = FakeBackend()
    backend.add_part('P')
    catalog = make_catalog(backend)
    catalog.run_full_sync()
    seen = []
    assert catalog.start_events(on_synced=lambda *args: seen.append(args)) == ['fake']
    backend.touch('P', 't2')
    backend.emit('P')
    assert catalog.db.get_part('fake', 'P')['change_token'] == 't2'
    assert seen[0][:2] == ('fake', 'P') and seen[0][2][0] is True
    catalog.stop_events()
    assert backend.listener is None


def test_no_events_without_the_capability(make_catalog):
    backend = FakeBackend(capabilities=ALL_CAPABILITIES - {Capability.EVENTS})
    assert make_catalog(backend).start_events() == []


def test_remote_plans_are_found_through_linked_attachments(make_catalog):
    backend = FakeBackend()
    backend.add_part('P', change_token='t1')
    first_cell = make_catalog(backend, root='one')
    first_cell.run_full_sync()
    _record(first_cell, 'fake', 'P')          # 'ordered' auto-uploads
    assert len(backend.attachments) == 1

    # A second cell finds it through the part, and reads back only the
    # plm_context block the catalog authored.
    second_cell = make_catalog(backend, root='two')
    second_cell.run_full_sync()
    status, path = second_cell.ensure_plan('fake', 'P')
    assert status.value == 'DOWNLOADED'
    row = second_cell.db.get_current_plan('fake', 'P')
    assert (row['source'], row['change_token'], row['stage']) == ('remote', 't1', 'ordered')

    backend.touch('P', 't2')
    third_cell = make_catalog(backend, root='three')
    third_cell.run_full_sync()
    assert third_cell.ensure_plan('fake', 'P')[0].value == 'STALE'


def test_no_remote_lookup_without_link_attachment(make_catalog):
    backend = FakeBackend(capabilities=ALL_CAPABILITIES - {Capability.LINK_ATTACHMENT})
    backend.add_part('P')
    first_cell = make_catalog(backend, root='one')
    first_cell.run_full_sync()
    _record(first_cell, 'fake', 'P')
    second_cell = make_catalog(backend, root='two')
    second_cell.run_full_sync()
    assert second_cell.ensure_plan('fake', 'P')[0].value == 'NONE'


def test_backend_info_reports_capabilities_and_connection(make_catalog):
    backend = FakeBackend(capabilities={Capability.GEOMETRY})
    catalog = make_catalog(backend)
    catalog.run_full_sync()
    [info] = catalog.backend_info()
    assert info['name'] == 'fake'
    assert info['capabilities'] == ['geometry']
    assert info['connected'] is True


def test_a_failed_write_reaches_the_operator(make_catalog):
    backend = FakeBackend()
    backend.add_part('P')
    backend.fail_writes = True
    catalog = make_catalog(backend)
    catalog.run_full_sync()
    plan_id, message = _record(catalog, 'fake', 'P')
    assert 'refused the upload' in message
    assert catalog.db.get_plan(plan_id)['upload_status'] == 'failed'
