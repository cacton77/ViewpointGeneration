"""Reconciliation through the dx3 backend (FakeDXClient: no network)."""

from catalog_fakes import doc_member, eng_member, make_item

from viewpoint_generation.catalog.core.paths import storage_key


def _add(client, *items):
    for item in items:
        client.items[item['eng_item_id']] = item


def _sync(catalog):
    [result] = catalog.run_full_sync()
    return result


def test_full_sync_adds_new_parts(catalog, fake_client):
    _add(fake_client, make_item('A'), make_item('B'))
    result = _sync(catalog)
    assert result.success and result.backend == 'dx3'
    assert (result.added, result.modified, result.unchanged, result.archived) == (2, 0, 0, 0)
    parts = catalog.db.get_all_parts('dx3')
    assert set(parts) == {'A', 'B'}
    assert parts['A']['sync_status'] == 'new'
    assert parts['A']['change_token'] == 'c1'
    assert parts['A']['container'] == 'Test Space'
    assert parts['A']['storage_key'] == storage_key('dx3', 'A')


def test_second_sync_reports_everything_unchanged(catalog, fake_client):
    _add(fake_client, make_item('A'), make_item('B'))
    _sync(catalog)
    result = _sync(catalog)
    assert (result.added, result.modified, result.unchanged) == (0, 0, 2)
    assert {p['sync_status'] for p in catalog.db.get_all_parts('dx3').values()} == {'synced'}


def test_modified_part_drops_cached_step_and_marks_plans_stale(
        catalog, fake_client, write_plan_file, record, plan_dir):
    _add(fake_client, make_item('A'))
    fake_client.step_files['A'] = b'ISO-10303-21; dummy'
    _sync(catalog)
    step_path, _ = catalog.fetch_geometry('dx3', 'A')
    assert step_path is not None and step_path.exists()
    plan_file = write_plan_file(plan_dir(catalog, 'A'), 'ordered')
    record(catalog, 'A', plan_file, upload=False)
    assert catalog.ensure_plan('dx3', 'A', check_remote=False)[0].value == 'CURRENT'

    fake_client.items['A']['cestamp'] = 'c2'
    result = _sync(catalog)

    assert result.modified == 1
    part = catalog.db.get_part('dx3', 'A')
    assert part['sync_status'] == 'modified'
    assert part['change_token'] == 'c2'
    assert part['geometry_path'] is None
    assert not step_path.exists()
    status, path = catalog.ensure_plan('dx3', 'A', check_remote=False)
    assert status.value == 'STALE'
    assert path.endswith(plan_file.name)
    assert catalog.list_plans('dx3', 'A')[0]['stale'] is True
    # Plans are never deleted by a CAD change.
    assert plan_file.exists()


def test_parts_missing_remotely_are_archived(catalog, fake_client):
    _add(fake_client, make_item('A'), make_item('B'))
    _sync(catalog)
    del fake_client.items['B']
    result = _sync(catalog)
    assert result.archived == 1
    assert catalog.db.get_part('dx3', 'B')['sync_status'] == 'archived'
    assert [p['part_id'] for p in catalog.db.list_parts()] == ['A']


def test_paging_continues_until_a_short_page(catalog, fake_client, dx3_settings):
    dx3_settings.page_size = 2
    _add(fake_client, *[make_item(str(i)) for i in range(5)])
    assert _sync(catalog).added == 5
    assert [(top, skip) for _, top, skip in fake_client.search_calls] == [(2, 0), (2, 2), (2, 4)]


def test_paging_stops_on_an_empty_page(catalog, fake_client, dx3_settings):
    dx3_settings.page_size = 2
    _add(fake_client, *[make_item(str(i)) for i in range(4)])
    assert _sync(catalog).added == 4
    assert [skip for _, _, skip in fake_client.search_calls] == [0, 2, 4]


def test_truncated_scan_does_not_archive(catalog, fake_client, dx3_settings):
    _add(fake_client, *[make_item(str(i)) for i in range(5)])
    _sync(catalog)
    dx3_settings.max_items = 3
    dx3_settings.page_size = 2
    result = _sync(catalog)
    assert result.archived == 0
    assert any('truncated' in d for d in result.error_details)
    assert all(p['sync_status'] != 'archived'
               for p in catalog.db.get_all_parts('dx3').values())


def test_bookmark_scope_keeps_only_engineering_items(catalog, fake_client, dx3_settings):
    dx3_settings.bookmark_scope = 'Inspection Parts'
    dx3_settings.collab_space_filter = 'Test Space'
    _add(fake_client, make_item('A', space='Other Space'), make_item('B'))
    fake_client.bookmarks['Inspection Parts'] = {'id': 'BK1', 'title': 'Inspection Parts'}
    fake_client.bookmark_members['BK1'] = [eng_member('A'), doc_member('D1')]

    result = _sync(catalog)

    assert result.added == 1
    # The bookmark is the scope: no search, and no space filter on its members.
    assert fake_client.search_calls == []
    assert set(catalog.db.get_all_parts('dx3')) == {'A'}
    assert catalog.backend('dx3').describe()['bookmark'] == 'Inspection Parts'


def test_space_filter_is_pushed_to_the_server(catalog, fake_client, dx3_settings):
    dx3_settings.bookmark_scope = 'INSP'
    dx3_settings.collab_space_filter = 'Test Space'
    _add(fake_client, make_item('A'))
    _sync(catalog)
    assert fake_client.search_calls[0][0] == '[ds6w:project]:"Test Space" AND INSP'


def test_space_predicate_falls_back_to_client_side_filtering(catalog, fake_client, dx3_settings):
    dx3_settings.bookmark_scope = 'INSP'
    dx3_settings.collab_space_filter = 'Test Space'
    fake_client.space_predicate_unsupported = True
    _add(fake_client, make_item('A'), make_item('B', space='Other Space'))
    result = _sync(catalog)
    assert [q for q, _, _ in fake_client.search_calls] == [
        '[ds6w:project]:"Test Space" AND INSP', 'INSP']
    assert result.added == 1
    assert set(catalog.db.get_all_parts('dx3')) == {'A'}


def test_maturity_filter(catalog, fake_client, core_config):
    core_config.maturity_filter = 'released'
    _add(fake_client, make_item('A', maturity='RELEASED'), make_item('B'))
    _sync(catalog)
    assert set(catalog.db.get_all_parts('dx3')) == {'A'}


def test_incremental_escalates_to_full_when_its_page_fills(catalog, fake_client, dx3_settings):
    dx3_settings.page_size = 2
    _add(fake_client, *[make_item(str(i)) for i in range(3)])
    [result] = catalog.run_incremental_sync()
    assert result.added == 3


def test_incremental_on_a_short_page_never_archives(catalog, fake_client):
    _add(fake_client, make_item('A'), make_item('B'))
    _sync(catalog)
    del fake_client.items['B']
    [result] = catalog.run_incremental_sync()
    assert (result.unchanged, result.archived) == (1, 0)


def test_sync_part(catalog, fake_client):
    _add(fake_client, make_item('A'))
    assert catalog.sync_part('dx3', 'A')[0]
    assert catalog.db.get_part('dx3', 'A') is not None
    del fake_client.items['A']
    catalog.sync_part('dx3', 'A')
    assert catalog.db.get_part('dx3', 'A')['sync_status'] == 'archived'


def test_fetch_geometry_downloads_once_then_uses_cache(catalog, fake_client, catalog_paths):
    _add(fake_client, make_item('A'))
    fake_client.step_files['A'] = b'ISO-10303-21; dummy'
    _sync(catalog)
    assert catalog.db.get_part('dx3', 'A')['geometry_available'] == 1
    path, message = catalog.fetch_geometry('dx3', 'A')
    assert path == catalog_paths.geometry_path(storage_key('dx3', 'A'), 'c1')
    assert 'Downloaded' in message
    del fake_client.step_files['A']
    path2, message2 = catalog.fetch_geometry('dx3', 'A')
    assert path2 == path and message2 == 'STEP already cached.'


def test_fetch_geometry_adopts_a_hand_placed_file(catalog, fake_client, catalog_paths):
    _add(fake_client, make_item('A'), make_item('B'))
    _sync(catalog)
    # The legacy location operators have always used...
    legacy = catalog_paths.legacy_step_path('A', 'A.1')
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b'ISO-10303-21; hand placed')
    path, message = catalog.fetch_geometry('dx3', 'A')
    assert path == legacy and 'locally provided' in message
    # ...and the part's own geometry directory.
    directory = catalog_paths.geometry_dir(storage_key('dx3', 'B'))
    directory.mkdir(parents=True)
    (directory / 'export.step').write_bytes(b'ISO-10303-21; hand placed')
    path, _ = catalog.fetch_geometry('dx3', 'B')
    assert path == directory / 'export.step'


def test_no_geometry_names_the_place_to_put_one(catalog, fake_client, catalog_paths):
    _add(fake_client, make_item('A'))
    _sync(catalog)
    path, message = catalog.fetch_geometry('dx3', 'A')
    assert path is None
    assert str(catalog_paths.geometry_dir(storage_key('dx3', 'A'))) in message


def test_resolved_paths_survive_a_different_mount_point(catalog, fake_client, catalog_paths):
    _add(fake_client, make_item('A'))
    fake_client.step_files['A'] = b'ISO-10303-21; dummy'
    _sync(catalog)
    catalog.fetch_geometry('dx3', 'A')
    # Simulate a row written from the host mount: the stored path is foreign here.
    catalog.db.set_geometry_cache('dx3', 'A', '/host/only/catalog/x.stp', 'c1')
    part = catalog.db.get_part('dx3', 'A')
    assert catalog.resolved_geometry_path(part) == catalog_paths.geometry_path(
        storage_key('dx3', 'A'), 'c1')


def test_unconfigured_backend_fails_the_sync_but_not_the_catalog(catalog, fake_client):
    _add(fake_client, make_item('A'))
    fake_client.config.password = ''
    result = _sync(catalog)
    assert not result.success and 'DX_PASSWORD' in result.message
    assert catalog.db.list_parts() == []
