"""Characterization tests for CatalogSync reconciliation (no network)."""

from catalog_fakes import doc_member, eng_member, make_item


def _add(client, *items):
    for item in items:
        client.items[item['eng_item_id']] = item


def test_full_sync_adds_new_parts(catalog, fake_client):
    _add(fake_client, make_item('A'), make_item('B'))
    result = catalog.run_full_sync()
    assert result.success
    assert (result.added, result.modified, result.unchanged, result.archived) == (2, 0, 0, 0)
    parts = catalog.db.get_all_parts()
    assert set(parts) == {'A', 'B'}
    assert parts['A']['sync_status'] == 'new'
    assert parts['A']['cestamp'] == 'c1'
    assert parts['A']['collab_space'] == 'Test Space'


def test_second_sync_reports_everything_unchanged(catalog, fake_client):
    _add(fake_client, make_item('A'), make_item('B'))
    catalog.run_full_sync()
    result = catalog.run_full_sync()
    assert (result.added, result.modified, result.unchanged) == (0, 0, 2)
    assert {p['sync_status'] for p in catalog.db.get_all_parts().values()} == {'synced'}


def test_modified_part_drops_cached_step_and_marks_plans_stale(
        catalog, fake_client, write_plan_file, record):
    _add(fake_client, make_item('A'))
    fake_client.step_files['A'] = b'ISO-10303-21; dummy'
    catalog.run_full_sync()
    step_path, _ = catalog.fetch_step('A')
    assert step_path is not None and step_path.exists()
    plan_file = write_plan_file(catalog.plan_dir_for('A'), 'ordered')
    record(catalog, 'A', plan_file, upload=False)
    assert catalog.ensure_plan('A', check_remote=False)[0].value == 'CURRENT'

    fake_client.items['A']['cestamp'] = 'c2'
    result = catalog.run_full_sync()

    assert result.modified == 1
    part = catalog.db.get_part('A')
    assert part['sync_status'] == 'modified'
    assert part['cestamp'] == 'c2'
    assert part['step_path'] is None
    assert not step_path.exists()
    status, path = catalog.ensure_plan('A', check_remote=False)
    assert status.value == 'STALE'
    assert path.endswith(plan_file.name)
    assert catalog.list_plans('A')[0]['stale'] is True
    # Plans are never deleted by a CAD change.
    assert plan_file.exists()


def test_parts_missing_remotely_are_archived(catalog, fake_client):
    _add(fake_client, make_item('A'), make_item('B'))
    catalog.run_full_sync()
    del fake_client.items['B']
    result = catalog.run_full_sync()
    assert result.archived == 1
    assert catalog.db.get_part('B')['sync_status'] == 'archived'
    assert [p['eng_item_id'] for p in catalog.db.list_parts()] == ['A']


def test_paging_continues_until_a_short_page(catalog, fake_client, sync_config):
    sync_config.page_size = 2
    _add(fake_client, *[make_item(str(i)) for i in range(5)])
    result = catalog.run_full_sync()
    assert result.added == 5
    assert [(top, skip) for _, top, skip in fake_client.search_calls] == [(2, 0), (2, 2), (2, 4)]


def test_paging_stops_on_an_empty_page(catalog, fake_client, sync_config):
    sync_config.page_size = 2
    _add(fake_client, *[make_item(str(i)) for i in range(4)])
    assert catalog.run_full_sync().added == 4
    assert [skip for _, _, skip in fake_client.search_calls] == [0, 2, 4]


def test_truncated_scan_does_not_archive(catalog, fake_client, sync_config):
    _add(fake_client, *[make_item(str(i)) for i in range(5)])
    catalog.run_full_sync()
    sync_config.max_items = 3
    sync_config.page_size = 2
    result = catalog.run_full_sync()
    assert result.archived == 0
    assert any('truncated' in d for d in result.error_details)
    assert all(p['sync_status'] != 'archived' for p in catalog.db.get_all_parts().values())


def test_bookmark_scope_keeps_only_engineering_items(catalog, fake_client, sync_config):
    sync_config.bookmark_scope = 'Inspection Parts'
    sync_config.collab_space_filter = 'Test Space'
    _add(fake_client, make_item('A', space='Other Space'), make_item('B'))
    fake_client.bookmarks['Inspection Parts'] = {'id': 'BK1', 'title': 'Inspection Parts'}
    fake_client.bookmark_members['BK1'] = [eng_member('A'), doc_member('D1')]

    result = catalog.run_full_sync()

    assert result.added == 1
    # The bookmark is the scope: no search, and no space filter on its members.
    assert fake_client.search_calls == []
    assert set(catalog.db.get_all_parts()) == {'A'}
    assert catalog.active_bookmark['id'] == 'BK1'


def test_space_filter_is_pushed_to_the_server(catalog, fake_client, sync_config):
    sync_config.bookmark_scope = 'INSP'
    sync_config.collab_space_filter = 'Test Space'
    _add(fake_client, make_item('A'))
    catalog.run_full_sync()
    assert fake_client.search_calls[0][0] == '[ds6w:project]:"Test Space" AND INSP'


def test_space_predicate_falls_back_to_client_side_filtering(catalog, fake_client, sync_config):
    sync_config.bookmark_scope = 'INSP'
    sync_config.collab_space_filter = 'Test Space'
    fake_client.space_predicate_unsupported = True
    _add(fake_client, make_item('A'), make_item('B', space='Other Space'))
    result = catalog.run_full_sync()
    assert [q for q, _, _ in fake_client.search_calls] == [
        '[ds6w:project]:"Test Space" AND INSP', 'INSP']
    assert result.added == 1
    assert set(catalog.db.get_all_parts()) == {'A'}


def test_maturity_filter(catalog, fake_client, sync_config):
    sync_config.maturity_filter = 'released'
    _add(fake_client, make_item('A', maturity='RELEASED'), make_item('B'))
    catalog.run_full_sync()
    assert set(catalog.db.get_all_parts()) == {'A'}


def test_incremental_escalates_to_full_when_its_page_fills(catalog, fake_client, sync_config):
    sync_config.page_size = 2
    _add(fake_client, *[make_item(str(i)) for i in range(3)])
    result = catalog.run_incremental_sync()
    assert result.added == 3


def test_sync_single_item(catalog, fake_client):
    _add(fake_client, make_item('A'))
    assert catalog.sync_single_item('A')[0]
    assert catalog.db.get_part('A') is not None
    del fake_client.items['A']
    catalog.sync_single_item('A')
    assert catalog.db.get_part('A')['sync_status'] == 'archived'


def test_fetch_step_downloads_once_then_uses_cache(catalog, fake_client, catalog_paths):
    _add(fake_client, make_item('A'))
    fake_client.step_files['A'] = b'ISO-10303-21; dummy'
    catalog.run_full_sync()
    assert catalog.db.get_part('A')['step_available'] == 1
    path, message = catalog.fetch_step('A')
    assert path == catalog_paths.step_path('A', 'A.1')
    assert 'Downloaded' in message
    del fake_client.step_files['A']
    path2, message2 = catalog.fetch_step('A')
    assert path2 == path and message2 == 'STEP already cached.'


def test_fetch_step_adopts_a_hand_placed_file(catalog, fake_client, catalog_paths):
    _add(fake_client, make_item('A'))
    catalog.run_full_sync()
    dest = catalog_paths.step_path('A', 'A.1')
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b'ISO-10303-21; hand placed')
    path, message = catalog.fetch_step('A')
    assert path == dest
    assert 'locally provided' in message


def test_resolved_paths_survive_a_different_mount_point(catalog, fake_client, catalog_paths):
    _add(fake_client, make_item('A'))
    fake_client.step_files['A'] = b'ISO-10303-21; dummy'
    catalog.run_full_sync()
    catalog.fetch_step('A')
    # Simulate a row written from the host mount: the stored path is foreign here.
    catalog.db.set_step_cache('A', '/host/only/catalog/steps/A/A.1.stp', 'c1')
    part = catalog.db.get_part('A')
    assert catalog.resolved_step_path(part) == catalog_paths.step_path('A', 'A.1')
