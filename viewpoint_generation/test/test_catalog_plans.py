"""Plan recording, history, staleness, upload policy, and runs (dx3 via FakeDXClient)."""

import json
import os
import time

from catalog_fakes import make_item, producer_describe, results_for

from viewpoint_generation import plan_io
from viewpoint_generation.catalog.core.paths import storage_key


def _sync(catalog, client, *items):
    for item in items:
        client.items[item['eng_item_id']] = item
    catalog.run_full_sync()


def test_every_stage_is_recorded_and_the_latest_is_current(
        catalog, fake_client, write_plan_file, record, plan_dir):
    _sync(catalog, fake_client, make_item('A'))
    directory = plan_dir(catalog, 'A')
    assert directory == catalog.paths.plans_for(storage_key('dx3', 'A'))
    recorded = []
    for stage in plan_io.STAGES:
        plan_id, message = record(catalog, 'A', write_plan_file(directory, stage), upload=False)
        assert plan_id, message
        recorded.append(plan_id)

    plans = catalog.list_plans('dx3', 'A')
    assert len(plans) == 4
    assert {p['stage'] for p in plans} == set(plan_io.STAGES)
    assert [p['plan_id'] for p in plans if p['is_current']] == [recorded[-1]]
    assert catalog.db.get_current_plan('dx3', 'A')['stage'] == 'ordered'
    row = catalog.db.get_plan(recorded[2])
    assert row['summary_text'] == '2 regions, 4 clusters, 4 viewpoints'
    assert json.loads(row['summary_json'])['num_viewpoints'] == 4
    assert row['stage_rank'] == 2
    assert row['change_token'] == 'c1'


def test_plan_carries_the_catalog_context_plus_producer_keys(
        catalog, fake_client, write_plan_file, record):
    fake_client.step_files['A'] = b'ISO-10303-21; dummy'
    _sync(catalog, fake_client, make_item('A'))
    selection, message = catalog.select('dx3', 'A')
    assert selection is not None, message
    context = json.loads(selection['plm_context_json'])
    assert (context['backend'], context['part_id'], context['change_token']) == ('dx3', 'A', 'c1')
    assert context['container'] == 'Test Space'
    assert context['cell_id'] == 'alpha'
    assert context['generated_by'] == 'tester'

    # The producer embeds that object and adds only its own keys.
    results = results_for('segmented')
    results['plm_context'] = context
    path, _ = plan_io.write_plan(selection['plan_dir'], results)
    before = open(path).read()
    plan_id, _ = record(catalog, 'A', path, upload=False)
    # Recording never rewrites the file.
    assert open(path).read() == before
    written = json.loads(before)
    assert written['schema_version'] == '3.0'
    assert set(written['plm_context']) - set(context) == {
        'stage', 'generated_at', 'software_version'}
    row = catalog.db.get_plan(plan_id)
    assert row['generated_at'] == written['plm_context']['generated_at']
    assert row['change_token'] == 'c1'


def test_catalog_records_what_the_producer_states_without_opening_the_file(
        catalog, fake_client, tmp_path):
    _sync(catalog, fake_client, make_item('A'))
    opaque = tmp_path / 'opaque.plan'
    opaque.write_bytes(b'\x00 not json at all')
    plan_id, message = catalog.record_plan(
        'dx3', 'A', opaque, stage='inspected', stage_rank=7, summary_json='{"k": 1}',
        summary_text='seven things', change_token='c0', generated_at='2026-01-01T00:00:00Z',
        upload=False)
    assert plan_id, message
    row = catalog.db.get_plan(plan_id)
    assert (row['stage'], row['stage_rank'], row['summary_json'], row['summary_text'],
            row['change_token'], row['generated_at']) == (
        'inspected', 7, '{"k": 1}', 'seven things', 'c0', '2026-01-01T00:00:00Z')
    assert catalog.ensure_plan('dx3', 'A', check_remote=False)[0].value == 'STALE'
    assert catalog.record_plan('dx3', 'A', opaque, stage='')[0] is None
    assert catalog.record_plan('dx3', 'A', tmp_path / 'missing.json', stage='x')[0] is None
    assert catalog.record_plan('dx3', 'nope', opaque, stage='x')[0] is None


def test_auto_upload_policy_uploads_only_configured_stages(
        catalog, fake_client, write_plan_file, record, plan_dir):
    _sync(catalog, fake_client, make_item('A'))
    directory = plan_dir(catalog, 'A')
    seg_id, _ = record(catalog, 'A', write_plan_file(directory, 'segmented'))
    assert fake_client.uploads == []
    assert catalog.db.get_plan(seg_id)['upload_status'] == 'local'

    ordered_id, message = record(catalog, 'A', write_plan_file(directory, 'ordered'))
    assert len(fake_client.uploads) == 1
    upload = fake_client.uploads[0]
    assert upload['title'] == 'PLAN_prd-A_A.1_ORDERED'
    assert upload['collab_space'] == 'Test Space'
    # Relating documents to items is opt-in (dx3.document_links).
    assert upload['link_to_item'] is False
    row = catalog.db.get_plan(ordered_id)
    assert row['upload_status'] == 'synced'
    assert row['remote_ref'] == 'DOC1'
    assert 'DOC1' in message


def test_title_prefix_marks_test_uploads(catalog, fake_client, core_config, write_plan_file,
                                         record, plan_dir):
    core_config.title_prefix = 'TEST_PLM_BRIDGE_'
    _sync(catalog, fake_client, make_item('A'))
    record(catalog, 'A', write_plan_file(plan_dir(catalog, 'A'), 'ordered'))
    assert fake_client.uploads[0]['title'] == 'TEST_PLM_BRIDGE_PLAN_prd-A_A.1_ORDERED'


def test_upload_policy_empty_means_manual_only(
        catalog, fake_client, core_config, write_plan_file, record, plan_dir):
    core_config.auto_upload_plan_stages = ''
    _sync(catalog, fake_client, make_item('A'))
    plan_id, _ = record(catalog, 'A', write_plan_file(plan_dir(catalog, 'A'), 'ordered'))
    assert fake_client.uploads == []
    ref, _ = catalog.upload_recorded_plan(plan_id)
    assert ref == 'DOC1'


def test_force_upload_overrides_policy(catalog, fake_client, write_plan_file, record, plan_dir):
    _sync(catalog, fake_client, make_item('A'))
    record(catalog, 'A', write_plan_file(plan_dir(catalog, 'A'), 'clustered'), upload=True)
    assert [u['title'] for u in fake_client.uploads] == ['PLAN_prd-A_A.1_CLUSTERED']


def test_upload_without_credentials_stays_local(
        catalog, fake_client, write_plan_file, record, plan_dir):
    _sync(catalog, fake_client, make_item('A'))
    fake_client.config.password = ''
    plan_id, message = record(catalog, 'A', write_plan_file(plan_dir(catalog, 'A'), 'ordered'))
    assert fake_client.uploads == []
    assert catalog.db.get_plan(plan_id)['upload_status'] == 'local'
    assert 'Not uploaded' in message


def test_staleness_and_explicit_selection(
        catalog, fake_client, write_plan_file, record, plan_dir):
    _sync(catalog, fake_client, make_item('A'))
    old_id, _ = record(catalog, 'A', write_plan_file(plan_dir(catalog, 'A'), 'ordered'),
                       upload=False)

    fake_client.items['A']['cestamp'] = 'c2'
    catalog.run_full_sync()
    assert catalog.ensure_plan('dx3', 'A', check_remote=True)[0].value == 'STALE'

    new_id, _ = record(catalog, 'A', write_plan_file(plan_dir(catalog, 'A'), 'segmented'),
                       upload=False)
    assert catalog.ensure_plan('dx3', 'A', check_remote=False)[0].value == 'CURRENT'
    stale = {p['plan_id']: p['stale'] for p in catalog.list_plans('dx3', 'A')}
    assert stale == {old_id: True, new_id: False}

    status, path = catalog.select_plan('dx3', 'A', old_id)
    assert status.value == 'SELECTED'
    assert catalog.db.get_current_plan('dx3', 'A')['plan_id'] == old_id
    status, _ = catalog.select_plan('dx3', 'A', new_id)
    assert status.value == 'CURRENT'
    assert catalog.select_plan('dx3', 'A', 'nope')[0].value == 'NONE'


def test_no_plan_is_none(catalog, fake_client):
    _sync(catalog, fake_client, make_item('A'))
    status, path = catalog.ensure_plan('dx3', 'A')
    assert (status.value, path) == ('NONE', '')
    assert catalog.ensure_plan('dx3', 'missing')[0].value == 'NONE'


def test_adoption_orders_by_generation_time_then_stage(
        catalog, fake_client, catalog_paths, plan_dir):
    _sync(catalog, fake_client, make_item('A'))
    directory = plan_dir(catalog, 'A')
    directory.mkdir(parents=True)
    # Bare results files, as written before plans carried any context. Recorded
    # alphabetically (clustered, ordered, segmented), so the last one recorded
    # is NOT the newest; adoption must hand the current flag back.
    stamp = time.time() - 3600
    for name, stage in (('clustered_x.json', 'clustered'), ('ordered_x.json', 'ordered'),
                        ('segmented_x.json', 'segmented')):
        path = directory / name
        path.write_text(json.dumps(results_for(stage)))
        os.utime(path, (stamp, stamp))
    # Where early versions wrote, beside the legacy STEP cache.
    legacy_dir = catalog_paths.legacy_step_dir / 'A' / 'A.1mm_results'
    legacy_dir.mkdir(parents=True)
    legacy = legacy_dir / 'part_12_regions.json'
    legacy.write_text(json.dumps(results_for('segmented')))
    os.utime(legacy, (stamp - 60, stamp - 60))
    (legacy_dir / 'empty.json').write_text(json.dumps({'meshes': [{'regions': []}]}))

    adopted, skipped = catalog.adopt_orphan_plans('dx3', 'A', describe=producer_describe)

    assert (adopted, skipped) == (4, 1)
    assert catalog.db.get_current_plan('dx3', 'A')['stage'] == 'ordered'
    # Adopting again finds nothing new.
    assert catalog.adopt_orphan_plans('dx3', 'A', describe=producer_describe) == (0, 1)


def test_adoption_keeps_a_files_own_change_token(catalog, fake_client, plan_dir):
    _sync(catalog, fake_client, make_item('A'))
    directory = plan_dir(catalog, 'A')
    old = results_for('ordered')
    old['plm_context'] = {'eng_item_id': 'A', 'cestamp': 'c0'}   # a 2.0 file
    directory.mkdir(parents=True)
    (directory / 'old.json').write_text(json.dumps(old))
    catalog.adopt_orphan_plans('dx3', 'A', describe=producer_describe)
    assert catalog.list_plans('dx3', 'A')[0]['stale'] is True


def test_adoption_backfills_missing_stage(
        catalog, fake_client, write_plan_file, record, plan_dir):
    _sync(catalog, fake_client, make_item('A'))
    plan_id, _ = record(catalog, 'A', write_plan_file(plan_dir(catalog, 'A'), 'projected'),
                        upload=False)
    catalog.db.update_plan(plan_id, stage=None)
    catalog.adopt_orphan_plans('dx3', 'A', describe=producer_describe)
    assert catalog.db.get_plan(plan_id)['stage'] == 'projected'


def test_inspection_run_result_upload_raises_issue_above_threshold(catalog, fake_client):
    _sync(catalog, fake_client, make_item('A'))
    run_id, _ = catalog.start_run('dx3', 'A', run_id='run_1')
    bundle = catalog.paths.result_dir_for(storage_key('dx3', 'A'), run_id)
    bundle.mkdir(parents=True)
    (bundle / 'image_0.png').write_bytes(b'png')
    manifest, _ = catalog.build_result_manifest(
        run_id, anomalies=[{'score': 0.95, 'label': 'scratch'}])
    data = json.loads(manifest.read_text())
    assert (data['plm_context']['backend'], data['plm_context']['part_id']) == ('dx3', 'A')
    assert data['plm_context']['change_token'] == 'c1'
    assert data['summary']['overall_result'] == 'FAIL'
    assert data['files'] == [{'path': 'image_0.png', 'type': 'image', 'size_bytes': 3}]

    ref, message = catalog.upload_results(run_id)

    assert ref == 'RES1'
    assert fake_client.result_uploads[0]['title'] == 'RESULT_prd-A_A.1_run_1'
    assert len(fake_client.issues) == 1
    assert 'ISS1' in message
    run = catalog.db.get_run(run_id)
    assert (run['upload_status'], run['remote_ref'], run['issue_ref']) == (
        'uploaded', 'RES1', 'ISS1')


def test_result_upload_below_threshold_raises_no_issue(catalog, fake_client):
    _sync(catalog, fake_client, make_item('A'))
    run_id, _ = catalog.start_run('dx3', 'A', run_id='run_2')
    catalog.build_result_manifest(run_id, anomalies=[{'score': 0.1}])
    catalog.upload_results(run_id)
    assert fake_client.issues == []
