"""Characterization tests for plan recording, history, staleness and upload policy."""

import json
import os
import time

from catalog_fakes import make_item, results_for

from viewpoint_generation import plan_io


def _sync(catalog, client, *items):
    for item in items:
        client.items[item['eng_item_id']] = item
    catalog.run_full_sync()


def test_every_stage_is_recorded_and_the_latest_is_current(
        catalog, fake_client, write_plan_file):
    _sync(catalog, fake_client, make_item('A'))
    plan_dir = catalog.plan_dir_for('A')
    recorded = []
    for stage in plan_io.STAGES:
        plan_id, message = catalog.record_plan('A', write_plan_file(plan_dir, stage),
                                               upload=False)
        assert plan_id, message
        recorded.append(plan_id)

    plans = catalog.list_plans('A')
    assert len(plans) == 4
    assert {p['stage'] for p in plans} == set(plan_io.STAGES)
    assert [p['plan_id'] for p in plans if p['is_current']] == [recorded[-1]]
    assert catalog.db.get_current_plan('A')['stage'] == 'ordered'
    row = catalog.db.get_plan(recorded[2])
    assert (row['num_regions'], row['num_clusters'], row['num_viewpoints']) == (2, 4, 4)
    assert row['cestamp'] == 'c1'


def test_record_plan_stamps_identity_into_the_file(catalog, fake_client, write_plan_file):
    _sync(catalog, fake_client, make_item('A'))
    path = write_plan_file(catalog.plan_dir_for('A'), 'segmented')
    catalog.record_plan('A', path, upload=False)
    data = json.loads(path.read_text())
    context = data['plm_context']
    assert context['eng_item_id'] == 'A'
    assert context['cestamp'] == 'c1'
    assert context['stage'] == 'segmented'
    assert context['cell_id'] == 'alpha'
    assert context['generated_by'] == 'tester'
    assert context['software_version'].startswith('ViewpointGeneration@')
    assert data['schema_version'] == '2.0'
    assert 'meshes' in data


def test_auto_upload_policy_uploads_only_configured_stages(
        catalog, fake_client, write_plan_file):
    _sync(catalog, fake_client, make_item('A'))
    plan_dir = catalog.plan_dir_for('A')
    seg_id, _ = catalog.record_plan('A', write_plan_file(plan_dir, 'segmented'))
    assert fake_client.uploads == []
    assert catalog.db.get_plan(seg_id)['upload_status'] == 'local'

    ordered_id, message = catalog.record_plan('A', write_plan_file(plan_dir, 'ordered'))
    assert len(fake_client.uploads) == 1
    assert fake_client.uploads[0]['title'] == 'PLAN_prd-A_A.1_ORDERED'
    assert fake_client.uploads[0]['collab_space'] == 'Test Space'
    row = catalog.db.get_plan(ordered_id)
    assert row['upload_status'] == 'synced'
    assert row['plan_doc_id'] == 'DOC1'
    assert 'DOC1' in message


def test_upload_policy_empty_means_manual_only(catalog, fake_client, sync_config,
                                               write_plan_file):
    sync_config.auto_upload_plan_stages = ''
    _sync(catalog, fake_client, make_item('A'))
    plan_id, _ = catalog.record_plan('A', write_plan_file(catalog.plan_dir_for('A'), 'ordered'))
    assert fake_client.uploads == []
    doc_id, _ = catalog.upload_recorded_plan(plan_id)
    assert doc_id == 'DOC1'


def test_force_upload_overrides_policy(catalog, fake_client, write_plan_file):
    _sync(catalog, fake_client, make_item('A'))
    catalog.record_plan('A', write_plan_file(catalog.plan_dir_for('A'), 'clustered'),
                        upload=True)
    assert [u['title'] for u in fake_client.uploads] == ['PLAN_prd-A_A.1_CLUSTERED']


def test_upload_without_credentials_stays_local(catalog, fake_client, write_plan_file):
    _sync(catalog, fake_client, make_item('A'))
    fake_client.config.password = ''
    plan_id, _ = catalog.record_plan('A', write_plan_file(catalog.plan_dir_for('A'), 'ordered'))
    assert fake_client.uploads == []
    assert catalog.db.get_plan(plan_id)['upload_status'] == 'local'


def test_staleness_and_explicit_selection(catalog, fake_client, write_plan_file):
    _sync(catalog, fake_client, make_item('A'))
    old_id, _ = catalog.record_plan(
        'A', write_plan_file(catalog.plan_dir_for('A'), 'ordered'), upload=False)

    fake_client.items['A']['cestamp'] = 'c2'
    catalog.run_full_sync()
    assert catalog.ensure_plan('A', check_remote=True)[0].value == 'STALE'

    new_id, _ = catalog.record_plan(
        'A', write_plan_file(catalog.plan_dir_for('A'), 'segmented'), upload=False)
    assert catalog.ensure_plan('A', check_remote=False)[0].value == 'CURRENT'
    stale = {p['plan_id']: p['stale'] for p in catalog.list_plans('A')}
    assert stale == {old_id: True, new_id: False}

    status, path = catalog.select_plan('A', old_id)
    assert status.value == 'SELECTED'
    assert catalog.db.get_current_plan('A')['plan_id'] == old_id
    status, _ = catalog.select_plan('A', new_id)
    assert status.value == 'CURRENT'
    assert catalog.select_plan('A', 'nope')[0].value == 'NONE'


def test_no_plan_is_none(catalog, fake_client):
    _sync(catalog, fake_client, make_item('A'))
    status, path = catalog.ensure_plan('A')
    assert (status.value, path) == ('NONE', '')
    assert catalog.ensure_plan('missing')[0].value == 'NONE'


def test_adoption_orders_by_generation_time_then_stage(catalog, fake_client, catalog_paths):
    _sync(catalog, fake_client, make_item('A'))
    plan_dir = catalog.plan_dir_for('A')
    plan_dir.mkdir(parents=True)
    # Bare results files, as written before plans carried any context. Recorded
    # alphabetically (clustered, ordered, segmented), so the last one recorded
    # is NOT the newest; adoption must hand the current flag back.
    stamp = time.time() - 3600
    for name, stage in (('clustered_x.json', 'clustered'), ('ordered_x.json', 'ordered'),
                        ('segmented_x.json', 'segmented')):
        path = plan_dir / name
        path.write_text(json.dumps(results_for(stage)))
        os.utime(path, (stamp, stamp))
    legacy_dir = catalog_paths.step_dir / 'A' / 'A.1mm_results'
    legacy_dir.mkdir(parents=True)
    legacy = legacy_dir / 'part_12_regions.json'
    legacy.write_text(json.dumps(results_for('segmented')))
    os.utime(legacy, (stamp - 60, stamp - 60))
    (legacy_dir / 'empty.json').write_text(json.dumps({'meshes': [{'regions': []}]}))

    adopted, skipped = catalog.adopt_orphan_plans('A')

    assert (adopted, skipped) == (4, 1)
    assert catalog.db.get_current_plan('A')['stage'] == 'ordered'
    # Adopting again finds nothing new.
    assert catalog.adopt_orphan_plans('A') == (0, 1)


def test_adoption_backfills_missing_stage(catalog, fake_client, write_plan_file):
    _sync(catalog, fake_client, make_item('A'))
    plan_id, _ = catalog.record_plan(
        'A', write_plan_file(catalog.plan_dir_for('A'), 'projected'), upload=False)
    catalog.db.update_plan(plan_id, stage=None)
    catalog.adopt_orphan_plans('A')
    assert catalog.db.get_plan(plan_id)['stage'] == 'projected'


def test_inspection_run_result_upload_raises_issue_above_threshold(catalog, fake_client):
    _sync(catalog, fake_client, make_item('A'))
    run_id, _ = catalog.start_run('A', run_id='run_1')
    bundle = catalog.paths.result_dir_for('A', run_id)
    bundle.mkdir(parents=True)
    (bundle / 'image_0.png').write_bytes(b'png')
    manifest, _ = catalog.build_result_manifest(
        run_id, anomalies=[{'score': 0.95, 'label': 'scratch'}])
    data = json.loads(manifest.read_text())
    assert data['plm_context']['eng_item_id'] == 'A'
    assert data['summary']['overall_result'] == 'FAIL'
    assert data['files'] == [{'path': 'image_0.png', 'type': 'image', 'size_bytes': 3}]

    doc_id, message = catalog.upload_results(run_id)

    assert doc_id == 'RES1'
    assert fake_client.result_uploads[0]['title'] == 'RESULT_prd-A_A.1_run_1'
    assert len(fake_client.issues) == 1
    assert 'ISS1' in message
    assert catalog.db.get_run(run_id)['upload_status'] == 'uploaded'


def test_result_upload_below_threshold_raises_no_issue(catalog, fake_client):
    _sync(catalog, fake_client, make_item('A'))
    run_id, _ = catalog.start_run('A', run_id='run_2')
    catalog.build_result_manifest(run_id, anomalies=[{'score': 0.1}])
    catalog.upload_results(run_id)
    assert fake_client.issues == []
