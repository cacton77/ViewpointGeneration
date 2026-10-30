"""Migration: old catalog.db files upgrade in place, losing nothing."""

import json
import sqlite3

from viewpoint_generation.catalog.core.paths import CatalogPaths, storage_key
from viewpoint_generation.catalog.core.schema import SCHEMA_VERSION, CatalogDB

# The schema as the first catalog release created it, before columns were
# added for thumbnails, STEP caching, plan stages, and run bookkeeping.
OLD_SCHEMA = """
CREATE TABLE parts (
    eng_item_id TEXT PRIMARY KEY, title TEXT NOT NULL, part_number TEXT,
    revision TEXT, cestamp TEXT NOT NULL, maturity TEXT, type TEXT,
    collab_space TEXT, sync_status TEXT DEFAULT 'new',
    first_seen TEXT NOT NULL, last_synced TEXT NOT NULL);
CREATE TABLE plans (
    plan_id TEXT PRIMARY KEY, eng_item_id TEXT NOT NULL, plan_doc_id TEXT,
    cestamp TEXT NOT NULL, file_path TEXT NOT NULL, num_regions INTEGER,
    num_clusters INTEGER, num_viewpoints INTEGER, seg_algorithm TEXT,
    traversal_algo TEXT, generated_at TEXT NOT NULL, uploaded_at TEXT,
    upload_status TEXT DEFAULT 'local');
CREATE TABLE inspection_runs (
    run_id TEXT PRIMARY KEY, eng_item_id TEXT NOT NULL, started_at TEXT NOT NULL,
    completed_at TEXT, viewpoints_executed INTEGER, anomalies_found INTEGER,
    result_doc_id TEXT, status TEXT DEFAULT 'running');
CREATE TABLE sync_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
    action TEXT NOT NULL, eng_item_id TEXT, detail TEXT);
"""


def _old_database(db_path):
    with sqlite3.connect(db_path) as conn:
        conn.executescript(OLD_SCHEMA)
        conn.execute("INSERT INTO parts VALUES ('A', 'Part A', 'prd-A', 'A.1', 'c1',"
                     " 'IN_WORK', 'VPMReference', 'Space', 'synced', 't0', 't0')")
        conn.execute("INSERT INTO plans VALUES ('A:p1', 'A', 'DOC9', 'c1',"
                     " '/catalog/plans/A/p1.json', 3, 9, 9, 'region_growth', NULL,"
                     " '2026-08-01T00:00:00Z', NULL, 'synced')")
        conn.execute("INSERT INTO plans VALUES ('A:p0', 'A', NULL, 'c0',"
                     " '/catalog/plans/A/p0.json', 2, 0, 0, NULL, NULL,"
                     " '2026-07-01T00:00:00Z', NULL, 'local')")
        conn.execute("INSERT INTO inspection_runs VALUES ('run_1', 'A', 't2', NULL, 5, 0,"
                     " 'RES9', 'completed')")
        conn.execute("INSERT INTO sync_log (timestamp, action, eng_item_id, detail)"
                     " VALUES ('t0', 'part_added', 'A', 'Part A')")
        conn.execute("INSERT INTO sync_log (timestamp, action) VALUES ('t1', 'full_sync')")


def test_old_database_migrates_without_losing_rows(tmp_path):
    db_path = tmp_path / 'catalog.db'
    _old_database(db_path)

    db = CatalogDB(db_path)

    with sqlite3.connect(db_path) as conn:
        assert conn.execute('PRAGMA user_version').fetchone()[0] == SCHEMA_VERSION == 2
    part = db.get_part('dx3', 'A')
    assert (part['title'], part['change_token'], part['container']) == ('Part A', 'c1', 'Space')
    assert part['storage_key'] == storage_key('dx3', 'A')
    assert json.loads(part['extra_json']) == {'type': 'VPMReference'}

    plans = {p['plan_id']: p for p in db.list_plans('dx3', 'A')}
    assert set(plans) == {'A:p1', 'A:p0'}
    assert plans['A:p1']['remote_ref'] == 'DOC9'
    assert plans['A:p1']['change_token'] == 'c1'
    assert plans['A:p0']['change_token'] == 'c0'
    # Step 1 moved the legacy counts into the plan-contract fields.
    assert json.loads(plans['A:p1']['summary_json'])['num_regions'] == 3
    assert plans['A:p1']['summary_text'] == '3 regions, 9 clusters, 9 viewpoints'
    assert plans['A:p0']['summary_text'] == '2 regions'
    assert db.get_current_plan('dx3', 'A')['plan_id'] == 'A:p1'

    run = db.get_run('run_1')
    assert (run['backend'], run['part_id'], run['remote_ref']) == ('dx3', 'A', 'RES9')
    log = db.recent_sync_log()
    assert [(e['action'], e['backend'], e['part_id']) for e in log] == [
        ('full_sync', None, None), ('part_added', 'dx3', 'A')]


def test_versioned_migration_backs_up_once_and_is_idempotent(tmp_path):
    db_path = tmp_path / 'catalog.db'
    _old_database(db_path)

    CatalogDB(db_path)
    backups = sorted(tmp_path.glob('catalog.db.bak-*'))
    assert len(backups) == 1
    with sqlite3.connect(backups[0]) as conn:
        assert conn.execute('PRAGMA user_version').fetchone()[0] == 0
        assert conn.execute('SELECT COUNT(*) FROM plans').fetchone()[0] == 2

    CatalogDB(db_path)
    assert len(list(tmp_path.glob('catalog.db.bak-*'))) == 1


def test_a_phase_1_database_takes_only_step_2(tmp_path):
    db_path = tmp_path / 'catalog.db'
    _old_database(db_path)
    with sqlite3.connect(db_path) as conn:
        for name, definition in (('stage', 'TEXT'), ('stage_rank', 'INTEGER DEFAULT -1'),
                                 ('summary_json', 'TEXT'), ('summary_text', 'TEXT'),
                                 ('is_current', 'INTEGER DEFAULT 1'),
                                 ('source', "TEXT DEFAULT 'local'")):
            conn.execute(f'ALTER TABLE plans ADD COLUMN {name} {definition}')
        conn.execute("UPDATE plans SET summary_json = '{\"k\": 1}', summary_text = 'kept'")
        conn.execute('PRAGMA user_version = 1')
    db = CatalogDB(db_path)
    assert {p['summary_text'] for p in db.list_plans('dx3', 'A')} == {'kept'}


def test_fresh_database_needs_no_backup(tmp_path):
    db = CatalogDB(tmp_path / 'catalog.db')
    assert list(tmp_path.glob('catalog.db.bak-*')) == []
    assert db.list_plans('dx3', 'nothing') == []


def test_legacy_cache_files_still_resolve(tmp_path):
    """Migration moves no files; the resolver finds the old layout."""
    paths = CatalogPaths.from_env(tmp_path)
    legacy_step = paths.legacy_step_path('A', 'A.1')
    legacy_step.parent.mkdir(parents=True)
    legacy_step.write_text('ISO-10303-21;')
    legacy_plan = paths.legacy_plans_for('A') / 'p1.json'
    legacy_plan.parent.mkdir(parents=True)
    legacy_plan.write_text('{}')
    assert paths.resolve_cached('/elsewhere/A.1.stp', None, legacy_step) == legacy_step
    # A part_id that is not a safe file name never becomes a legacy path.
    assert paths.legacy_step_path('doc/w/e/part', 'A') is None
    assert paths.legacy_plans_for('..') is None
