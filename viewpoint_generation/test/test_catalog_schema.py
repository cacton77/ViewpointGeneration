"""Characterization test: an old catalog.db gains newer columns in place."""

import json
import sqlite3

from viewpoint_generation.catalog.schema import CatalogDB

# The schema as the first catalog release created it, before _migrate() added
# thumbnails, STEP caching, plan stages, and run bookkeeping.
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


def _columns(db_path, table):
    with sqlite3.connect(db_path) as conn:
        return {row[1] for row in conn.execute(f'PRAGMA table_info({table})')}


def test_old_database_migrates_without_losing_rows(tmp_path):
    db_path = tmp_path / 'catalog.db'
    with sqlite3.connect(db_path) as conn:
        conn.executescript(OLD_SCHEMA)
        conn.execute("INSERT INTO parts VALUES ('A', 'Part A', 'prd-A', 'A.1', 'c1',"
                     " 'IN_WORK', 'VPMReference', 'Space', 'synced', 't0', 't0')")
        conn.execute("INSERT INTO plans VALUES ('A:p1', 'A', NULL, 'c1',"
                     " '/catalog/plans/A/p1.json', 3, 9, 9, 'region_growth', NULL,"
                     " '2026-08-01T00:00:00Z', NULL, 'local')")
        conn.execute("INSERT INTO sync_log (timestamp, action) VALUES ('t0', 'full_sync')")

    db = CatalogDB(db_path)

    assert {'thumbnail_path', 'step_path', 'step_cestamp',
            'step_available'} <= _columns(db_path, 'parts')
    assert {'stage', 'source', 'is_current'} <= _columns(db_path, 'plans')
    assert {'plan_id', 'result_bundle_path',
            'upload_status'} <= _columns(db_path, 'inspection_runs')
    assert db.get_part('A')['title'] == 'Part A'
    plan = db.get_current_plan('A')
    assert plan['plan_id'] == 'A:p1'
    assert plan['stage'] is None
    # Schema step 1: the legacy counts move into the plan-contract fields.
    assert json.loads(plan['summary_json'])['num_regions'] == 3
    assert plan['summary_text'] == '3 regions, 9 clusters, 9 viewpoints'
    assert plan['stage_rank'] == -1
    assert len(db.recent_sync_log()) == 1

    # Opening again is a no-op.
    CatalogDB(db_path)
    assert db.get_plan('A:p1')['file_path'] == '/catalog/plans/A/p1.json'


def test_versioned_migration_backs_up_once_and_is_idempotent(tmp_path):
    db_path = tmp_path / 'catalog.db'
    with sqlite3.connect(db_path) as conn:
        conn.executescript(OLD_SCHEMA)
        conn.execute("INSERT INTO parts VALUES ('A', 'Part A', 'prd-A', 'A.1', 'c1',"
                     " 'IN_WORK', 'VPMReference', 'Space', 'synced', 't0', 't0')")
        conn.execute("INSERT INTO plans VALUES ('A:o', 'A', NULL, 'c1', '/x/o.json',"
                     " 2, 4, 4, NULL, NULL, 't1', NULL, 'local')")
        conn.execute("UPDATE plans SET upload_status = 'local'")

    CatalogDB(db_path)
    backups = sorted(tmp_path.glob('catalog.db.bak-*'))
    assert len(backups) == 1
    with sqlite3.connect(backups[0]) as conn:
        assert conn.execute('PRAGMA user_version').fetchone()[0] == 0
        assert conn.execute('SELECT COUNT(*) FROM plans').fetchone()[0] == 1
    with sqlite3.connect(db_path) as conn:
        assert conn.execute('PRAGMA user_version').fetchone()[0] >= 1

    CatalogDB(db_path)
    assert len(list(tmp_path.glob('catalog.db.bak-*'))) == 1


def test_fresh_database_needs_no_backup(tmp_path):
    db = CatalogDB(tmp_path / 'catalog.db')
    assert list(tmp_path.glob('catalog.db.bak-*')) == []
    assert db.list_plans('nothing') == []
