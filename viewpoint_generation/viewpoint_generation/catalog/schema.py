"""SQLite schema and access layer for the parts catalog.

One file-backed database (`catalog/catalog.db`) holds four tables:

    parts            -- one row per known 3DX engineering item
    plans            -- inspection plan envelopes generated for / downloaded
                        from those items
    inspection_runs  -- executed inspections and their upload state
    sync_log         -- audit trail of sync actions, for debugging

`init_db()` is idempotent: tables are created only when absent, and
`_migrate()` adds columns that older databases predate, so an existing
catalog.db survives an upgrade of this package without being rebuilt.

All access goes through `CatalogDB`, which opens a short-lived connection per
operation via a context manager and uses parameterized queries throughout.
"""

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


PARTS_TABLE = """
CREATE TABLE IF NOT EXISTS parts (
    eng_item_id     TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    part_number     TEXT,
    revision        TEXT,
    cestamp         TEXT NOT NULL,
    maturity        TEXT,
    type            TEXT,
    collab_space    TEXT,
    description     TEXT,
    thumbnail_url   TEXT,
    thumbnail_path  TEXT,
    step_available  INTEGER DEFAULT 0,
    step_path       TEXT,
    step_cestamp    TEXT,
    sync_status     TEXT DEFAULT 'new',
    first_seen      TEXT NOT NULL,
    last_synced     TEXT NOT NULL
)
"""

PLANS_TABLE = """
CREATE TABLE IF NOT EXISTS plans (
    plan_id         TEXT PRIMARY KEY,
    eng_item_id     TEXT NOT NULL REFERENCES parts(eng_item_id),
    plan_doc_id     TEXT,
    cestamp         TEXT NOT NULL,
    file_path       TEXT NOT NULL,
    num_regions     INTEGER,
    num_clusters    INTEGER,
    num_viewpoints  INTEGER,
    seg_algorithm   TEXT,
    traversal_algo  TEXT,
    generated_at    TEXT NOT NULL,
    uploaded_at     TEXT,
    upload_status   TEXT DEFAULT 'local',
    is_current      INTEGER DEFAULT 1,
    source          TEXT DEFAULT 'local',
    stage           TEXT
)
"""

INSPECTION_RUNS_TABLE = """
CREATE TABLE IF NOT EXISTS inspection_runs (
    run_id              TEXT PRIMARY KEY,
    eng_item_id         TEXT NOT NULL REFERENCES parts(eng_item_id),
    plan_id             TEXT REFERENCES plans(plan_id),
    started_at          TEXT NOT NULL,
    completed_at        TEXT,
    viewpoints_executed INTEGER,
    viewpoints_skipped  INTEGER DEFAULT 0,
    anomalies_found     INTEGER,
    max_anomaly_score   REAL,
    overall_result      TEXT,
    result_bundle_path  TEXT,
    result_doc_id       TEXT,
    upload_status       TEXT DEFAULT 'pending',
    status              TEXT DEFAULT 'running'
)
"""

SYNC_LOG_TABLE = """
CREATE TABLE IF NOT EXISTS sync_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    action          TEXT NOT NULL,
    eng_item_id     TEXT,
    detail          TEXT
)
"""

INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_plans_item ON plans(eng_item_id)",
    "CREATE INDEX IF NOT EXISTS idx_runs_item ON inspection_runs(eng_item_id)",
    "CREATE INDEX IF NOT EXISTS idx_sync_log_time ON sync_log(timestamp)",
)

TABLES = (PARTS_TABLE, PLANS_TABLE, INSPECTION_RUNS_TABLE, SYNC_LOG_TABLE)

# Columns added after the initial release, applied to existing databases by
# _migrate(). Keyed by table, each entry is (column_name, column_definition).
MIGRATIONS = {
    'parts': (
        ('description', 'TEXT'),
        ('thumbnail_url', 'TEXT'),
        ('thumbnail_path', 'TEXT'),
        ('step_available', 'INTEGER DEFAULT 0'),
        ('step_path', 'TEXT'),
        ('step_cestamp', 'TEXT'),
    ),
    'plans': (
        ('source', "TEXT DEFAULT 'local'"),
        ('is_current', 'INTEGER DEFAULT 1'),
        ('stage', 'TEXT'),
    ),
    'inspection_runs': (
        ('plan_id', 'TEXT'),
        ('viewpoints_skipped', 'INTEGER DEFAULT 0'),
        ('max_anomaly_score', 'REAL'),
        ('overall_result', 'TEXT'),
        ('result_bundle_path', 'TEXT'),
        ('upload_status', "TEXT DEFAULT 'pending'"),
    ),
}


def utc_now():
    """Current UTC time as an ISO-8601 string with a trailing Z."""
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _table_exists(conn, table):
    """True when the named table is already present in the database."""
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,)).fetchone()
    return row is not None


def _column_names(conn, table):
    """The set of column names on an existing table."""
    return {row[1] for row in conn.execute(f'PRAGMA table_info({table})')}


def _migrate(conn):
    """Add any columns missing from an older database."""
    for table, columns in MIGRATIONS.items():
        if not _table_exists(conn, table):
            continue
        existing = _column_names(conn, table)
        for name, definition in columns:
            if name not in existing:
                logger.info('Migrating %s: adding column %s', table, name)
                # Table and column names are module constants, never user input.
                conn.execute(f'ALTER TABLE {table} ADD COLUMN {name} {definition}')


def init_db(db_path):
    """Create the catalog database and its tables if they do not exist.

    Args:
        db_path: Path to the SQLite file. Parent directories are created.

    Returns:
        tuple: (bool, str) success flag and message.
    """
    db_path = Path(db_path)
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute('PRAGMA foreign_keys = ON')
            for statement in TABLES:
                conn.execute(statement)
            for statement in INDEXES:
                conn.execute(statement)
            _migrate(conn)
            conn.commit()
    except (sqlite3.Error, OSError) as e:
        return False, f'Could not initialize catalog database at {db_path}: {e}'
    return True, f'Catalog database ready at {db_path}.'


class CatalogDB:
    """Query and update helpers over the catalog SQLite database.

    Every method opens its own connection through a context manager, so
    instances are safe to share between the sync thread and ROS service
    callbacks without holding a long-lived cursor.
    """

    def __init__(self, db_path):
        """Open (creating if needed) the catalog database at db_path."""
        self.db_path = Path(db_path)
        success, message = init_db(self.db_path)
        if not success:
            raise RuntimeError(message)
        logger.debug(message)

    def _connect(self):
        """A new connection with row access by column name."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA foreign_keys = ON')
        return conn

    # --- parts ---------------------------------------------------------

    def get_all_parts(self):
        """Every catalogued part, keyed by engineering item id.

        Returns:
            dict: {eng_item_id: dict of column values}
        """
        with self._connect() as conn:
            rows = conn.execute('SELECT * FROM parts').fetchall()
        return {row['eng_item_id']: dict(row) for row in rows}

    def list_parts(self, search=None, maturity=None, collab_space=None,
                   include_archived=False):
        """Parts matching optional search/maturity/space filters.

        Args:
            search: Case-insensitive substring matched against title,
                part number, and description.
            maturity: Exact maturity state to keep (e.g. 'RELEASED').
            collab_space: Exact collaborative space to keep.
            include_archived: Include items no longer present remotely.

        Returns:
            list: dicts of column values, ordered by title.
        """
        clauses = []
        params = []
        if search:
            clauses.append('(LOWER(title) LIKE ? OR LOWER(IFNULL(part_number, "")) LIKE ?'
                           ' OR LOWER(IFNULL(description, "")) LIKE ?)')
            needle = f'%{search.lower()}%'
            params.extend([needle, needle, needle])
        if maturity:
            clauses.append('UPPER(IFNULL(maturity, "")) = ?')
            params.append(maturity.upper())
        if collab_space:
            clauses.append('collab_space = ?')
            params.append(collab_space)
        if not include_archived:
            clauses.append("sync_status != 'archived'")
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ''
        with self._connect() as conn:
            rows = conn.execute(
                f'SELECT * FROM parts{where} ORDER BY title COLLATE NOCASE',
                params).fetchall()
        return [dict(row) for row in rows]

    def get_part(self, eng_item_id):
        """One part by engineering item id, or None."""
        with self._connect() as conn:
            row = conn.execute('SELECT * FROM parts WHERE eng_item_id = ?',
                               (eng_item_id,)).fetchone()
        return dict(row) if row else None

    def insert_part(self, item):
        """Insert a newly discovered remote item with sync_status='new'.

        Args:
            item: dict of normalized item fields (see DXClient.search_eng_items).
        """
        now = utc_now()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO parts (eng_item_id, title, part_number, revision,
                                      cestamp, maturity, type, collab_space,
                                      description, thumbnail_url, sync_status,
                                      first_seen, last_synced)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'new', ?, ?)""",
                (item['eng_item_id'], item.get('title') or '', item.get('part_number'),
                 item.get('revision'), item.get('cestamp') or '', item.get('maturity'),
                 item.get('type'), item.get('collab_space'), item.get('description'),
                 item.get('thumbnail_url'), now, now))
            conn.commit()

    def update_part(self, item, sync_status='modified'):
        """Refresh a part's metadata after its cestamp changed."""
        with self._connect() as conn:
            conn.execute(
                """UPDATE parts
                      SET title = ?, part_number = ?, revision = ?, cestamp = ?,
                          maturity = ?, type = ?, collab_space = ?, description = ?,
                          thumbnail_url = ?, sync_status = ?, last_synced = ?
                    WHERE eng_item_id = ?""",
                (item.get('title') or '', item.get('part_number'), item.get('revision'),
                 item.get('cestamp') or '', item.get('maturity'), item.get('type'),
                 item.get('collab_space'), item.get('description'),
                 item.get('thumbnail_url'), sync_status, utc_now(),
                 item['eng_item_id']))
            conn.commit()

    def touch_part(self, eng_item_id, sync_status='synced'):
        """Mark an unchanged part as seen in this sync pass."""
        with self._connect() as conn:
            conn.execute(
                'UPDATE parts SET last_synced = ?, sync_status = ? WHERE eng_item_id = ?',
                (utc_now(), sync_status, eng_item_id))
            conn.commit()

    def archive_parts(self, eng_item_ids):
        """Mark local parts no longer visible remotely as archived."""
        if not eng_item_ids:
            return
        with self._connect() as conn:
            conn.executemany(
                "UPDATE parts SET sync_status = 'archived', last_synced = ?"
                ' WHERE eng_item_id = ?',
                [(utc_now(), item_id) for item_id in eng_item_ids])
            conn.commit()

    def set_thumbnail(self, eng_item_id, thumbnail_path):
        """Record the local cache path of a thumbnail."""
        with self._connect() as conn:
            conn.execute('UPDATE parts SET thumbnail_path = ? WHERE eng_item_id = ?',
                         (str(thumbnail_path) if thumbnail_path else None, eng_item_id))
            conn.commit()

    def set_thumbnail_url(self, eng_item_id, thumbnail_url):
        """Record the remote URL an item's image came from."""
        with self._connect() as conn:
            conn.execute('UPDATE parts SET thumbnail_url = ? WHERE eng_item_id = ?',
                         (thumbnail_url or None, eng_item_id))
            conn.commit()

    def set_step_available(self, eng_item_id, available):
        """Record whether 3DX reports a STEP derived output for this item."""
        with self._connect() as conn:
            conn.execute('UPDATE parts SET step_available = ? WHERE eng_item_id = ?',
                         (1 if available else 0, eng_item_id))
            conn.commit()

    def set_step_cache(self, eng_item_id, step_path, step_cestamp):
        """Record the local STEP file and the cestamp it was downloaded at."""
        with self._connect() as conn:
            conn.execute(
                'UPDATE parts SET step_path = ?, step_cestamp = ?, step_available = 1'
                ' WHERE eng_item_id = ?',
                (str(step_path) if step_path else None, step_cestamp, eng_item_id))
            conn.commit()

    def clear_step_cache(self, eng_item_id):
        """Forget a cached STEP file (used when the remote revision changed)."""
        with self._connect() as conn:
            conn.execute(
                'UPDATE parts SET step_path = NULL, step_cestamp = NULL'
                ' WHERE eng_item_id = ?', (eng_item_id,))
            conn.commit()

    def counts(self):
        """Summary counts for the picker UI status bar.

        Returns:
            dict: total, ready (STEP cached), and per-sync_status counts.
        """
        with self._connect() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM parts WHERE sync_status != 'archived'").fetchone()[0]
            ready = conn.execute(
                'SELECT COUNT(*) FROM parts WHERE step_path IS NOT NULL'
                " AND sync_status != 'archived'").fetchone()[0]
            by_status = {row[0]: row[1] for row in conn.execute(
                'SELECT sync_status, COUNT(*) FROM parts GROUP BY sync_status')}
        return {'total': total, 'ready': ready, 'by_status': by_status}

    # --- plans ---------------------------------------------------------

    def get_current_plan(self, eng_item_id):
        """The active plan for a part, or None."""
        with self._connect() as conn:
            row = conn.execute(
                'SELECT * FROM plans WHERE eng_item_id = ? AND is_current = 1'
                ' ORDER BY generated_at DESC LIMIT 1', (eng_item_id,)).fetchone()
        return dict(row) if row else None

    def get_plan(self, plan_id):
        """One plan by id, or None."""
        with self._connect() as conn:
            row = conn.execute('SELECT * FROM plans WHERE plan_id = ?',
                               (plan_id,)).fetchone()
        return dict(row) if row else None

    def list_plans(self, eng_item_id):
        """Every plan recorded for a part, newest first.

        Two pipeline stages can complete inside the same second, so rowid
        breaks ties on generated_at -- otherwise the order of a run's own
        stages is whatever SQLite happens to return.
        """
        with self._connect() as conn:
            rows = conn.execute(
                'SELECT * FROM plans WHERE eng_item_id = ?'
                ' ORDER BY generated_at DESC, rowid DESC',
                (eng_item_id,)).fetchall()
        return [dict(row) for row in rows]

    def upsert_plan(self, plan):
        """Insert or replace a plan row, making it the part's current plan.

        Args:
            plan: dict with at least plan_id, eng_item_id, cestamp, file_path.

        Returns:
            str: the plan_id written.
        """
        with self._connect() as conn:
            if plan.get('is_current', 1):
                conn.execute('UPDATE plans SET is_current = 0 WHERE eng_item_id = ?',
                             (plan['eng_item_id'],))
            conn.execute(
                """INSERT INTO plans (plan_id, eng_item_id, plan_doc_id, cestamp,
                                      file_path, num_regions, num_clusters,
                                      num_viewpoints, seg_algorithm, traversal_algo,
                                      generated_at, uploaded_at, upload_status,
                                      is_current, source, stage)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(plan_id) DO UPDATE SET
                       plan_doc_id = excluded.plan_doc_id,
                       cestamp = excluded.cestamp,
                       file_path = excluded.file_path,
                       num_regions = excluded.num_regions,
                       num_clusters = excluded.num_clusters,
                       num_viewpoints = excluded.num_viewpoints,
                       seg_algorithm = excluded.seg_algorithm,
                       traversal_algo = excluded.traversal_algo,
                       uploaded_at = excluded.uploaded_at,
                       upload_status = excluded.upload_status,
                       is_current = excluded.is_current,
                       source = excluded.source,
                       stage = excluded.stage""",
                (plan['plan_id'], plan['eng_item_id'], plan.get('plan_doc_id'),
                 plan.get('cestamp') or '', str(plan['file_path']),
                 plan.get('num_regions'), plan.get('num_clusters'),
                 plan.get('num_viewpoints'), plan.get('seg_algorithm'),
                 plan.get('traversal_algo'), plan.get('generated_at') or utc_now(),
                 plan.get('uploaded_at'), plan.get('upload_status', 'local'),
                 1 if plan.get('is_current', 1) else 0,
                 plan.get('source', 'local'), plan.get('stage')))
            conn.commit()
        return plan['plan_id']

    def set_current_plan(self, plan_id):
        """Make one plan the part's current plan, clearing the flag on its
        siblings. Used when an operator picks a specific plan in the picker.

        Returns:
            bool: False when no such plan exists.
        """
        with self._connect() as conn:
            row = conn.execute('SELECT eng_item_id FROM plans WHERE plan_id = ?',
                               (plan_id,)).fetchone()
            if row is None:
                return False
            conn.execute('UPDATE plans SET is_current = 0 WHERE eng_item_id = ?',
                         (row['eng_item_id'],))
            conn.execute('UPDATE plans SET is_current = 1 WHERE plan_id = ?',
                         (plan_id,))
            conn.commit()
        return True

    def delete_plan(self, plan_id):
        """Remove a plan row. The file on disk is left alone."""
        with self._connect() as conn:
            conn.execute('DELETE FROM plans WHERE plan_id = ?', (plan_id,))
            conn.commit()

    def update_plan(self, plan_id, **fields):
        """Update named columns on a plan row."""
        allowed = {'plan_doc_id', 'uploaded_at', 'upload_status', 'is_current',
                   'file_path', 'num_regions', 'num_clusters', 'num_viewpoints',
                   'seg_algorithm', 'traversal_algo', 'source', 'cestamp',
                   'stage'}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return
        assignments = ', '.join(f'{k} = ?' for k in updates)
        with self._connect() as conn:
            conn.execute(f'UPDATE plans SET {assignments} WHERE plan_id = ?',
                         [*updates.values(), plan_id])
            conn.commit()

    # --- inspection runs -----------------------------------------------

    def create_run(self, run):
        """Insert a new inspection run row.

        Args:
            run: dict with at least run_id and eng_item_id.

        Returns:
            str: the run_id written.
        """
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO inspection_runs (run_id, eng_item_id, plan_id,
                                                started_at, status)
                   VALUES (?, ?, ?, ?, ?)""",
                (run['run_id'], run['eng_item_id'], run.get('plan_id'),
                 run.get('started_at') or utc_now(), run.get('status', 'running')))
            conn.commit()
        return run['run_id']

    def get_run(self, run_id):
        """One inspection run by id, or None."""
        with self._connect() as conn:
            row = conn.execute('SELECT * FROM inspection_runs WHERE run_id = ?',
                               (run_id,)).fetchone()
        return dict(row) if row else None

    def list_runs(self, eng_item_id, limit=50):
        """Inspection history for a part, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                'SELECT * FROM inspection_runs WHERE eng_item_id = ?'
                ' ORDER BY started_at DESC LIMIT ?', (eng_item_id, limit)).fetchall()
        return [dict(row) for row in rows]

    def update_run(self, run_id, **fields):
        """Update named columns on an inspection run row."""
        allowed = {'completed_at', 'viewpoints_executed', 'viewpoints_skipped',
                   'anomalies_found', 'max_anomaly_score', 'overall_result',
                   'result_bundle_path', 'result_doc_id', 'upload_status',
                   'status', 'plan_id'}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return
        assignments = ', '.join(f'{k} = ?' for k in updates)
        with self._connect() as conn:
            conn.execute(
                f'UPDATE inspection_runs SET {assignments} WHERE run_id = ?',
                [*updates.values(), run_id])
            conn.commit()

    # --- sync log ------------------------------------------------------

    def log_sync(self, action, eng_item_id=None, detail=None):
        """Append an entry to the sync audit trail."""
        with self._connect() as conn:
            conn.execute(
                'INSERT INTO sync_log (timestamp, action, eng_item_id, detail)'
                ' VALUES (?, ?, ?, ?)',
                (utc_now(), action, eng_item_id, detail))
            conn.commit()

    def last_sync(self, action=None):
        """The most recent sync_log entry (optionally of one action), or None."""
        query = 'SELECT * FROM sync_log'
        params = []
        if action:
            query += ' WHERE action = ?'
            params.append(action)
        query += ' ORDER BY id DESC LIMIT 1'
        with self._connect() as conn:
            row = conn.execute(query, params).fetchone()
        return dict(row) if row else None

    def recent_sync_log(self, limit=50):
        """The most recent sync_log entries, newest first."""
        with self._connect() as conn:
            rows = conn.execute('SELECT * FROM sync_log ORDER BY id DESC LIMIT ?',
                                (limit,)).fetchall()
        return [dict(row) for row in rows]
