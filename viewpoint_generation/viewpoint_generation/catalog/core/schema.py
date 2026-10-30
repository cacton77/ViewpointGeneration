"""SQLite schema and access layer for the parts catalog.

One file-backed database (`catalog.db`) holds four tables:

    parts            -- one row per known part, keyed by (backend, part_id)
    plans            -- inspection plans recorded for those parts (history)
    inspection_runs  -- executed inspections and their upload state
    sync_log         -- audit trail of sync actions, for debugging

**Migration.** Existing databases are upgraded in place, never rebuilt from
scratch and never by dropping data. Numbered steps in `VERSIONED_STEPS` are
tracked with `PRAGMA user_version`; before the first step runs on an existing
database a consistent copy is saved beside it as `catalog.db.bak-<UTC stamp>`.
Every step is idempotent.

    0 -> legacy column additions the pre-versioning releases made in place
    1 -> plan contract: summary_json/summary_text/stage_rank replace the
         planner-specific counts
    2 -> backend identity: (backend, part_id) replaces eng_item_id,
         change_token replaces cestamp, container replaces collab_space,
         storage_key and extra_json are added. Pre-existing rows are 3DX rows.

All access goes through `CatalogDB`, which opens a short-lived connection per
operation and uses parameterized queries throughout.
"""

import json
import logging
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from viewpoint_generation.catalog.core.paths import storage_key

logger = logging.getLogger(__name__)

# Every row that predates backends came from the 3DEXPERIENCE catalog.
LEGACY_BACKEND = 'dx3'

PARTS_TABLE = """
CREATE TABLE IF NOT EXISTS parts (
    backend            TEXT NOT NULL,
    part_id            TEXT NOT NULL,
    storage_key        TEXT NOT NULL UNIQUE,
    title              TEXT NOT NULL,
    part_number        TEXT,
    revision           TEXT,
    change_token       TEXT NOT NULL,
    maturity           TEXT,
    container          TEXT,
    description        TEXT,
    extra_json         TEXT,
    thumbnail_url      TEXT,
    thumbnail_path     TEXT,
    geometry_available INTEGER,
    geometry_path      TEXT,
    geometry_token     TEXT,
    sync_status        TEXT DEFAULT 'new',
    first_seen         TEXT NOT NULL,
    last_synced        TEXT NOT NULL,
    PRIMARY KEY (backend, part_id)
)
"""

PLANS_TABLE = """
CREATE TABLE IF NOT EXISTS plans (
    plan_id         TEXT PRIMARY KEY,
    backend         TEXT NOT NULL,
    part_id         TEXT NOT NULL,
    remote_ref      TEXT,
    change_token    TEXT NOT NULL,
    file_path       TEXT NOT NULL,
    generated_at    TEXT NOT NULL,
    uploaded_at     TEXT,
    upload_status   TEXT DEFAULT 'local',
    is_current      INTEGER DEFAULT 1,
    source          TEXT DEFAULT 'local',
    -- Plan contract: supplied by the producer when the plan is recorded.
    stage           TEXT,
    stage_rank      INTEGER DEFAULT -1,
    summary_json    TEXT,
    summary_text    TEXT,
    FOREIGN KEY (backend, part_id) REFERENCES parts(backend, part_id)
)
"""

INSPECTION_RUNS_TABLE = """
CREATE TABLE IF NOT EXISTS inspection_runs (
    run_id              TEXT PRIMARY KEY,
    backend             TEXT NOT NULL,
    part_id             TEXT NOT NULL,
    plan_id             TEXT REFERENCES plans(plan_id),
    started_at          TEXT NOT NULL,
    completed_at        TEXT,
    viewpoints_executed INTEGER,
    viewpoints_skipped  INTEGER DEFAULT 0,
    anomalies_found     INTEGER,
    max_anomaly_score   REAL,
    overall_result      TEXT,
    result_bundle_path  TEXT,
    remote_ref          TEXT,
    issue_ref           TEXT,
    upload_status       TEXT DEFAULT 'pending',
    status              TEXT DEFAULT 'running',
    FOREIGN KEY (backend, part_id) REFERENCES parts(backend, part_id)
)
"""

SYNC_LOG_TABLE = """
CREATE TABLE IF NOT EXISTS sync_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    action          TEXT NOT NULL,
    backend         TEXT,
    part_id         TEXT,
    detail          TEXT
)
"""

INDEXES = (
    'CREATE INDEX IF NOT EXISTS idx_plans_part ON plans(backend, part_id)',
    'CREATE INDEX IF NOT EXISTS idx_runs_part ON inspection_runs(backend, part_id)',
    'CREATE INDEX IF NOT EXISTS idx_sync_log_time ON sync_log(timestamp)',
)

TABLES = (PARTS_TABLE, PLANS_TABLE, INSPECTION_RUNS_TABLE, SYNC_LOG_TABLE)

# --- legacy (version 0) ----------------------------------------------------------

# Columns the pre-versioning releases added to existing databases in place.
# Applied only to a database at user_version 0, before the numbered steps.
LEGACY_COLUMNS = {
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

# Ranks for stage labels recorded before the producer supplied its own, copied
# from ViewpointGeneration's plan_io.STAGES when the plan contract was
# introduced. Used only to fill stage_rank on rows that predate it.
LEGACY_STAGE_RANKS = {'segmented': 0, 'clustered': 1, 'projected': 2, 'ordered': 3}

# Planner-specific plan columns from before the plan contract. Step 1 carries
# their values into summary_json; step 2 drops the columns.
LEGACY_PLAN_SUMMARY_COLUMNS = ('num_regions', 'num_clusters', 'num_viewpoints',
                               'seg_algorithm', 'traversal_algo')


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


def _legacy_columns(conn):
    """Add any columns a pre-versioning database is missing."""
    for table, columns in LEGACY_COLUMNS.items():
        if not _table_exists(conn, table):
            continue
        existing = _column_names(conn, table)
        for name, definition in columns:
            if name not in existing:
                logger.info('Migrating %s: adding column %s', table, name)
                # Table and column names are module constants, never user input.
                conn.execute(f'ALTER TABLE {table} ADD COLUMN {name} {definition}')


def _legacy_summary_text(summary):
    """One-line description of a legacy plan row, as plan_io.describe() wrote it."""
    bits = [f'{summary[key]} {label}' for key, label in (
        ('num_regions', 'regions'), ('num_clusters', 'clusters'),
        ('num_viewpoints', 'viewpoints')) if summary.get(key)]
    return ', '.join(bits) or 'empty'


def _step_1_plan_contract(conn):
    """Give every plan row the producer-supplied fields of the plan contract.

    Rows recorded before the contract carry their size in planner-specific
    columns; those values move into summary_json, summary_text is regenerated
    in the producer's format, and stage_rank is derived from the stage label.
    """
    if not _table_exists(conn, 'plans') or 'eng_item_id' not in _column_names(conn, 'plans'):
        return
    existing = _column_names(conn, 'plans')
    for name, definition in (('stage_rank', 'INTEGER DEFAULT -1'),
                             ('summary_json', 'TEXT'), ('summary_text', 'TEXT')):
        if name not in existing:
            conn.execute(f'ALTER TABLE plans ADD COLUMN {name} {definition}')
    legacy = [c for c in LEGACY_PLAN_SUMMARY_COLUMNS if c in existing]
    rows = conn.execute(
        f"SELECT plan_id, stage{''.join(', ' + c for c in legacy)} FROM plans"
        ' WHERE summary_json IS NULL').fetchall()
    for row in rows:
        values = dict(zip(['plan_id', 'stage', *legacy], row))
        summary = {c: values[c] for c in legacy if values[c] is not None}
        conn.execute(
            'UPDATE plans SET summary_json = ?, summary_text = ?, stage_rank = ?'
            ' WHERE plan_id = ?',
            (json.dumps(summary, sort_keys=True), _legacy_summary_text(summary),
             LEGACY_STAGE_RANKS.get(values['stage'] or '', -1), values['plan_id']))


def _rows(conn, table):
    """Every row of a table as dicts (empty when the table is absent)."""
    if not _table_exists(conn, table):
        return []
    cursor = conn.execute(f'SELECT * FROM {table}')
    names = [d[0] for d in cursor.description]
    return [dict(zip(names, row)) for row in cursor.fetchall()]


def _step_2_backend_identity(conn):
    """Key everything by (backend, part_id) instead of the 3DX eng_item_id.

    Tables are rebuilt (SQLite cannot change a primary key in place) with
    every row carried over: parts, every plan row and its history, runs and
    the sync log. Cached files are not moved; `CatalogPaths` still resolves
    the legacy layout.
    """
    if _table_exists(conn, 'parts') and 'part_id' in _column_names(conn, 'parts'):
        return
    parts = _rows(conn, 'parts')
    plans = _rows(conn, 'plans')
    runs = _rows(conn, 'inspection_runs')
    log = _rows(conn, 'sync_log')
    for table in ('sync_log', 'inspection_runs', 'plans', 'parts'):
        conn.execute(f'DROP TABLE IF EXISTS {table}')
    for statement in TABLES:
        conn.execute(statement)

    now = utc_now()
    for row in parts:
        part_id = row['eng_item_id']
        extra = {'type': row['type']} if row.get('type') else {}
        available = row.get('step_available')
        conn.execute(
            'INSERT INTO parts (backend, part_id, storage_key, title, part_number,'
            ' revision, change_token, maturity, container, description, extra_json,'
            ' thumbnail_url, thumbnail_path, geometry_available, geometry_path,'
            ' geometry_token, sync_status, first_seen, last_synced)'
            ' VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (LEGACY_BACKEND, part_id, storage_key(LEGACY_BACKEND, part_id),
             row.get('title') or '', row.get('part_number'), row.get('revision'),
             row.get('cestamp') or '', row.get('maturity'), row.get('collab_space'),
             row.get('description'), json.dumps(extra, sort_keys=True),
             row.get('thumbnail_url'), row.get('thumbnail_path'),
             None if available is None else int(bool(available)),
             row.get('step_path'), row.get('step_cestamp'),
             row.get('sync_status') or 'synced', row.get('first_seen') or now,
             row.get('last_synced') or now))
    for row in plans:
        conn.execute(
            'INSERT INTO plans (plan_id, backend, part_id, remote_ref, change_token,'
            ' file_path, generated_at, uploaded_at, upload_status, is_current, source,'
            ' stage, stage_rank, summary_json, summary_text)'
            ' VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (row['plan_id'], LEGACY_BACKEND, row['eng_item_id'], row.get('plan_doc_id'),
             row.get('cestamp') or '', row['file_path'], row.get('generated_at') or now,
             row.get('uploaded_at'), row.get('upload_status') or 'local',
             1 if row.get('is_current', 1) else 0, row.get('source') or 'local',
             row.get('stage'), row.get('stage_rank', -1) if row.get('stage_rank')
             is not None else -1, row.get('summary_json') or '{}',
             row.get('summary_text') or ''))
    for row in runs:
        conn.execute(
            'INSERT INTO inspection_runs (run_id, backend, part_id, plan_id, started_at,'
            ' completed_at, viewpoints_executed, viewpoints_skipped, anomalies_found,'
            ' max_anomaly_score, overall_result, result_bundle_path, remote_ref,'
            ' upload_status, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (row['run_id'], LEGACY_BACKEND, row['eng_item_id'], row.get('plan_id'),
             row.get('started_at') or now, row.get('completed_at'),
             row.get('viewpoints_executed'), row.get('viewpoints_skipped') or 0,
             row.get('anomalies_found'), row.get('max_anomaly_score'),
             row.get('overall_result'), row.get('result_bundle_path'),
             row.get('result_doc_id'), row.get('upload_status') or 'pending',
             row.get('status') or 'running'))
    for row in log:
        conn.execute(
            'INSERT INTO sync_log (id, timestamp, action, backend, part_id, detail)'
            ' VALUES (?, ?, ?, ?, ?, ?)',
            (row['id'], row['timestamp'], row['action'],
             LEGACY_BACKEND if row.get('eng_item_id') else None,
             row.get('eng_item_id'), row.get('detail')))


# Numbered migrations. Step N brings a database from user_version N-1 to N.
VERSIONED_STEPS = (_step_1_plan_contract, _step_2_backend_identity)
SCHEMA_VERSION = len(VERSIONED_STEPS)


def _backup(db_path):
    """Save a consistent copy of the database beside it before migrating.

    Returns:
        Path: the backup written.
    """
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup = db_path.with_name(f'{db_path.name}.bak-{stamp}')
    counter = 2
    while backup.exists():
        backup = db_path.with_name(f'{db_path.name}.bak-{stamp}-{counter}')
        counter += 1
    with closing(sqlite3.connect(str(db_path))) as source, \
            closing(sqlite3.connect(str(backup))) as dest:
        source.backup(dest)
    return backup


def init_db(db_path):
    """Create the catalog database and its tables, migrating an older one.

    Args:
        db_path: Path to the SQLite file. Parent directories are created.

    Returns:
        tuple: (bool, str) success flag and message.
    """
    db_path = Path(db_path)
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # Autocommit, so the migration controls its own transaction and can
        # switch foreign keys off around the table rebuilds.
        with closing(sqlite3.connect(str(db_path), isolation_level=None)) as conn:
            existed = _table_exists(conn, 'parts')
            version = conn.execute('PRAGMA user_version').fetchone()[0]
            if not existed:
                conn.execute('BEGIN')
                for statement in TABLES + INDEXES:
                    conn.execute(statement)
                conn.execute(f'PRAGMA user_version = {SCHEMA_VERSION}')
                conn.execute('COMMIT')
                return True, f'Catalog database created at {db_path}.'

            if version < SCHEMA_VERSION:
                backup = _backup(db_path)
                logger.info('Backed up %s to %s before migrating from schema %d to %d.',
                            db_path, backup, version, SCHEMA_VERSION)
                conn.execute('PRAGMA foreign_keys = OFF')
                conn.execute('BEGIN')
                try:
                    if version == 0:
                        _legacy_columns(conn)
                    for step_number in range(version + 1, SCHEMA_VERSION + 1):
                        logger.info('Migrating %s to schema %d.', db_path, step_number)
                        VERSIONED_STEPS[step_number - 1](conn)
                    conn.execute(f'PRAGMA user_version = {SCHEMA_VERSION}')
                    conn.execute('COMMIT')
                except Exception:
                    conn.execute('ROLLBACK')
                    raise
                conn.execute('PRAGMA foreign_keys = ON')

            for statement in TABLES + INDEXES:
                conn.execute(statement)
    except (sqlite3.Error, OSError) as e:
        return False, f'Could not initialize catalog database at {db_path}: {e}'
    return True, f'Catalog database ready at {db_path}.'


class CatalogDB:
    """Query and update helpers over the catalog SQLite database.

    Every method opens its own connection, so instances are safe to share
    between the sync thread, ROS service callbacks and the picker.
    """

    def __init__(self, db_path):
        """Open (creating or migrating if needed) the catalog database."""
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

    def _execute(self, sql, params=()):
        with closing(self._connect()) as conn, conn:
            conn.execute(sql, params)

    def _one(self, sql, params=()):
        with closing(self._connect()) as conn:
            row = conn.execute(sql, params).fetchone()
        return dict(row) if row else None

    def _all(self, sql, params=()):
        with closing(self._connect()) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    # --- parts -------------------------------------------------------------------

    def get_all_parts(self, backend):
        """Every catalogued part of one backend, keyed by part_id."""
        return {row['part_id']: row for row in
                self._all('SELECT * FROM parts WHERE backend = ?', (backend,))}

    def list_parts(self, search=None, maturity=None, container=None,
                   include_archived=False, backend=None):
        """Parts matching optional filters, ordered by title.

        Args:
            search: Case-insensitive substring of title, part number, or
                description.
            maturity: Exact maturity state to keep (e.g. 'RELEASED').
            container: Exact container (collaborative space, folder) to keep.
            include_archived: Include parts no longer present remotely.
            backend: Keep only this backend's parts.
        """
        clauses, params = [], []
        if search:
            clauses.append('(LOWER(title) LIKE ? OR LOWER(IFNULL(part_number, "")) LIKE ?'
                           ' OR LOWER(IFNULL(description, "")) LIKE ?)')
            needle = f'%{search.lower()}%'
            params.extend([needle, needle, needle])
        if maturity:
            clauses.append('UPPER(IFNULL(maturity, "")) = ?')
            params.append(maturity.upper())
        if container:
            clauses.append('container = ?')
            params.append(container)
        if backend:
            clauses.append('backend = ?')
            params.append(backend)
        if not include_archived:
            clauses.append("sync_status != 'archived'")
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ''
        return self._all(f'SELECT * FROM parts{where} ORDER BY title COLLATE NOCASE', params)

    def get_part(self, backend, part_id):
        return self._one('SELECT * FROM parts WHERE backend = ? AND part_id = ?',
                         (backend, part_id))

    def get_part_by_key(self, key):
        return self._one('SELECT * FROM parts WHERE storage_key = ?', (key,))

    def insert_part(self, remote, key):
        """Insert a newly discovered RemotePart with sync_status='new'."""
        now = utc_now()
        self._execute(
            'INSERT INTO parts (backend, part_id, storage_key, title, part_number,'
            ' revision, change_token, maturity, container, description, extra_json,'
            ' geometry_available, sync_status, first_seen, last_synced)'
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'new', ?, ?)",
            (remote.backend, remote.part_id, key, remote.title or '', remote.part_number,
             remote.revision, remote.change_token or '', remote.maturity, remote.container,
             remote.description, json.dumps(dict(remote.extra), sort_keys=True),
             None if remote.geometry_available is None else int(remote.geometry_available),
             now, now))

    def update_part(self, remote, sync_status='modified'):
        """Refresh a part's metadata from the remote copy."""
        self._execute(
            'UPDATE parts SET title = ?, part_number = ?, revision = ?, change_token = ?,'
            ' maturity = ?, container = ?, description = ?, extra_json = ?,'
            ' sync_status = ?, last_synced = ? WHERE backend = ? AND part_id = ?',
            (remote.title or '', remote.part_number, remote.revision,
             remote.change_token or '', remote.maturity, remote.container,
             remote.description, json.dumps(dict(remote.extra), sort_keys=True),
             sync_status, utc_now(), remote.backend, remote.part_id))

    def touch_part(self, backend, part_id, sync_status='synced'):
        """Mark an unchanged part as seen in this sync pass."""
        self._execute('UPDATE parts SET last_synced = ?, sync_status = ?'
                      ' WHERE backend = ? AND part_id = ?',
                      (utc_now(), sync_status, backend, part_id))

    def archive_parts(self, backend, part_ids):
        """Mark parts no longer visible remotely as archived."""
        if not part_ids:
            return
        with closing(self._connect()) as conn, conn:
            conn.executemany(
                "UPDATE parts SET sync_status = 'archived', last_synced = ?"
                ' WHERE backend = ? AND part_id = ?',
                [(utc_now(), backend, part_id) for part_id in part_ids])

    def set_thumbnail(self, backend, part_id, thumbnail_path):
        self._execute('UPDATE parts SET thumbnail_path = ? WHERE backend = ? AND part_id = ?',
                      (str(thumbnail_path) if thumbnail_path else None, backend, part_id))

    def set_geometry_available(self, backend, part_id, available):
        self._execute('UPDATE parts SET geometry_available = ?'
                      ' WHERE backend = ? AND part_id = ?',
                      (1 if available else 0, backend, part_id))

    def set_geometry_cache(self, backend, part_id, geometry_path, geometry_token):
        """Record the local STEP file and the change token it was fetched at."""
        self._execute(
            'UPDATE parts SET geometry_path = ?, geometry_token = ?, geometry_available = 1'
            ' WHERE backend = ? AND part_id = ?',
            (str(geometry_path) if geometry_path else None, geometry_token,
             backend, part_id))

    def clear_geometry_cache(self, backend, part_id):
        """Forget a cached STEP file (used when the remote part changed)."""
        self._execute('UPDATE parts SET geometry_path = NULL, geometry_token = NULL'
                      ' WHERE backend = ? AND part_id = ?', (backend, part_id))

    def counts(self, backend=None):
        """Summary counts for status displays.

        Returns:
            dict: total, ready (geometry cached), and per-sync_status counts.
        """
        scope, params = ('', ()) if backend is None else (' AND backend = ?', (backend,))
        with closing(self._connect()) as conn:
            total = conn.execute("SELECT COUNT(*) FROM parts WHERE sync_status != 'archived'"
                                 + scope, params).fetchone()[0]
            ready = conn.execute('SELECT COUNT(*) FROM parts WHERE geometry_path IS NOT NULL'
                                 " AND sync_status != 'archived'" + scope, params).fetchone()[0]
            by_status = {row[0]: row[1] for row in conn.execute(
                'SELECT sync_status, COUNT(*) FROM parts WHERE 1 = 1' + scope
                + ' GROUP BY sync_status', params)}
        return {'total': total, 'ready': ready, 'by_status': by_status}

    # --- plans -------------------------------------------------------------------

    def get_current_plan(self, backend, part_id):
        return self._one('SELECT * FROM plans WHERE backend = ? AND part_id = ?'
                         ' AND is_current = 1 ORDER BY generated_at DESC LIMIT 1',
                         (backend, part_id))

    def get_plan(self, plan_id):
        return self._one('SELECT * FROM plans WHERE plan_id = ?', (plan_id,))

    def list_plans(self, backend, part_id):
        """Every plan recorded for a part, newest first.

        Two pipeline stages can complete inside the same second, so rowid
        breaks ties on generated_at.
        """
        return self._all('SELECT * FROM plans WHERE backend = ? AND part_id = ?'
                         ' ORDER BY generated_at DESC, rowid DESC', (backend, part_id))

    def upsert_plan(self, plan):
        """Insert or update a plan row, making it current unless told otherwise.

        Args:
            plan: dict with at least plan_id, backend, part_id, change_token,
                file_path. Everything describing the plan's content (stage,
                stage_rank, summary_json, summary_text) comes from the producer.
                A re-recorded plan keeps its original generated_at.
        """
        with closing(self._connect()) as conn, conn:
            if plan.get('is_current', 1):
                conn.execute('UPDATE plans SET is_current = 0 WHERE backend = ? AND part_id = ?',
                             (plan['backend'], plan['part_id']))
            conn.execute(
                """INSERT INTO plans (plan_id, backend, part_id, remote_ref, change_token,
                                      file_path, generated_at, uploaded_at, upload_status,
                                      is_current, source, stage, stage_rank, summary_json,
                                      summary_text)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(plan_id) DO UPDATE SET
                       remote_ref = excluded.remote_ref,
                       change_token = excluded.change_token,
                       file_path = excluded.file_path,
                       uploaded_at = excluded.uploaded_at,
                       upload_status = excluded.upload_status,
                       is_current = excluded.is_current,
                       source = excluded.source,
                       stage = excluded.stage,
                       stage_rank = excluded.stage_rank,
                       summary_json = excluded.summary_json,
                       summary_text = excluded.summary_text""",
                (plan['plan_id'], plan['backend'], plan['part_id'], plan.get('remote_ref'),
                 plan.get('change_token') or '', str(plan['file_path']),
                 plan.get('generated_at') or utc_now(), plan.get('uploaded_at'),
                 plan.get('upload_status', 'local'), 1 if plan.get('is_current', 1) else 0,
                 plan.get('source', 'local'), plan.get('stage'),
                 int(plan.get('stage_rank', -1)), plan.get('summary_json') or '{}',
                 plan.get('summary_text') or ''))
        return plan['plan_id']

    def set_current_plan(self, plan_id):
        """Make one plan its part's current plan (an operator's explicit choice).

        Returns:
            bool: False when no such plan exists.
        """
        with closing(self._connect()) as conn, conn:
            row = conn.execute('SELECT backend, part_id FROM plans WHERE plan_id = ?',
                               (plan_id,)).fetchone()
            if row is None:
                return False
            conn.execute('UPDATE plans SET is_current = 0 WHERE backend = ? AND part_id = ?',
                         (row['backend'], row['part_id']))
            conn.execute('UPDATE plans SET is_current = 1 WHERE plan_id = ?', (plan_id,))
        return True

    def update_plan(self, plan_id, **fields):
        """Update named columns on a plan row."""
        allowed = {'remote_ref', 'uploaded_at', 'upload_status', 'is_current', 'file_path',
                   'source', 'change_token', 'stage', 'stage_rank', 'summary_json',
                   'summary_text', 'generated_at'}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if updates:
            assignments = ', '.join(f'{k} = ?' for k in updates)
            self._execute(f'UPDATE plans SET {assignments} WHERE plan_id = ?',
                          [*updates.values(), plan_id])

    # --- inspection runs -----------------------------------------------------------

    def create_run(self, run):
        """Insert a new inspection run row (run_id, backend, part_id required)."""
        self._execute(
            'INSERT INTO inspection_runs (run_id, backend, part_id, plan_id, started_at, status)'
            ' VALUES (?, ?, ?, ?, ?, ?)',
            (run['run_id'], run['backend'], run['part_id'], run.get('plan_id'),
             run.get('started_at') or utc_now(), run.get('status', 'running')))
        return run['run_id']

    def get_run(self, run_id):
        return self._one('SELECT * FROM inspection_runs WHERE run_id = ?', (run_id,))

    def list_runs(self, backend, part_id, limit=50):
        """Inspection history for a part, newest first."""
        return self._all('SELECT * FROM inspection_runs WHERE backend = ? AND part_id = ?'
                         ' ORDER BY started_at DESC LIMIT ?', (backend, part_id, limit))

    def update_run(self, run_id, **fields):
        """Update named columns on an inspection run row."""
        allowed = {'completed_at', 'viewpoints_executed', 'viewpoints_skipped',
                   'anomalies_found', 'max_anomaly_score', 'overall_result',
                   'result_bundle_path', 'remote_ref', 'issue_ref', 'upload_status',
                   'status', 'plan_id'}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if updates:
            assignments = ', '.join(f'{k} = ?' for k in updates)
            self._execute(f'UPDATE inspection_runs SET {assignments} WHERE run_id = ?',
                          [*updates.values(), run_id])

    # --- sync log ------------------------------------------------------------------

    def log_sync(self, action, backend=None, part_id=None, detail=None):
        """Append an entry to the sync audit trail."""
        self._execute('INSERT INTO sync_log (timestamp, action, backend, part_id, detail)'
                      ' VALUES (?, ?, ?, ?, ?)', (utc_now(), action, backend, part_id, detail))

    def last_sync(self, action=None, backend=None):
        """The most recent sync_log entry (optionally filtered), or None."""
        clauses, params = [], []
        if action:
            clauses.append('action = ?')
            params.append(action)
        if backend:
            clauses.append('backend = ?')
            params.append(backend)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ''
        return self._one(f'SELECT * FROM sync_log{where} ORDER BY id DESC LIMIT 1', params)

    def recent_sync_log(self, limit=50):
        """The most recent sync_log entries, newest first."""
        return self._all('SELECT * FROM sync_log ORDER BY id DESC LIMIT ?', (limit,))
