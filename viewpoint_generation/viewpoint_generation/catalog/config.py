"""Configuration for the 3DEXPERIENCE catalog subsystem.

Three dataclasses, all loadable from environment variables (the container gets
them from the host's `.env` via docker-compose):

    DXConfig      -- tenant URLs and credentials for the REST client
    SyncConfig    -- sync cadence and scoping/filtering policy
    CatalogPaths  -- on-disk locations of the DB and the file caches

`SyncConfig` follows the repo's config-dataclass convention (a `to_dict()`
returning `{field: {value, type, description, control, range}}`) so the ROS
node can auto-declare its fields as parameters the same way it does for
`RegionGrowingConfig` and friends. `DXConfig` deliberately does NOT expose a
`to_dict()`: it holds a password and must never be published as a ROS
parameter or written to a results file.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name, default=''):
    """Read an environment variable, treating unset and empty as the default."""
    value = os.environ.get(name, '')
    return value if value != '' else default


def _env_int(name, default):
    """Read an integer environment variable, falling back on a bad value."""
    raw = _env(name, '')
    if raw == '':
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name, default):
    """Read a float environment variable, falling back on a bad value."""
    raw = _env(name, '')
    if raw == '':
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass
class DXConfig:
    """Connection parameters for a 3DEXPERIENCE tenant.

    The passport (IAM) and space (3DSpace) hosts are frequently in *different*
    regions on cloud tenants -- this deployment authenticates against `eu1` and
    calls data services on `usw2`. Authentication is a CAS service-ticket
    redirect, so both URLs are needed and neither can be derived from the other.
    """

    passport_url: str = ''
    space_url: str = ''
    tenant: str = ''
    username: str = ''
    password: str = ''
    # Full SecurityContext triplet INCLUDING the 'ctx::' prefix, e.g.
    # 'ctx::VPLMProjectLeader.Company Name.Colin Acton Space'. Sent unencoded
    # as an HTTP header; url-encoded only when placed in a query string.
    security_context: str = ''
    # Optional STOMP/AMQP bridge for event-driven sync (Phase 5). When empty,
    # the catalog falls back to timer-based polling.
    jms_broker_url: str = ''
    # Network timeouts (seconds) for metadata calls and file transfers.
    request_timeout: float = 60.0
    download_timeout: float = 300.0

    @classmethod
    def from_env(cls):
        """Build a DXConfig from the DX_* environment variables."""
        return cls(
            passport_url=_env('DX_PASSPORT_URL').rstrip('/'),
            space_url=_env('DX_SPACE_URL').rstrip('/'),
            tenant=_env('DX_TENANT'),
            username=_env('DX_USERNAME'),
            password=_env('DX_PASSWORD'),
            security_context=_env('DX_SECURITY_CONTEXT'),
            jms_broker_url=_env('DX_JMS_BROKER_URL'),
            request_timeout=_env_float('DX_REQUEST_TIMEOUT', 60.0),
            download_timeout=_env_float('DX_DOWNLOAD_TIMEOUT', 300.0),
        )

    def is_configured(self):
        """True when enough is set to attempt a login.

        Returns:
            tuple: (bool, str) success flag and a message naming what's missing.
        """
        missing = [name for name, value in (
            ('DX_PASSPORT_URL', self.passport_url),
            ('DX_SPACE_URL', self.space_url),
            ('DX_USERNAME', self.username),
            ('DX_PASSWORD', self.password),
        ) if not value]
        if missing:
            return False, f"Missing 3DX configuration: {', '.join(missing)}."
        return True, '3DX configuration present.'

    def redacted(self):
        """A dict of the configuration safe for logging (password removed)."""
        return {
            'passport_url': self.passport_url,
            'space_url': self.space_url,
            'tenant': self.tenant,
            'username': self.username,
            'password': '***' if self.password else '',
            'security_context': self.security_context,
            'jms_broker_url': self.jms_broker_url,
        }


@dataclass
class SyncConfig:
    """Cadence and scope policy for catalog synchronization."""

    # Seconds between background incremental syncs. 0 disables the timer.
    sync_interval: int = 300
    # Search string identifying the parts in scope. This is the 3DX bookmark
    # name the operator curates ("Inspection Parts"); it is passed to the
    # dseng search as $searchStr. '*' matches everything the security context
    # can see.
    bookmark_scope: str = '*'
    # Comma-separated maturity states to keep, e.g. 'RELEASED,IN_WORK'.
    # Empty means no maturity filtering.
    maturity_filter: str = ''
    # Restrict results to a single collaborative space (matched against the
    # item's `collabspace`). Empty means every space the context can read.
    # The tenant-wide search returns demo/template content from other spaces,
    # so this is the practical scoping knob.
    collab_space_filter: str = ''
    # Maximum items pulled per search request, and overall.
    page_size: int = 100
    max_items: int = 1000
    # Download thumbnails during sync.
    fetch_thumbnails: bool = True
    # Anomaly score above which an inspection result auto-creates an Issue.
    ncr_threshold: float = 0.8
    # Cell identity stamped into uploaded plan/result envelopes.
    cell_id: str = 'alpha'

    @classmethod
    def from_env(cls):
        """Build a SyncConfig from the CATALOG_*/DX_* environment variables."""
        return cls(
            sync_interval=_env_int('CATALOG_SYNC_INTERVAL', 300),
            bookmark_scope=_env('DX_BOOKMARK_SCOPE', '*'),
            maturity_filter=_env('DX_MATURITY_FILTER', ''),
            collab_space_filter=_env('DX_COLLAB_SPACE', ''),
            page_size=_env_int('CATALOG_PAGE_SIZE', 100),
            max_items=_env_int('CATALOG_MAX_ITEMS', 1000),
            fetch_thumbnails=_env('CATALOG_FETCH_THUMBNAILS', '1') not in ('0', 'false', 'False'),
            ncr_threshold=_env_float('CATALOG_NCR_THRESHOLD', 0.8),
            cell_id=_env('CELL_ID', 'alpha'),
        )

    def maturity_states(self):
        """The maturity filter parsed into an upper-cased list (empty = all)."""
        return [s.strip().upper() for s in self.maturity_filter.split(',') if s.strip()]

    def to_dict(self):
        return {
            "sync_interval": {
                "value": self.sync_interval,
                "type": "integer",
                "description": "Seconds between background incremental syncs (0 disables)",
                "control": "slider",
                "range": [0, 3600],
            },
            "bookmark_scope": {
                "value": self.bookmark_scope,
                "type": "string",
                "description": "3DX search string scoping the catalog (bookmark name, or '*' for everything visible)",
                "control": "text",
            },
            "maturity_filter": {
                "value": self.maturity_filter,
                "type": "string",
                "description": "Comma-separated maturity states to keep (empty = all), e.g. 'RELEASED,IN_WORK'",
                "control": "text",
            },
            "collab_space_filter": {
                "value": self.collab_space_filter,
                "type": "string",
                "description": "Keep only items in this collaborative space (empty = all readable spaces)",
                "control": "text",
            },
            "page_size": {
                "value": self.page_size,
                "type": "integer",
                "description": "Items requested per 3DX search page",
                "control": "slider",
                "range": [1, 500],
            },
            "max_items": {
                "value": self.max_items,
                "type": "integer",
                "description": "Maximum items pulled in one full sync",
                "control": "slider",
                "range": [1, 10000],
            },
            "fetch_thumbnails": {
                "value": self.fetch_thumbnails,
                "type": "boolean",
                "description": "Download and cache item thumbnails during sync",
                "control": "toggle",
            },
            "ncr_threshold": {
                "value": self.ncr_threshold,
                "type": "float",
                "description": "Anomaly score above which an inspection result auto-creates a 3DX Issue",
                "control": "slider",
                "range": [0.0, 1.0],
            },
            "cell_id": {
                "value": self.cell_id,
                "type": "string",
                "description": "Identifier of this inspection cell, stamped into uploaded envelopes",
                "control": "text",
            },
        }


@dataclass
class CatalogPaths:
    """On-disk locations for the catalog database and file caches.

    Defaults are the container-side paths produced by the `./catalog`
    host mount declared in docker-compose.yaml.
    """

    db_path: Path = Path('/workspaces/shared_ws/catalog/catalog.db')
    step_dir: Path = Path('/workspaces/shared_ws/catalog/steps')
    thumb_dir: Path = Path('/workspaces/shared_ws/catalog/thumbnails')
    plan_dir: Path = Path('/workspaces/shared_ws/catalog/plans')
    result_dir: Path = Path('/workspaces/shared_ws/catalog/results')

    @classmethod
    def from_env(cls):
        """Build CatalogPaths from CATALOG_* environment variables.

        CATALOG_ROOT sets every unset path at once; the individual variables
        win when both are present.
        """
        root = _env('CATALOG_ROOT', '/workspaces/shared_ws/catalog')
        return cls(
            db_path=Path(_env('CATALOG_DB_PATH', f'{root}/catalog.db')),
            step_dir=Path(_env('CATALOG_STEP_DIR', f'{root}/steps')),
            thumb_dir=Path(_env('CATALOG_THUMB_DIR', f'{root}/thumbnails')),
            plan_dir=Path(_env('CATALOG_PLAN_DIR', f'{root}/plans')),
            result_dir=Path(_env('CATALOG_RESULT_DIR', f'{root}/results')),
        )

    def ensure(self):
        """Create every catalog directory that does not yet exist.

        Returns:
            tuple: (bool, str) success flag and message.
        """
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            for directory in (self.step_dir, self.thumb_dir,
                              self.plan_dir, self.result_dir):
                directory.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            return False, f'Could not create catalog directories: {e}'
        return True, 'Catalog directories ready.'

    def step_path(self, eng_item_id, revision):
        """Cache path for one item revision's STEP file."""
        safe_rev = str(revision).replace('/', '_') or 'unknown'
        return self.step_dir / str(eng_item_id) / f'{safe_rev}.stp'

    def thumb_path(self, eng_item_id):
        """Cache path for one item's thumbnail image."""
        return self.thumb_dir / f'{eng_item_id}.png'

    def plan_dir_for(self, eng_item_id):
        """Directory holding one item's inspection plan envelopes."""
        return self.plan_dir / str(eng_item_id)

    def result_dir_for(self, eng_item_id, run_id):
        """Directory holding one inspection run's result bundle."""
        return self.result_dir / str(eng_item_id) / str(run_id)

    @staticmethod
    def resolve_cached(stored, canonical):
        """Map a cache path recorded in the database onto this environment.

        The same `catalog.db` is read from two different mount points -- the
        host (`./catalog`) and the container (`/workspaces/shared_ws/catalog`)
        -- so an absolute path written by one is meaningless to the other.
        Cache locations are deterministic, so the canonical path for the
        current environment is authoritative whenever the file is actually
        there, and the stored value is only a fallback.

        Args:
            stored: The path recorded in the database (may be None).
            canonical: Where this environment expects the file to live.

        Returns:
            Path: The usable path, or None when the file is missing here.
        """
        if canonical is not None and Path(canonical).exists():
            return Path(canonical)
        if stored:
            candidate = Path(stored)
            if candidate.exists():
                return candidate
        return None
