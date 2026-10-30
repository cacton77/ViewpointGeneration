"""On-disk layout of the catalog, and resolution of cached files.

Everything lives under `CATALOG_ROOT`::

    catalog.db
    {storage_key}/geometry/{token_hash}.stp   STEP, one per change token
    {storage_key}/thumbnail.png
    plans/{storage_key}/...                   plan files
    results/{storage_key}/{run_id}/...        inspection result bundles

`storage_key` is `{backend}-{sha1(part_id)[:16]}`: a part_id is the
backend's own identifier and may contain '/' or other characters no path
should, so it is never used as a path component for anything written.

**Legacy layout.** Before backends existed the cache was keyed by the 3DX
engineering item id: `steps/{id}/{revision}.stp`, `thumbnails/{id}.png`,
`plans/{id}/`. Those files are never moved or re-downloaded; the resolver
below still finds them (read-only), and new files go to the new layout.

**Environment-relative paths.** The same `catalog.db` is read from the host
(`./catalog`) and the container (`/catalog`), so an absolute path stored in
it may be meaningless here. Cache locations are deterministic, so the
canonical location for this environment wins whenever the file is there, and
the stored path is only a fallback. Never trust a stored path directly.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

from viewpoint_generation.catalog.core.config import env


def storage_key(backend, part_id):
    """Filesystem-safe, stable key for one part of one backend."""
    digest = hashlib.sha1(str(part_id).encode('utf-8')).hexdigest()[:16]
    return f'{backend}-{digest}'


def token_hash(change_token):
    """Short filesystem-safe digest of a change token."""
    return hashlib.sha1(str(change_token or '').encode('utf-8')).hexdigest()[:12]


def _safe_component(value):
    """True when value can be read back as a single legacy path component."""
    value = str(value or '')
    return bool(value) and value not in ('.', '..') and '/' not in value and '\\' not in value


@dataclass
class CatalogPaths:
    """Locations of the catalog database and caches."""

    root: Path = Path('/catalog')
    db_path: Path = Path('/catalog/catalog.db')
    plan_dir: Path = Path('/catalog/plans')
    result_dir: Path = Path('/catalog/results')
    # Legacy (pre-backend) cache directories, read but never written.
    legacy_step_dir: Path = Path('/catalog/steps')
    legacy_thumb_dir: Path = Path('/catalog/thumbnails')

    @classmethod
    def from_env(cls, root=None):
        """Build CatalogPaths from CATALOG_* environment variables.

        CATALOG_ROOT sets every unset path at once; the individual variables
        win when both are present.
        """
        root = Path(root or env('CATALOG_ROOT', '/catalog'))
        return cls(
            root=root,
            db_path=Path(env('CATALOG_DB_PATH', str(root / 'catalog.db'))),
            plan_dir=Path(env('CATALOG_PLAN_DIR', str(root / 'plans'))),
            result_dir=Path(env('CATALOG_RESULT_DIR', str(root / 'results'))),
            legacy_step_dir=Path(env('CATALOG_STEP_DIR', str(root / 'steps'))),
            legacy_thumb_dir=Path(env('CATALOG_THUMB_DIR', str(root / 'thumbnails'))),
        )

    def ensure(self):
        """Create the directories the catalog writes into.

        Returns:
            tuple: (bool, str) success flag and message.
        """
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            for directory in (self.root, self.plan_dir, self.result_dir):
                directory.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            return False, f'Could not create catalog directories: {e}'
        return True, 'Catalog directories ready.'

    # --- new layout -----------------------------------------------------------

    def part_dir(self, key):
        return self.root / key

    def geometry_dir(self, key):
        return self.root / key / 'geometry'

    def geometry_path(self, key, change_token):
        """Where the STEP for one change token of a part is cached."""
        return self.geometry_dir(key) / f'{token_hash(change_token)}.stp'

    def thumb_path(self, key):
        return self.root / key / 'thumbnail.png'

    def plans_for(self, key):
        """Directory new plan files for a part are written into."""
        return self.plan_dir / key

    def result_dir_for(self, key, run_id):
        """Directory holding one inspection run's result bundle."""
        return self.result_dir / key / str(run_id)

    # --- legacy layout (read-only) ----------------------------------------------

    def legacy_step_path(self, part_id, revision):
        """`steps/{id}/{revision}.stp`, or None when part_id is not a safe name."""
        if not _safe_component(part_id):
            return None
        safe_rev = str(revision).replace('/', '_') if revision else 'unknown'
        return self.legacy_step_dir / str(part_id) / f'{safe_rev}.stp'

    def legacy_thumb_path(self, part_id):
        if not _safe_component(part_id):
            return None
        return self.legacy_thumb_dir / f'{part_id}.png'

    def legacy_plans_for(self, part_id):
        if not _safe_component(part_id):
            return None
        return self.plan_dir / str(part_id)

    def legacy_results_dirs(self, part_id):
        """`steps/{id}/*_results`, where early versions wrote plan files."""
        if not _safe_component(part_id):
            return []
        step_dir = self.legacy_step_dir / str(part_id)
        return sorted(step_dir.glob('*_results')) if step_dir.exists() else []

    # --- resolution ------------------------------------------------------------

    @staticmethod
    def resolve_cached(stored, *candidates):
        """Map a cache path recorded in the database onto this environment.

        Args:
            stored: The path recorded in the database (may be None).
            candidates: Where this environment expects the file to live, in
                order of preference (None entries are skipped).

        Returns:
            Path: The usable path, or None when the file is missing here.
        """
        for candidate in candidates:
            if candidate is not None and Path(candidate).exists():
                return Path(candidate)
        if stored:
            path = Path(stored)
            if path.exists():
                return path
        return None
