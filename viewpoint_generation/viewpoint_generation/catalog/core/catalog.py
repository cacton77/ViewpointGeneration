"""The catalog core: a local index of PLM parts, kept in step with its backends.

`Catalog` keeps the SQLite catalog aligned with the parts each configured
backend reports, lazily caches what inspection needs (STEP files,
thumbnails), and records, versions and uploads the plans and results the cell
produces. It talks to PLM systems only through the `PLMBackend` protocol
(`backends.base`) and never imports a concrete backend.

Change detection is by each part's opaque `change_token` (on 3DX, the
cestamp). When it moves, the cached STEP is dropped and every plan generated
against the old token becomes STALE.

**The cell keeps working when the PLM does not.** Everything selection and
planning need is read from the local database and disk; a backend is
contacted only to sync, to fetch geometry that is not cached yet, and to
upload.
"""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

from viewpoint_generation.catalog.backends.base import (BackendError,
                                                        Capability, ScopeTruncated,
                                                        part_from_row)
from viewpoint_generation.catalog.core import plan_contract
from viewpoint_generation.catalog.core.config import CoreConfig
from viewpoint_generation.catalog.core.paths import CatalogPaths, storage_key
from viewpoint_generation.catalog.core.runs import RunsMixin
from viewpoint_generation.catalog.core.schema import CatalogDB, utc_now

logger = logging.getLogger(__name__)

STEP_SUFFIXES = ('.stp', '.step')


class PlanStatus(str, Enum):
    """Validity of the inspection plan held for a part.

    A plan is bound to the change token it was generated against, so a plan
    whose token no longer matches the part's is STALE: the geometry may have
    moved under it, and the operator decides whether to re-plan.
    """

    CURRENT = 'CURRENT'
    DOWNLOADED = 'DOWNLOADED'
    UPDATED_FROM_REMOTE = 'UPDATED_FROM_REMOTE'
    STALE = 'STALE'
    NONE = 'NONE'
    # An operator picked this specific plan in the picker. It overrides the
    # staleness guard -- an explicit choice is honoured even when the plan
    # targets an earlier CAD revision -- so consumers must warn loudly rather
    # than refuse.
    SELECTED = 'SELECTED'


@dataclass
class SyncResult:
    """Outcome of one synchronization pass over one backend."""

    backend: str = ''
    added: int = 0
    modified: int = 0
    unchanged: int = 0
    archived: int = 0
    errors: int = 0
    success: bool = True
    message: str = ''
    started_at: str = ''
    completed_at: str = ''
    duration_sec: float = 0.0
    error_details: list = field(default_factory=list)

    def to_dict(self):
        """A plain dict of the result, for JSON responses and logging."""
        return asdict(self)

    def summary(self):
        """One-line human-readable summary."""
        return (f'{self.added} added, {self.modified} modified, '
                f'{self.unchanged} unchanged, {self.archived} archived, '
                f'{self.errors} errors ({self.duration_sec:.1f}s)')


class Catalog(RunsMixin):
    """The local parts catalog over one or more PLM backends.

    Args:
        backends: Mapping of registry name -> `PLMBackend`.
        config: A `CoreConfig`.
        paths: A `CatalogPaths`; derived from `config.root` when omitted.
    """

    def __init__(self, backends, config=None, paths=None):
        self.backends = dict(backends)
        self.config = config or CoreConfig()
        self.paths = paths or CatalogPaths.from_env(self.config.root)
        success, message = self.paths.ensure()
        if not success:
            raise RuntimeError(message)
        self.db = CatalogDB(self.paths.db_path)
        self.last_results = {}
        self.syncing = False
        self._connection = {}
        self._event_stops = {}

    # --- backends -------------------------------------------------------------

    def backend(self, name):
        """A hosted backend by name, or None."""
        return self.backends.get(name)

    def has(self, name, capability):
        """True when backend `name` is hosted here and declares `capability`."""
        backend = self.backend(name)
        return backend is not None and capability in backend.capabilities

    def _connect(self, backend):
        """Connect a backend, remembering the outcome for status displays.

        Returns:
            tuple: (bool, str)
        """
        configured, message = backend.is_configured()
        if not configured:
            self._connection[backend.name] = (False, message)
            return False, message
        try:
            backend.connect()
        except BackendError as e:
            self._connection[backend.name] = (False, str(e))
            return False, str(e)
        self._connection[backend.name] = (True, 'Connected.')
        return True, 'Connected.'

    def _account(self, name):
        """The account a backend acts as, for plan and result provenance."""
        backend = self.backend(name)
        try:
            return (backend.describe() or {}).get('user', '') if backend else ''
        except Exception:  # noqa: BLE001 - status details must never break a write
            return ''

    def backend_info(self):
        """One dict per hosted backend, for ListBackends and the picker."""
        info = []
        for name, backend in sorted(self.backends.items()):
            configured, message = backend.is_configured()
            connected, connect_message = self._connection.get(name, (False, ''))
            try:
                details = backend.describe()
            except Exception as e:  # noqa: BLE001
                details = {'error': str(e)}
            info.append({
                'name': name,
                'capabilities': sorted(c.value for c in backend.capabilities),
                'configured': configured,
                'connected': connected,
                'message': connect_message or message,
                'event_driven': name in self._event_stops,
                'details': details,
            })
        return info

    # --- events ------------------------------------------------------------------

    def start_events(self, on_synced=None):
        """Subscribe to every backend that can push changes.

        Args:
            on_synced: Optional callable(backend_name, part_id, (ok, message))
                run after each event-driven re-read.

        Returns:
            list: names of the backends that are now event-driven.
        """
        for name, backend in self.backends.items():
            if Capability.EVENTS not in backend.capabilities or name in self._event_stops:
                continue

            def on_change(part_id, name=name):
                outcome = self.sync_part(name, part_id, recheck_geometry=True)
                if on_synced is not None:
                    on_synced(name, part_id, outcome)

            stop = backend.subscribe(on_change)
            if stop is not None:
                self._event_stops[name] = stop
        return sorted(self._event_stops)

    def stop_events(self):
        for stop in self._event_stops.values():
            try:
                stop()
            except Exception as e:  # noqa: BLE001
                logger.warning('Stopping an event subscription failed: %s', e)
        self._event_stops.clear()

    def close(self):
        """Stop event subscriptions and release every backend."""
        self.stop_events()
        for backend in self.backends.values():
            try:
                backend.close()
            except Exception as e:  # noqa: BLE001
                logger.warning('Closing backend %s failed: %s', backend.name, e)

    # --- reconciliation ---------------------------------------------------------

    def _passes_filters(self, remote):
        states = self.config.maturity_states()
        return not states or (remote.maturity or '').upper() in states

    def _reconcile(self, backend, remote, local, result):
        """Insert, update or touch one part from its remote copy."""
        if local is None:
            self.db.insert_part(remote, storage_key(backend.name, remote.part_id))
            self._refresh_artifacts(backend, remote)
            self.db.log_sync('part_added', backend.name, remote.part_id, remote.title)
            result.added += 1
        elif local.get('change_token') != remote.change_token:
            self.db.update_part(remote, sync_status='modified')
            self._invalidate_geometry(local)
            self._refresh_artifacts(backend, remote)
            self.db.log_sync('part_modified', backend.name, remote.part_id,
                             f"{local.get('change_token')} -> {remote.change_token}")
            result.modified += 1
        else:
            self.db.touch_part(backend.name, remote.part_id, sync_status='synced')
            result.unchanged += 1

    def _finish(self, result, started, action):
        result.completed_at = utc_now()
        # Duration must be known before summary() renders it into the message.
        result.duration_sec = time.monotonic() - started
        if not result.message:
            result.message = result.summary()
        self.last_results[result.backend] = result
        self.db.log_sync(action, result.backend, None, result.message)
        logger.info('%s %s: %s', result.backend, action, result.message)
        return result

    def _full_sync_one(self, backend):
        started = time.monotonic()
        result = SyncResult(backend=backend.name, started_at=utc_now())
        connected, message = self._connect(backend)
        if not connected:
            result.success, result.message, result.errors = False, message, 1
            return self._finish(result, started, 'full_sync')
        try:
            local_parts = self.db.get_all_parts(backend.name)
            seen = set()
            truncated = None
            try:
                for remote in backend.list_scope(backend.scope):
                    if not self._passes_filters(remote):
                        continue
                    seen.add(remote.part_id)
                    try:
                        self._reconcile(backend, remote, local_parts.get(remote.part_id),
                                        result)
                    except (BackendError, OSError) as e:
                        result.errors += 1
                        result.error_details.append(f'{remote.part_id}: {e}')
                        logger.warning('Sync error on %s %s: %s', backend.name,
                                       remote.part_id, e)
            except ScopeTruncated as e:
                truncated = e
            # Archiving means "no longer visible remotely". That inference is
            # only sound when the scope was enumerated exhaustively: parts
            # beyond a budget were never looked at, not removed.
            stale = [part_id for part_id, part in local_parts.items()
                     if part_id not in seen and part.get('sync_status') != 'archived']
            if stale and truncated is not None:
                logger.warning('%s; %d local part(s) were not seen this pass and are '
                               'left alone rather than archived.', truncated, len(stale))
                result.error_details.append(
                    f'Scan truncated at {truncated.budget} items; '
                    f'{len(stale)} part(s) left unreconciled.')
            elif stale:
                self.db.archive_parts(backend.name, stale)
                result.archived = len(stale)
        except BackendError as e:
            result.success = False
            result.errors += 1
            result.message = f'Full sync failed: {e}'
            result.error_details.append(str(e))
        return self._finish(result, started, 'full_sync')

    def _incremental_one(self, backend):
        started = time.monotonic()
        result = SyncResult(backend=backend.name, started_at=utc_now())
        connected, message = self._connect(backend)
        if not connected:
            result.success, result.message, result.errors = False, message, 1
            return self._finish(result, started, 'incremental')
        try:
            listing = backend.list_recent(backend.scope)
            if listing is None:
                return self._full_sync_one(backend)
            local_parts = self.db.get_all_parts(backend.name)
            # An incremental pass never archives: it saw only part of the scope.
            for remote in listing:
                if not self._passes_filters(remote):
                    continue
                try:
                    self._reconcile(backend, remote, local_parts.get(remote.part_id), result)
                except (BackendError, OSError) as e:
                    result.errors += 1
                    result.error_details.append(f'{remote.part_id}: {e}')
        except BackendError as e:
            result.success = False
            result.errors += 1
            result.message = f'Incremental sync failed: {e}'
            result.error_details.append(str(e))
        return self._finish(result, started, 'incremental')

    def _each_backend(self, name, run):
        names = [name] if name else sorted(self.backends)
        results = []
        self.syncing = True
        try:
            for backend_name in names:
                backend = self.backend(backend_name)
                if backend is None:
                    results.append(SyncResult(backend=backend_name, success=False, errors=1,
                                              message=f'Backend {backend_name!r} is not '
                                                      'hosted by this catalog.'))
                    continue
                results.append(run(backend))
        finally:
            self.syncing = False
        return results

    def run_full_sync(self, backend=None):
        """Reconcile every part in each backend's scope (or one backend's).

        New parts are inserted, parts whose change token moved are updated
        (and their cached STEP invalidated), unchanged parts are touched, and
        parts no longer reported are archived.

        Returns:
            list: one SyncResult per backend.
        """
        return self._each_backend(backend, self._full_sync_one)

    def run_incremental_sync(self, backend=None):
        """A cheap pass where a backend offers one, a full sync otherwise.

        Returns:
            list: one SyncResult per backend.
        """
        return self._each_backend(backend, self._incremental_one)

    def sync_part(self, backend_name, part_id, recheck_geometry=False):
        """Re-read one part and reconcile just that row, regardless of scope.

        Used for change events and for adding a part that sits outside the
        configured scope.

        Returns:
            tuple: (bool, str) success flag and message.
        """
        backend = self.backend(backend_name)
        if backend is None:
            return False, f'Backend {backend_name!r} is not hosted by this catalog.'
        connected, message = self._connect(backend)
        if not connected:
            return False, message
        try:
            remote = backend.get_part(part_id)
        except BackendError as e:
            return False, f'Could not fetch {part_id}: {e}'
        if remote is None:
            self.db.archive_parts(backend_name, [part_id])
            self.db.log_sync('part_archived', backend_name, part_id, 'not visible remotely')
            return True, f'{part_id} is no longer visible; archived.'
        local = self.db.get_part(backend_name, part_id)
        result = SyncResult(backend=backend_name)
        self._reconcile(backend, remote, local, result)
        if recheck_geometry and result.unchanged:
            self._refresh_geometry_availability(backend, remote)
        if result.added:
            return True, f'Added {remote.title}.'
        if result.modified:
            return True, f'Updated {remote.title}.'
        return True, f'{remote.title} is unchanged.'

    # --- per-part artifacts ---------------------------------------------------------

    def _refresh_geometry_availability(self, backend, remote):
        if Capability.GEOMETRY not in backend.capabilities:
            available = False
        elif remote.geometry_available is not None:
            available = remote.geometry_available
        else:
            available = backend.check_geometry(remote)
        self.db.set_geometry_available(backend.name, remote.part_id, available)
        return available

    def _refresh_artifacts(self, backend, remote):
        """Refresh a new or changed part's thumbnail and STEP availability."""
        if Capability.THUMBNAILS in backend.capabilities:
            self._fetch_remote_thumbnail(backend, remote)
        self._refresh_geometry_availability(backend, remote)

    def _fetch_remote_thumbnail(self, backend, remote):
        """Cache a real preview image published by the backend, if any.

        Backends without THUMBNAILS (or with nothing to offer) get a preview
        rendered locally from the STEP instead, once one is cached.
        """
        key = storage_key(backend.name, remote.part_id)
        try:
            path = backend.fetch_thumbnail(remote, self.paths.part_dir(key))
        except BackendError as e:
            logger.debug('Thumbnail unavailable for %s: %s', remote.part_id, e)
            return None
        if path is None:
            return None
        dest = self.paths.thumb_path(key)
        if Path(path) != dest:
            os.replace(path, dest)
        self.db.set_thumbnail(backend.name, remote.part_id, dest)
        return dest

    def _invalidate_geometry(self, local_part):
        """Delete a cached STEP whose part has changed remotely."""
        cached = self.resolved_geometry_path(local_part)
        for path in {cached, local_part.get('geometry_path')}:
            if not path:
                continue
            try:
                Path(path).unlink(missing_ok=True)
            except OSError as e:
                logger.warning('Could not remove stale STEP %s: %s', path, e)
        self.db.clear_geometry_cache(local_part['backend'], local_part['part_id'])

    def render_thumbnail(self, backend_name, part_id, force=False):
        """Render a preview from a part's cached STEP file.

        Returns:
            tuple: (Path or None, str) the rendered image and a message.
        """
        from viewpoint_generation.catalog.thumbnails import render_step_thumbnail

        part = self.db.get_part(backend_name, part_id)
        if part is None:
            return None, f'Unknown part: {backend_name}/{part_id}.'
        dest = self.paths.thumb_path(part['storage_key'])
        if dest.exists() and not force:
            return dest, 'Thumbnail already cached.'
        step_path = self.resolved_geometry_path(part)
        if step_path is None:
            return None, 'No cached STEP file to render a thumbnail from.'
        success, message = render_step_thumbnail(step_path, dest, units=self.config.mesh_units)
        if not success:
            return None, message
        self.db.set_thumbnail(backend_name, part_id, dest)
        return dest, message

    def _render_thumbnail_if_enabled(self, backend_name, part_id):
        """Best-effort preview after a STEP lands; never breaks the fetch."""
        if not self.config.render_thumbnails:
            return
        try:
            path, message = self.render_thumbnail(backend_name, part_id, force=True)
        except Exception as e:  # noqa: BLE001 - rendering must never break a fetch
            logger.debug('Thumbnail rendering failed for %s: %s', part_id, e)
            return
        if path is None:
            logger.debug('Thumbnail for %s: %s', part_id, message)

    def render_missing_thumbnails(self, force=False):
        """Render previews for every part that has a STEP but no thumbnail.

        Returns:
            tuple: (int, int) counts of rendered and skipped/failed parts.
        """
        rendered = failed = 0
        # Archived parts are included: one whose STEP is cached is still
        # selectable and still deserves a preview.
        for part in self.db.list_parts(include_archived=True):
            if self.resolved_geometry_path(part) is None:
                continue
            if self.resolved_thumb_path(part) is not None and not force:
                continue
            path, message = self.render_thumbnail(part['backend'], part['part_id'], force=force)
            if path is None:
                failed += 1
                logger.debug('Thumbnail for %s: %s', part['part_id'], message)
            else:
                rendered += 1
        return rendered, failed

    # --- cache path resolution ---------------------------------------------------

    def resolved_geometry_path(self, part):
        """The part's cached STEP as seen from this process, or None."""
        key = part.get('storage_key') or storage_key(part['backend'], part['part_id'])
        token = part.get('geometry_token')
        return self.paths.resolve_cached(
            part.get('geometry_path'),
            self.paths.geometry_path(key, token) if token else None,
            self.paths.legacy_step_path(part['part_id'], part.get('revision')))

    def resolved_thumb_path(self, part):
        """The part's cached thumbnail as seen from this process, or None."""
        key = part.get('storage_key') or storage_key(part['backend'], part['part_id'])
        return self.paths.resolve_cached(
            part.get('thumbnail_path'), self.paths.thumb_path(key),
            self.paths.legacy_thumb_path(part['part_id']))

    def resolved_plan_path(self, plan):
        """A recorded plan's file as seen from this process, or None."""
        stored = plan.get('file_path')
        if not stored:
            return None
        name = Path(stored).name
        legacy = self.paths.legacy_plans_for(plan['part_id'])
        return self.paths.resolve_cached(
            stored, self.paths.plans_for(storage_key(plan['backend'], plan['part_id'])) / name,
            legacy / name if legacy is not None else None)

    # --- geometry --------------------------------------------------------------------

    def _hand_placed_geometry(self, part):
        """A STEP an operator put in the cache by hand, if there is one."""
        legacy = self.paths.legacy_step_path(part['part_id'], part.get('revision'))
        if legacy is not None and legacy.exists():
            return legacy
        directory = self.paths.geometry_dir(part['storage_key'])
        if directory.exists():
            candidates = [p for p in directory.iterdir()
                          if p.is_file() and p.suffix.lower() in STEP_SUFFIXES]
            if candidates:
                return max(candidates, key=lambda p: p.stat().st_mtime)
        return None

    def fetch_geometry(self, backend_name, part_id):
        """Ensure a part's STEP file is present locally, downloading if needed.

        A cached file fetched at the part's current change token is returned
        untouched. Otherwise the backend is asked for one (when it declares
        GEOMETRY). A STEP placed in the part's geometry directory by hand --
        or at the legacy `steps/{id}/{revision}.stp` -- is honoured too, which
        keeps the pipeline usable for parts the PLM publishes no STEP for.

        Returns:
            tuple: (Path or None, str) the local STEP path and a message.
        """
        part = self.db.get_part(backend_name, part_id)
        if part is None:
            return None, f'Unknown part: {backend_name}/{part_id}.'

        cached = self.resolved_geometry_path(part)
        if cached is not None and part.get('geometry_token') == part.get('change_token'):
            if str(cached) != (part.get('geometry_path') or ''):
                # The database was written from a different mount point.
                self.db.set_geometry_cache(backend_name, part_id, cached,
                                           part.get('geometry_token'))
            return cached, 'STEP already cached.'

        key = part['storage_key']
        dest = self.paths.geometry_path(key, part.get('change_token'))
        backend = self.backend(backend_name)
        if backend is not None and Capability.GEOMETRY in backend.capabilities:
            connected, message = self._connect(backend)
            path = None
            if connected:
                try:
                    path = backend.fetch_geometry(part_from_row(part), dest.parent)
                except BackendError as e:
                    logger.warning('STEP fetch failed for %s: %s', part_id, e)
            else:
                logger.warning('STEP fetch for %s skipped: %s', part_id, message)
            if path is not None:
                if Path(path) != dest:
                    os.replace(path, dest)
                self.db.set_geometry_cache(backend_name, part_id, dest, part.get('change_token'))
                self.db.log_sync('geometry_fetched', backend_name, part_id, str(dest))
                self._render_thumbnail_if_enabled(backend_name, part_id)
                return dest, f'Downloaded STEP to {dest}.'

        hand_placed = self._hand_placed_geometry(part)
        if hand_placed is not None:
            self.db.set_geometry_cache(backend_name, part_id, hand_placed,
                                       part.get('change_token'))
            self.db.log_sync('geometry_adopted', backend_name, part_id, str(hand_placed))
            self._render_thumbnail_if_enabled(backend_name, part_id)
            return hand_placed, f'Using locally provided STEP at {hand_placed}.'
        if cached is not None:
            return cached, ('Using cached STEP from an earlier revision; '
                            'the remote part has changed since it was downloaded.')
        return None, (f'No STEP available for this part. Its backend ({backend_name}) '
                      'publishes none -- export one and place it in '
                      f'{self.paths.geometry_dir(key)}/.')

    def prefetch_all_geometry(self):
        """Download every advertised-but-missing STEP file.

        Returns:
            tuple: (int, int) counts of fetched and failed parts.
        """
        fetched = failed = 0
        for part in self.db.list_parts():
            if part.get('geometry_available') and self.resolved_geometry_path(part) is None:
                path, _ = self.fetch_geometry(part['backend'], part['part_id'])
                if path is None:
                    failed += 1
                else:
                    fetched += 1
        return fetched, failed

    # --- selection --------------------------------------------------------------------

    def plm_context_json(self, part):
        """The PLM context published with a selection of `part` (see plan_contract)."""
        return plan_contract.context_json(
            part, cell_id=self.config.cell_id, generated_by=self._account(part['backend']))

    def plan_dir_for(self, part):
        """Directory the producer writes a part's new plan files into."""
        return self.paths.plans_for(part['storage_key'])

    def select(self, backend_name, part_id, plan_id=None, model_only=False,
               skip_plan_check=False):
        """Everything a part selection needs: geometry, plan, and plan contract.

        Shared by the `select_part` service and the picker so both announce
        exactly the same thing.

        Args:
            plan_id: Load this recorded plan rather than the current one. An
                explicit choice comes back SELECTED when it targets an earlier
                change token.
            model_only: Load the model with no plan.
            skip_plan_check: Resolve the geometry only.

        Returns:
            tuple: (dict or None, str) the selection and a message. The dict
            holds `part`, `geometry_path`, `mesh_units`, `plan_status`,
            `plan_path`, `plan_dir` and `plm_context_json`.
        """
        part = self.db.get_part(backend_name, part_id)
        if part is None:
            return None, f'Unknown part: {backend_name}/{part_id}.'
        geometry, message = self.fetch_geometry(backend_name, part_id)
        if geometry is None:
            return None, message
        if plan_id:
            plan_status, plan_path = self.select_plan(backend_name, part_id, plan_id)
            if plan_status == PlanStatus.NONE:
                return None, f'Plan is unavailable: {plan_id}.'
        elif model_only or skip_plan_check:
            plan_status, plan_path = PlanStatus.NONE, ''
        else:
            plan_status, plan_path = self.ensure_plan(backend_name, part_id)
        part = self.db.get_part(backend_name, part_id)
        return {
            'part': part,
            'geometry_path': str(geometry),
            'mesh_units': self.config.mesh_units,
            'plan_status': plan_status,
            'plan_path': plan_path or '',
            'plan_dir': str(self.plan_dir_for(part)),
            'plm_context_json': self.plm_context_json(part),
        }, message

    # --- plans ------------------------------------------------------------------------

    def ensure_plan(self, backend_name, part_id, check_remote=True):
        """Resolve the inspection plan for a part, downloading one if possible.

        Checks the local catalog first. When the backend can find attachments
        from the part (LINK_ATTACHMENT), another cell may have uploaded a plan
        for this exact change token, and reusing it beats re-planning.

        Args:
            check_remote: Ask the backend when no current local plan exists.
                List views pass False: a round trip per row is unaffordable.

        Returns:
            tuple: (PlanStatus, str) the plan's validity and its file path.
        """
        part = self.db.get_part(backend_name, part_id)
        if part is None:
            return PlanStatus.NONE, ''
        local_plan = self.db.get_current_plan(backend_name, part_id)
        local_path = self.resolved_plan_path(local_plan) if local_plan else None
        if local_path is not None and local_plan.get('change_token') == part.get('change_token'):
            return PlanStatus.CURRENT, str(local_path)
        if check_remote and self.has(backend_name, Capability.LINK_ATTACHMENT):
            remote_status, remote_path = self._try_download_remote_plan(part)
            if remote_status is not None:
                return remote_status, remote_path
        if local_path is not None:
            return PlanStatus.STALE, str(local_path)
        return PlanStatus.NONE, ''

    def _try_download_remote_plan(self, part):
        """Look for a remote plan matching the part's current change token.

        A downloaded plan has no producer to describe it, so its token, stage
        and generation time are read back from the `plm_context` block this
        catalog authored (`plan_contract.read_context`), nothing else.

        Returns:
            tuple: (PlanStatus or None, str) None when nothing usable was found.
        """
        backend = self.backend(part['backend'])
        connected, _ = self._connect(backend)
        if not connected:
            return None, ''
        remote = part_from_row(part)
        try:
            refs = backend.list_attachments(remote, 'plan')
        except BackendError as e:
            logger.debug('Remote plan lookup failed for %s: %s', part['part_id'], e)
            return None, ''
        for ref in sorted(refs, key=lambda r: r.modified or '', reverse=True):
            try:
                path = backend.fetch_attachment(ref, self.plan_dir_for(part))
            except BackendError as e:
                logger.debug('Plan %s not downloadable: %s', ref.ref_id, e)
                continue
            if path is None:
                continue
            context = plan_contract.read_context(path)
            token = plan_contract.token_of(context)
            had_local = self.db.get_current_plan(part['backend'], part['part_id']) is not None
            self.db.upsert_plan({
                'plan_id': f"{part['part_id']}:{ref.ref_id}",
                'backend': part['backend'], 'part_id': part['part_id'],
                'remote_ref': ref.ref_id, 'change_token': token, 'file_path': str(path),
                'stage': context.get('stage'), 'stage_rank': context.get('stage_rank', -1),
                'summary_text': ref.title,
                'generated_at': context.get('generated_at') or utc_now(),
                'uploaded_at': context.get('generated_at'),
                'upload_status': 'synced', 'source': 'remote',
            })
            self.db.log_sync('plan_downloaded', part['backend'], part['part_id'], ref.ref_id)
            if token and token != part.get('change_token'):
                # A remote plan exists but targets different geometry.
                return PlanStatus.STALE, str(path)
            return (PlanStatus.UPDATED_FROM_REMOTE if had_local
                    else PlanStatus.DOWNLOADED), str(path)
        return None, ''

    def record_plan(self, backend_name, part_id, plan_path, stage, stage_rank=-1,
                    summary_json='', summary_text='', change_token='', generated_at='',
                    upload=None, make_current=True):
        """Register a plan file the producer wrote, as the producer describes it.

        Every plan the pipeline writes is recorded, so the picker can offer a
        part's full history. The file is recorded where it lives and is never
        opened (see `plan_contract`).

        Args:
            stage: Producer-defined stage label (required).
            stage_rank: Producer-defined order; higher is further along.
            summary_json: Producer summary, stored verbatim.
            summary_text: One-line summary shown in the picker.
            change_token: The token the plan was generated against; empty
                means the part's current one.
            generated_at: When the plan was generated; empty means now.
            upload: Force upload on/off; None consults auto_upload_plan_stages.
            make_current: Make this the part's current plan.

        Returns:
            tuple: (str or None, str) the plan_id recorded and a message.
        """
        part = self.db.get_part(backend_name, part_id)
        if part is None:
            return None, f'Unknown part: {backend_name}/{part_id}.'
        if not stage:
            return None, 'A plan stage is required to record a plan.'
        plan_path = Path(plan_path)
        if not plan_path.is_file():
            return None, f'Plan file not found: {plan_path}.'

        plan_id = f'{part_id}:{plan_path.stem}'
        self.db.upsert_plan({
            'plan_id': plan_id, 'backend': backend_name, 'part_id': part_id,
            'change_token': change_token or part.get('change_token'),
            'file_path': str(plan_path), 'stage': stage, 'stage_rank': stage_rank,
            'summary_json': summary_json or '{}', 'summary_text': summary_text or '',
            'generated_at': generated_at or utc_now(), 'upload_status': 'local',
            'source': 'local', 'is_current': make_current,
        })
        described = f' ({summary_text})' if summary_text else ''
        self.db.log_sync('plan_recorded', backend_name, part_id, f'{stage}{described}')

        if upload is None:
            upload = stage.lower() in self.config.auto_upload_stages()
        if not upload:
            return plan_id, f'Recorded {stage} plan{described}.'
        _ref, upload_message = self.upload_recorded_plan(plan_id)
        return plan_id, f'Recorded {stage} plan{described}. {upload_message}'

    def _title(self, kind, part, suffix):
        """Title of an uploaded plan or result, as the cell has always named them."""
        label = 'PLAN' if kind == 'plan' else 'RESULT'
        return (f"{self.config.title_prefix}{label}_"
                f"{part.get('part_number') or part['part_id']}_"
                f"{part.get('revision') or 'NA'}_{suffix}")

    def upload_recorded_plan(self, plan_id):
        """Upload an already-recorded plan through its part's backend.

        Returns:
            tuple: (remote ref id or None, str) and a message.
        """
        plan = self.db.get_plan(plan_id)
        if plan is None:
            return None, f'Unknown plan: {plan_id}.'
        part = self.db.get_part(plan['backend'], plan['part_id'])
        if part is None:
            return None, f"Unknown part: {plan['backend']}/{plan['part_id']}."
        path = self.resolved_plan_path(plan)
        if path is None:
            return None, f"Plan file is missing: {plan.get('file_path')}."
        backend = self.backend(plan['backend'])
        if backend is None:
            return None, f"Backend {plan['backend']!r} is not hosted here; not uploaded."
        if Capability.ATTACH not in backend.capabilities:
            self.db.update_plan(plan_id, upload_status='local')
            return None, (f'{backend.name} does not store files remotely; '
                          'the plan stays on this cell.')
        configured, message = backend.is_configured()
        if not configured:
            self.db.update_plan(plan_id, upload_status='local')
            return None, f'Not uploaded: {message}'

        self.db.update_plan(plan_id, upload_status='uploading')
        title = self._title('plan', part, (plan.get('stage') or 'plan').upper())
        ref, message = None, ''
        connected, connect_message = self._connect(backend)
        if not connected:
            message = f'Plan upload failed: {connect_message}'
        else:
            try:
                ref = backend.attach_file(part_from_row(part), path, 'plan', title)
                if ref is None:
                    message = f'{backend.name} could not store the plan.'
            except BackendError as e:
                message = f'Plan upload failed: {e}'
        if ref is None:
            self.db.update_plan(plan_id, upload_status='failed')
            self.db.log_sync('plan_upload_failed', part['backend'], part['part_id'], message)
            return None, message
        self.db.update_plan(plan_id, remote_ref=ref.ref_id, uploaded_at=utc_now(),
                            upload_status='synced')
        self.db.log_sync('plan_uploaded', part['backend'], part['part_id'], ref.ref_id)
        return ref.ref_id, ref.note or f'Uploaded plan as {ref.ref_id}.'

    def list_plans(self, backend_name, part_id):
        """Every plan recorded for a part, newest first, enriched for display.

        Adds whether the file is present in *this* environment and whether
        the plan still matches the part's current change token.
        """
        part = self.db.get_part(backend_name, part_id) or {}
        plans = []
        for plan in self.db.list_plans(backend_name, part_id):
            path = self.resolved_plan_path(plan)
            enriched = dict(plan)
            enriched['file_path'] = str(path) if path else plan.get('file_path')
            enriched['available'] = path is not None
            enriched['stale'] = bool(part.get('change_token')
                                     and plan.get('change_token') != part['change_token'])
            enriched['is_current'] = bool(plan.get('is_current'))
            enriched['stage'] = plan.get('stage') or 'unknown'
            try:
                enriched['summary'] = json.loads(plan.get('summary_json') or '{}')
            except ValueError:
                enriched['summary'] = {}
            enriched['summary_text'] = plan.get('summary_text') or ''
            plans.append(enriched)
        return plans

    def select_plan(self, backend_name, part_id, plan_id):
        """Make one recorded plan the part's current plan and resolve its path.

        An explicit operator choice overrides the staleness guard: the status
        comes back SELECTED, not STALE, and consumers load it with a warning.
        """
        plan = self.db.get_plan(plan_id)
        if plan is None or (plan['backend'], plan['part_id']) != (backend_name, part_id):
            return PlanStatus.NONE, ''
        path = self.resolved_plan_path(plan)
        if path is None:
            return PlanStatus.NONE, ''
        self.db.set_current_plan(plan_id)
        part = self.db.get_part(backend_name, part_id) or {}
        if plan.get('change_token') == part.get('change_token'):
            return PlanStatus.CURRENT, str(path)
        return PlanStatus.SELECTED, str(path)

    def orphan_plan_candidates(self, backend_name, part_id):
        """Plan files on disk the catalog should ask the producer about.

        Files in the part's plan directories (new and legacy layout) and in
        the `_results` directories beside its geometry that no row records,
        plus recorded files whose row has no stage yet. Only names are
        listed; no file is opened.
        """
        part = self.db.get_part(backend_name, part_id)
        if part is None:
            return []
        known = {Path(plan['file_path']).name: plan
                 for plan in self.db.list_plans(backend_name, part_id)}
        candidates = []
        for plan in known.values():
            if not plan.get('stage'):
                path = self.resolved_plan_path(plan)
                if path is not None:
                    candidates.append(path)

        directories = [self.plan_dir_for(part), self.paths.legacy_plans_for(part_id)]
        directories += self.paths.legacy_results_dirs(part_id)
        geometry_dir = self.paths.geometry_dir(part['storage_key'])
        if geometry_dir.exists():
            directories += sorted(geometry_dir.glob('*_results'))
        seen = set()
        for directory in directories:
            if directory is None or not Path(directory).exists():
                continue
            for path in sorted(Path(directory).glob('*.json')):
                if path.name not in known and path.name not in seen:
                    seen.add(path.name)
                    candidates.append(path)
        return candidates

    def adopt_orphan_plans(self, backend_name, part_id, describe):
        """Record plan files on disk that the database never captured.

        The catalog lists candidates; the producer describes them (the
        `DescribePlans` contract), so no plan file is opened here.

        Args:
            describe: Callable taking a list of paths and returning one dict
                per path with `path`, `valid`, `stage`, `stage_rank`,
                `summary_json`, `summary_text`, `change_token`, `generated_at`.

        Returns:
            tuple: (int, int) counts of plans adopted and files skipped.
        """
        candidates = self.orphan_plan_candidates(backend_name, part_id)
        if not candidates:
            return 0, 0
        known = {plan['plan_id'] for plan in self.db.list_plans(backend_name, part_id)}
        adopted = skipped = 0
        for description in describe([str(path) for path in candidates]):
            if not description.get('valid'):
                skipped += 1
                continue
            path = Path(description['path'])
            plan_id = f'{part_id}:{path.stem}'
            if plan_id in known:
                # A recorded row missing its stage: backfill what the producer
                # said, leaving everything else about the row alone.
                self.db.update_plan(
                    plan_id, stage=description.get('stage'),
                    stage_rank=description.get('stage_rank', -1),
                    summary_json=description.get('summary_json') or '{}',
                    summary_text=description.get('summary_text') or '')
                continue
            recorded, _message = self.record_plan(
                backend_name, part_id, path, stage=description.get('stage'),
                stage_rank=description.get('stage_rank', -1),
                summary_json=description.get('summary_json', ''),
                summary_text=description.get('summary_text', ''),
                change_token=description.get('change_token', ''),
                generated_at=description.get('generated_at', ''),
                upload=False, make_current=False)
            if recorded is None:
                skipped += 1
            else:
                adopted += 1
        # Hand the current flag to the genuinely newest plan, breaking ties by
        # how far through the pipeline it got, since a whole run's stages
        # routinely share a timestamp.
        if adopted:
            plans = self.db.list_plans(backend_name, part_id)
            newest = max(plans, key=lambda p: (p.get('generated_at') or '',
                                               p.get('stage_rank', -1)))
            self.db.set_current_plan(newest['plan_id'])
        return adopted, skipped

    # --- status -------------------------------------------------------------------------

    def status(self):
        """Catalog and sync status for the picker and ROS services."""
        return {
            'counts': self.db.counts(),
            'syncing': self.syncing,
            'last_sync': self.db.last_sync(),
            'last_results': {name: result.to_dict()
                             for name, result in self.last_results.items()},
            'backends': self.backend_info(),
        }
