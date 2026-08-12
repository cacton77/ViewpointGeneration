"""Catalog synchronization between a 3DEXPERIENCE tenant and the local cell.

`CatalogSync` keeps the SQLite catalog aligned with the engineering items
visible on the tenant, and lazily caches the heavyweight artifacts (STEP files,
thumbnails) that inspection actually needs.

Change detection is by **cestamp** -- the optimistic-concurrency stamp 3DX
mutates on every modification of an object. Comparing the remote cestamp
against the stored one is cheaper and far more reliable than comparing
timestamps, and it is the same stamp used to decide whether a cached STEP file
or a generated inspection plan still corresponds to the CAD it came from.

Run a sync from the command line with:

    python -m viewpoint_generation.catalog.sync --full
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from viewpoint_generation.catalog.client import DXAPIError, DXAuthError, DXClient
from viewpoint_generation.catalog.config import CatalogPaths, DXConfig, SyncConfig
from viewpoint_generation.catalog.schema import CatalogDB, utc_now

logger = logging.getLogger(__name__)


def _software_version():
    """Short git revision of the ViewpointGeneration checkout, for traceability.

    Stamped into uploaded envelopes so a plan can be traced back to the code
    that produced it. Returns 'unknown' outside a git checkout.
    """
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=str(Path(__file__).resolve().parent),
            stderr=subprocess.DEVNULL, text=True, timeout=5).strip()
    except (subprocess.SubprocessError, OSError):
        return 'unknown'


def _file_kind(path):
    """Classify a result-bundle file for the manifest's file list."""
    suffix = path.suffix.lower()
    if suffix in ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'):
        return 'image'
    if suffix in ('.ply', '.pcd', '.xyz', '.las'):
        return 'point_cloud'
    if suffix == '.json':
        return 'metadata'
    return 'other'


class PlanStatus(str, Enum):
    """Validity of the inspection plan held for a part.

    A plan is bound to the CAD revision it was generated against, so a plan
    whose cestamp no longer matches the part's is STALE: the geometry may have
    moved under it, and the operator decides whether to re-plan.
    """

    CURRENT = 'CURRENT'
    DOWNLOADED = 'DOWNLOADED'
    UPDATED_FROM_REMOTE = 'UPDATED_FROM_REMOTE'
    STALE = 'STALE'
    NONE = 'NONE'


@dataclass
class SyncResult:
    """Outcome of one synchronization pass."""

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


class CatalogSync:
    """Synchronizes the local parts catalog with a 3DEXPERIENCE tenant.

    Args:
        client: An authenticated (or authenticatable) `DXClient`.
        db_path: Path to the catalog SQLite database.
        config: A `SyncConfig` describing cadence, scope, and filters.
        paths: A `CatalogPaths` for the on-disk caches. Defaults to the
            environment-derived paths.
    """

    def __init__(self, client, db_path, config, paths=None):
        self.client = client
        self.config = config or SyncConfig()
        self.paths = paths or CatalogPaths.from_env()
        # Directory creation on first run, before the DB file is opened.
        success, message = self.paths.ensure()
        if not success:
            raise RuntimeError(message)
        self.db = CatalogDB(db_path)
        self.last_result = None
        self.syncing = False

    # --- scope and filtering ---------------------------------------------

    def _in_scope(self, item):
        """True when a remote item passes the configured maturity/space filters."""
        states = self.config.maturity_states()
        if states and (item.get('maturity') or '').upper() not in states:
            return False
        space_filter = self.config.collab_space_filter
        if space_filter and (item.get('collab_space') or '') != space_filter:
            return False
        return True

    def _fetch_remote_items(self):
        """Page through the tenant's engineering items within the configured scope.

        The dseng search's `totalItems` reports the size of the page it just
        returned, NOT the size of the whole result set -- asking for 200 items
        answers `totalItems: 200`. It is therefore useless as a paging bound,
        and the loop instead advances until a short page arrives (or the
        max_items scan budget is spent).

        Returns:
            tuple: (items, scanned) of normalized in-scope items and the number
            of remote items examined.
        """
        collected = {}
        skip = 0
        scanned = 0
        while scanned < self.config.max_items:
            page_size = min(self.config.page_size, self.config.max_items - scanned)
            items, _ = self.client.search_eng_items(
                query=self.config.bookmark_scope, top=page_size, skip=skip)
            if not items:
                break
            for item in items:
                if item.get('eng_item_id') and self._in_scope(item):
                    collected[item['eng_item_id']] = item
            scanned += len(items)
            skip += len(items)
            if len(items) < page_size:
                break
        return list(collected.values()), scanned

    # --- sync passes -------------------------------------------------------

    def run_full_sync(self):
        """Enumerate every in-scope remote item and reconcile the local catalog.

        New items are inserted, items whose cestamp changed are updated (and
        their cached STEP invalidated), unchanged items are touched, and local
        items that are no longer visible remotely are archived.

        Returns:
            SyncResult
        """
        started = time.monotonic()
        result = SyncResult(started_at=utc_now())
        self.syncing = True
        try:
            success, message = self.client.ensure_login()
            if not success:
                result.success = False
                result.message = message
                result.errors = 1
                return result

            remote_items, _ = self._fetch_remote_items()
            local_parts = self.db.get_all_parts()
            seen = set()

            for item in remote_items:
                item_id = item['eng_item_id']
                seen.add(item_id)
                local = local_parts.get(item_id)
                try:
                    if local is None:
                        self.db.insert_part(item)
                        self._refresh_artifacts(item, new_item=True)
                        self.db.log_sync('part_added', item_id, item.get('title'))
                        result.added += 1
                    elif local.get('cestamp') != item.get('cestamp'):
                        self.db.update_part(item, sync_status='modified')
                        self._invalidate_step_cache(local)
                        self._refresh_artifacts(item, new_item=False)
                        self.db.log_sync(
                            'part_modified', item_id,
                            f"cestamp {local.get('cestamp')} -> {item.get('cestamp')}")
                        result.modified += 1
                    else:
                        self.db.touch_part(item_id, sync_status='synced')
                        result.unchanged += 1
                except (DXAPIError, OSError) as e:
                    result.errors += 1
                    result.error_details.append(f'{item_id}: {e}')
                    logger.warning('Sync error on item %s: %s', item_id, e)

            stale = [item_id for item_id, part in local_parts.items()
                     if item_id not in seen and part.get('sync_status') != 'archived']
            if stale:
                self.db.archive_parts(stale)
                result.archived = len(stale)
        except (DXAuthError, DXAPIError) as e:
            result.success = False
            result.errors += 1
            result.message = f'Full sync failed: {e}'
            result.error_details.append(str(e))
            logger.error(result.message)
        finally:
            self.syncing = False
            result.completed_at = utc_now()
            # Duration must be known before summary() renders it into the
            # message, so the timing is computed here rather than in the body.
            result.duration_sec = time.monotonic() - started
            if not result.message:
                result.message = result.summary()
            self.last_result = result
            self.db.log_sync('full_sync', None, result.message)
        logger.info('Full sync: %s', result.message)
        return result

    def run_incremental_sync(self):
        """Lightweight check for changes since the last pass.

        The tenant's dseng search exposes no `$modifiedAfter` filter, so this
        scans a single page and skips the archival step -- an incremental pass
        never removes anything, it only picks up additions and cestamp changes.
        A full page means there are probably more items behind it, so the pass
        escalates to `run_full_sync()` rather than silently covering a prefix
        of the catalog.

        Returns:
            SyncResult
        """
        started = time.monotonic()
        result = SyncResult(started_at=utc_now())
        self.syncing = True
        try:
            success, message = self.client.ensure_login()
            if not success:
                result.success = False
                result.message = message
                result.errors = 1
                return result

            items, _ = self.client.search_eng_items(
                query=self.config.bookmark_scope, top=self.config.page_size)
            if len(items) >= self.config.page_size:
                logger.info('Incremental sync filled its %d-item page; '
                            'escalating to a full sync.', self.config.page_size)
                self.syncing = False
                return self.run_full_sync()

            local_parts = self.db.get_all_parts()
            for item in items:
                if not item.get('eng_item_id') or not self._in_scope(item):
                    continue
                item_id = item['eng_item_id']
                local = local_parts.get(item_id)
                try:
                    if local is None:
                        self.db.insert_part(item)
                        self._refresh_artifacts(item, new_item=True)
                        result.added += 1
                    elif local.get('cestamp') != item.get('cestamp'):
                        self.db.update_part(item, sync_status='modified')
                        self._invalidate_step_cache(local)
                        self._refresh_artifacts(item, new_item=False)
                        result.modified += 1
                    else:
                        self.db.touch_part(item_id, sync_status='synced')
                        result.unchanged += 1
                except (DXAPIError, OSError) as e:
                    result.errors += 1
                    result.error_details.append(f'{item_id}: {e}')
        except (DXAuthError, DXAPIError) as e:
            result.success = False
            result.errors += 1
            result.message = f'Incremental sync failed: {e}'
            result.error_details.append(str(e))
            logger.error(result.message)
        finally:
            self.syncing = False
            result.completed_at = utc_now()
            result.duration_sec = time.monotonic() - started
            if not result.message:
                result.message = result.summary()
            self.last_result = result
            self.db.log_sync('incremental', None, result.message)
        logger.info('Incremental sync: %s', result.message)
        return result

    def sync_single_item(self, eng_item_id):
        """Re-read one item from 3DX and reconcile just that row.

        Used by the JMS event listener, which learns about individual object
        changes rather than scanning the whole scope.

        Returns:
            tuple: (bool, str) success flag and message.
        """
        try:
            success, message = self.client.ensure_login()
            if not success:
                return False, message
            item = self.client.get_eng_item(eng_item_id)
        except (DXAuthError, DXAPIError) as e:
            return False, f'Could not fetch item {eng_item_id}: {e}'

        if item is None:
            self.db.archive_parts([eng_item_id])
            self.db.log_sync('part_archived', eng_item_id, 'not visible remotely')
            return True, f'Item {eng_item_id} is no longer visible; archived.'

        local = self.db.get_part(eng_item_id)
        if local is None:
            self.db.insert_part(item)
            self._refresh_artifacts(item, new_item=True)
            self.db.log_sync('part_added', eng_item_id, item.get('title'))
            return True, f"Added {item.get('title')}."
        if local.get('cestamp') != item.get('cestamp'):
            self.db.update_part(item, sync_status='modified')
            self._invalidate_step_cache(local)
            self._refresh_artifacts(item, new_item=False)
            self.db.log_sync('part_modified', eng_item_id, 'cestamp changed')
            return True, f"Updated {item.get('title')}."
        self.db.touch_part(eng_item_id, sync_status='synced')
        return True, f"{item.get('title')} is unchanged."

    # --- per-item artifacts -------------------------------------------------

    def _refresh_artifacts(self, item, new_item):
        """Refresh the thumbnail and STEP availability flag for one item."""
        item_id = item['eng_item_id']
        if self.config.fetch_thumbnails:
            self._fetch_thumbnail(item_id)
        self._check_step_availability(item_id)

    def _fetch_thumbnail(self, eng_item_id):
        """Download and cache an item's thumbnail.

        Returns:
            Path: The cached image, or None when the item publishes no image.
        """
        try:
            data, source_url = self.client.get_thumbnail(eng_item_id)
        except DXAPIError as e:
            logger.debug('Thumbnail unavailable for %s: %s', eng_item_id, e)
            return None
        if not data:
            return None
        dest = self.paths.thumb_path(eng_item_id)
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
        except OSError as e:
            logger.warning('Could not write thumbnail for %s: %s', eng_item_id, e)
            return None
        self.db.set_thumbnail(eng_item_id, dest)
        return dest

    def _check_step_availability(self, eng_item_id):
        """Record whether the tenant reports a STEP derived output for an item."""
        try:
            available = self.client.find_step_output(eng_item_id) is not None
        except (DXAPIError, DXAuthError) as e:
            logger.debug('Derived-output check failed for %s: %s', eng_item_id, e)
            available = False
        self.db.set_step_available(eng_item_id, available)
        return available

    def _invalidate_step_cache(self, local_part):
        """Delete a cached STEP file whose CAD revision has moved on."""
        step_path = local_part.get('step_path')
        if not step_path:
            return
        try:
            Path(step_path).unlink(missing_ok=True)
        except OSError as e:
            logger.warning('Could not remove stale STEP %s: %s', step_path, e)
        self.db.clear_step_cache(local_part['eng_item_id'])

    # --- cache path resolution ---------------------------------------------

    def resolved_step_path(self, part):
        """The part's cached STEP file as seen from this process, or None.

        Paths recorded by a sync run on the host do not resolve inside the
        container (and vice versa), so the cache location is recomputed for
        the current environment.
        """
        return self.paths.resolve_cached(
            part.get('step_path'),
            self.paths.step_path(part['eng_item_id'], part.get('revision')))

    def resolved_thumb_path(self, part):
        """The part's cached thumbnail as seen from this process, or None."""
        return self.paths.resolve_cached(
            part.get('thumbnail_path'),
            self.paths.thumb_path(part['eng_item_id']))

    # --- STEP files ---------------------------------------------------------

    def fetch_step(self, eng_item_id):
        """Ensure a part's STEP file is present locally, downloading if needed.

        Called when the operator selects a part. A cached file whose
        `step_cestamp` still matches the part's current cestamp is returned
        untouched; otherwise the STEP derived output is downloaded through an
        FCS ticket.

        A STEP dropped into `catalog/steps/{eng_item_id}/{revision}.stp` by
        hand is honoured too, which is what makes the pipeline usable on
        tenants that publish no derived outputs.

        Returns:
            tuple: (Path or None, str) the local STEP path and a message.
        """
        part = self.db.get_part(eng_item_id)
        if part is None:
            return None, f'Unknown part: {eng_item_id}.'

        cached = self.resolved_step_path(part)
        if cached is not None and part.get('step_cestamp') == part.get('cestamp'):
            if str(cached) != (part.get('step_path') or ''):
                # The database was written from a different mount point;
                # record the path that is valid here.
                self.db.set_step_cache(eng_item_id, cached, part.get('step_cestamp'))
            return cached, 'STEP already cached.'

        dest = self.paths.step_path(eng_item_id, part.get('revision'))

        try:
            success, message = self.client.ensure_login()
            if not success:
                raise DXAuthError(message)
            output = self.client.find_step_output(eng_item_id)
            if output is not None:
                ticket = self.client.get_download_ticket(
                    output.get('id') or eng_item_id)
                if ticket.get('url'):
                    ok, message = self.client.download_file(ticket['url'], dest)
                    if ok:
                        self.db.set_step_cache(eng_item_id, dest, part.get('cestamp'))
                        self.db.log_sync('step_fetched', eng_item_id, str(dest))
                        return dest, f'Downloaded STEP to {dest}.'
                    logger.warning('STEP download failed for %s: %s',
                                   eng_item_id, message)
        except (DXAuthError, DXAPIError) as e:
            logger.warning('STEP fetch failed for %s: %s', eng_item_id, e)

        # No derived output was retrievable. A hand-placed file at the cache
        # location is still a perfectly good STEP -- adopt it rather than
        # forcing the operator to re-export.
        if dest.exists():
            self.db.set_step_cache(eng_item_id, dest, part.get('cestamp'))
            self.db.log_sync('step_adopted', eng_item_id, str(dest))
            return dest, f'Using locally provided STEP at {dest}.'
        if cached is not None:
            return cached, ('Using cached STEP from an earlier revision; '
                            'the remote CAD has changed since it was downloaded.')

        return None, ('No STEP available for this part. The tenant publishes no '
                      'STEP derived output for it -- export one from 3DX and place '
                      f'it at {dest}.')

    def prefetch_all_steps(self):
        """Download every advertised-but-missing STEP file.

        Returns:
            tuple: (int, int) counts of fetched and failed items.
        """
        fetched = failed = 0
        for part in self.db.list_parts():
            if part.get('step_available') and not part.get('step_path'):
                path, _ = self.fetch_step(part['eng_item_id'])
                if path is None:
                    failed += 1
                else:
                    fetched += 1
        return fetched, failed

    # --- plan envelopes -----------------------------------------------------

    def _build_envelope(self, part, results_json_path, envelope_type='inspection_plan'):
        """Wrap a ViewpointGeneration results JSON in a PLM envelope.

        The results JSON is embedded verbatim under `plan`, so the existing
        consumers (viewpoint_traversal, gui, task_planning) keep reading the
        exact structure they always have. The envelope only adds the PLM
        context needed to tie the plan to a CAD revision.

        Returns:
            tuple: (dict or None, str) the envelope and a message.
        """
        results_json_path = Path(results_json_path)
        try:
            with open(results_json_path) as handle:
                plan_payload = json.load(handle)
        except (OSError, ValueError) as e:
            return None, f'Could not read results JSON {results_json_path}: {e}'

        meshes = plan_payload.get('meshes') or [{}]
        mesh = meshes[0]
        regions = mesh.get('regions') or []
        num_clusters = sum(len(region.get('clusters') or []) for region in regions)
        num_viewpoints = sum(
            1 for region in regions for cluster in (region.get('clusters') or [])
            if 'viewpoint' in cluster)

        envelope = {
            'envelope_version': '1.0',
            'type': envelope_type,
            'plm_context': {
                'eng_item_id': part['eng_item_id'],
                'part_number': part.get('part_number'),
                'revision': part.get('revision'),
                'cestamp': part.get('cestamp'),
                'collab_space': part.get('collab_space'),
                'plan_doc_id': None,
                'generated_at': utc_now(),
                'generated_by': self.client.config.username,
                'cell_id': self.config.cell_id,
                'software_version': f'ViewpointGeneration@{_software_version()}',
            },
            'pipeline_config': {
                'camera_config': mesh.get('camera_config', {}),
                'source_format': mesh.get('source_format'),
            },
            'summary': {
                'num_regions': len(regions),
                'num_clusters': num_clusters,
                'num_viewpoints': num_viewpoints,
                'mesh_file': mesh.get('file'),
                'mesh_units': mesh.get('units'),
                'mesh_dimensions': mesh.get('dimensions'),
                'surface_area': mesh.get('surface_area'),
            },
            'plan': plan_payload,
        }
        return envelope, 'Envelope built.'

    @staticmethod
    def _unwrap_envelope(envelope_path):
        """Extract the plan payload from an envelope file.

        Accepts a bare results JSON too, so a plan written before the envelope
        existed (or exported by hand) still loads.

        Returns:
            tuple: (dict or None, dict, str) the plan payload, its plm_context,
            and a message.
        """
        try:
            with open(envelope_path) as handle:
                data = json.load(handle)
        except (OSError, ValueError) as e:
            return None, {}, f'Could not read plan file {envelope_path}: {e}'

        if isinstance(data, dict) and 'plan' in data and 'plm_context' in data:
            return data['plan'], data.get('plm_context', {}), 'Envelope unwrapped.'
        if isinstance(data, dict) and 'meshes' in data:
            return data, {}, 'Plan file is a bare results JSON.'
        return None, {}, f'Unrecognized plan file structure: {envelope_path}'

    def _write_envelope(self, envelope, eng_item_id, part):
        """Write an envelope to the catalog's plan directory.

        Returns:
            tuple: (Path or None, str) the written path and a message.
        """
        directory = self.paths.plan_dir_for(eng_item_id)
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        name = (f"PLAN_{part.get('part_number') or eng_item_id}_"
                f"{part.get('revision') or 'NA'}_{timestamp}.json")
        path = directory / name
        try:
            directory.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as handle:
                json.dump(envelope, handle, indent=2)
        except OSError as e:
            return None, f'Could not write plan envelope: {e}'
        return path, f'Wrote plan envelope to {path}.'

    # --- plans --------------------------------------------------------------

    def ensure_plan(self, eng_item_id, check_remote=True):
        """Resolve the inspection plan for a part, downloading one if needed.

        Checks the local catalog first, then 3DX: another cell may already have
        generated and uploaded a plan for this exact CAD revision, and
        reusing it beats re-running the pipeline.

        Args:
            eng_item_id: The part to resolve a plan for.
            check_remote: Query 3DX when no current local plan exists. Set
                False to keep the call purely local (list views call it for
                every row, and a network round trip per row is unaffordable).

        Returns:
            tuple: (PlanStatus, str) the plan's validity and its file path
            ('' when there is no plan).
        """
        part = self.db.get_part(eng_item_id)
        if part is None:
            return PlanStatus.NONE, ''

        local_plan = self.db.get_current_plan(eng_item_id)
        local_path = Path(local_plan['file_path']) if local_plan else None
        local_exists = local_path is not None and local_path.exists()

        if local_exists and local_plan.get('cestamp') == part.get('cestamp'):
            return PlanStatus.CURRENT, str(local_path)

        if check_remote:
            remote_status, remote_path = self._try_download_remote_plan(part)
            if remote_status is not None:
                return remote_status, remote_path

        if local_exists:
            return PlanStatus.STALE, str(local_path)
        return PlanStatus.NONE, ''

    def _try_download_remote_plan(self, part):
        """Look for a 3DX plan matching the part's current CAD revision.

        Returns:
            tuple: (PlanStatus or None, str) None when nothing usable was
            found remotely, so the caller falls back to the local answer.
        """
        eng_item_id = part['eng_item_id']
        try:
            success, _ = self.client.ensure_login()
            if not success:
                return None, ''
            documents = self.client.list_related_documents(
                eng_item_id, title_prefix='PLAN_')
        except (DXAuthError, DXAPIError) as e:
            logger.debug('Remote plan lookup failed for %s: %s', eng_item_id, e)
            return None, ''

        if not documents:
            return None, ''

        # Newest first: the most recently modified plan is the one to trust.
        documents.sort(key=lambda doc: doc.get('modified') or '', reverse=True)
        for document in documents:
            doc_id = document.get('id')
            if not doc_id:
                continue
            destination = (self.paths.plan_dir_for(eng_item_id)
                           / f'{doc_id}.json')
            path, message = self.client.download_document_file(doc_id, destination)
            if path is None:
                logger.debug('Plan document %s not downloadable: %s', doc_id, message)
                continue

            _, plm_context, unwrap_message = self._unwrap_envelope(path)
            plan_cestamp = plm_context.get('cestamp') or ''
            had_local = self.db.get_current_plan(eng_item_id) is not None
            self.db.upsert_plan({
                'plan_id': f'{eng_item_id}:{doc_id}',
                'eng_item_id': eng_item_id,
                'plan_doc_id': doc_id,
                'cestamp': plan_cestamp,
                'file_path': str(path),
                'num_regions': (plm_context.get('num_regions')),
                'generated_at': plm_context.get('generated_at') or utc_now(),
                'uploaded_at': plm_context.get('generated_at'),
                'upload_status': 'synced',
                'source': 'remote',
            })
            self.db.log_sync('plan_downloaded', eng_item_id, doc_id)
            if plan_cestamp and plan_cestamp != part.get('cestamp'):
                # A remote plan exists but targets different geometry; it is
                # no better than a stale local one.
                return PlanStatus.STALE, str(path)
            return (PlanStatus.UPDATED_FROM_REMOTE if had_local
                    else PlanStatus.DOWNLOADED), str(path)
        return None, ''

    def upload_plan(self, eng_item_id, results_json_path):
        """Wrap a results JSON in an envelope and upload it to 3DX.

        Returns:
            tuple: (doc_id or None, str) the created document id and a message.
        """
        part = self.db.get_part(eng_item_id)
        if part is None:
            return None, f'Unknown part: {eng_item_id}.'

        envelope, message = self._build_envelope(part, results_json_path)
        if envelope is None:
            return None, message

        envelope_path, message = self._write_envelope(envelope, eng_item_id, part)
        if envelope_path is None:
            return None, message

        summary = envelope['summary']
        plan_id = f"{eng_item_id}:{envelope_path.stem}"
        # Record the plan locally before the upload: a failed upload must still
        # leave a usable plan on the cell, retryable later.
        self.db.upsert_plan({
            'plan_id': plan_id,
            'eng_item_id': eng_item_id,
            'plan_doc_id': None,
            'cestamp': part.get('cestamp'),
            'file_path': str(envelope_path),
            'num_regions': summary['num_regions'],
            'num_clusters': summary['num_clusters'],
            'num_viewpoints': summary['num_viewpoints'],
            'seg_algorithm': envelope['pipeline_config'].get('segmentation_algorithm'),
            'generated_at': envelope['plm_context']['generated_at'],
            'upload_status': 'uploading',
            'source': 'local',
        })

        title = (f"PLAN_{part.get('part_number') or eng_item_id}_"
                 f"{part.get('revision') or 'NA'}")
        try:
            doc_id, message = self.client.upload_inspection_plan(
                eng_item_id, envelope_path, title,
                collab_space=part.get('collab_space'))
        except (DXAuthError, DXAPIError) as e:
            doc_id, message = None, f'Plan upload failed: {e}'

        if doc_id is None:
            self.db.update_plan(plan_id, upload_status='failed')
            self.db.log_sync('plan_upload_failed', eng_item_id, message)
            return None, message

        self.db.update_plan(plan_id, plan_doc_id=doc_id, uploaded_at=utc_now(),
                            upload_status='synced')
        self.db.log_sync('plan_uploaded', eng_item_id, doc_id)
        return doc_id, message

    # --- inspection results --------------------------------------------------

    def start_run(self, eng_item_id, run_id=None, plan_id=None):
        """Record the start of an inspection run.

        Returns:
            tuple: (run_id, str) the run identifier and a message.
        """
        if self.db.get_part(eng_item_id) is None:
            return None, f'Unknown part: {eng_item_id}.'
        if run_id is None:
            run_id = ('run_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'))
        if plan_id is None:
            current = self.db.get_current_plan(eng_item_id)
            plan_id = current['plan_id'] if current else None
        self.db.create_run({
            'run_id': run_id,
            'eng_item_id': eng_item_id,
            'plan_id': plan_id,
            'started_at': utc_now(),
            'status': 'running',
        })
        self.db.log_sync('run_started', eng_item_id, run_id)
        return run_id, f'Started inspection run {run_id}.'

    def build_result_manifest(self, run_id, anomalies=None, summary=None):
        """Write the result_manifest.json for a run's bundle.

        Returns:
            tuple: (Path or None, str) the manifest path and a message.
        """
        run = self.db.get_run(run_id)
        if run is None:
            return None, f'Unknown inspection run: {run_id}.'
        part = self.db.get_part(run['eng_item_id']) or {}
        plan = self.db.get_plan(run['plan_id']) if run.get('plan_id') else None

        bundle = Path(run.get('result_bundle_path')
                      or self.paths.result_dir_for(run['eng_item_id'], run_id))
        try:
            bundle.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            return None, f'Could not create result bundle directory: {e}'

        files = [
            {
                'path': str(path.relative_to(bundle)),
                'type': _file_kind(path),
                'size_bytes': path.stat().st_size,
            }
            for path in sorted(bundle.rglob('*'))
            if path.is_file() and path.name != 'result_manifest.json'
        ]

        anomalies = anomalies or []
        scores = [a.get('score', 0.0) for a in anomalies]
        manifest = {
            'envelope_version': '1.0',
            'type': 'inspection_result',
            'plm_context': {
                'eng_item_id': run['eng_item_id'],
                'part_number': part.get('part_number'),
                'revision': part.get('revision'),
                'cestamp': part.get('cestamp'),
                'plan_doc_id': plan.get('plan_doc_id') if plan else None,
                'result_doc_id': None,
            },
            'run_context': {
                'run_id': run_id,
                'cell_id': self.config.cell_id,
                'started_at': run.get('started_at'),
                'completed_at': run.get('completed_at') or utc_now(),
                'operator': self.client.config.username,
            },
            'summary': summary or {
                'viewpoints_executed': run.get('viewpoints_executed'),
                'viewpoints_skipped': run.get('viewpoints_skipped') or 0,
                'anomalies_detected': len(anomalies),
                'max_anomaly_score': max(scores) if scores else 0.0,
                'overall_result': run.get('overall_result')
                or ('FAIL' if anomalies else 'PASS'),
                'total_files': len(files),
            },
            'anomalies': anomalies,
            'files': files,
        }

        path = bundle / 'result_manifest.json'
        try:
            with open(path, 'w') as handle:
                json.dump(manifest, handle, indent=2)
        except OSError as e:
            return None, f'Could not write result manifest: {e}'

        self.db.update_run(
            run_id,
            result_bundle_path=str(bundle),
            anomalies_found=len(anomalies),
            max_anomaly_score=max(scores) if scores else 0.0,
            overall_result=manifest['summary']['overall_result'],
            completed_at=manifest['run_context']['completed_at'],
            status='completed')
        return path, f'Wrote result manifest to {path}.'

    def upload_results(self, run_id):
        """Upload an inspection run's result bundle to 3DX.

        Creates an Issue as well when the run's worst anomaly exceeds the
        configured NCR threshold, so a failing inspection lands in the quality
        workflow rather than only in a document.

        Returns:
            tuple: (doc_id or None, str) the created document id and a message.
        """
        run = self.db.get_run(run_id)
        if run is None:
            return None, f'Unknown inspection run: {run_id}.'
        part = self.db.get_part(run['eng_item_id'])
        if part is None:
            return None, f"Unknown part: {run['eng_item_id']}."

        bundle = run.get('result_bundle_path')
        if not bundle or not Path(bundle).exists():
            return None, (f'Run {run_id} has no result bundle on disk; '
                          'build one with build_result_manifest() first.')

        self.db.update_run(run_id, upload_status='uploading')
        title = (f"RESULT_{part.get('part_number') or run['eng_item_id']}_"
                 f"{part.get('revision') or 'NA'}_{run_id}")
        try:
            doc_id, message = self.client.upload_result_bundle(
                eng_item_id=run['eng_item_id'],
                bundle_path=Path(bundle),
                title=title,
                collab_space=part.get('collab_space'))
        except (DXAuthError, DXAPIError) as e:
            doc_id, message = None, f'Result upload failed: {e}'

        if doc_id is None:
            self.db.update_run(run_id, upload_status='failed')
            self.db.log_sync('result_upload_failed', run['eng_item_id'], message)
            return None, message

        self.db.update_run(run_id, result_doc_id=doc_id, upload_status='uploaded')
        self.db.log_sync('result_uploaded', run['eng_item_id'], doc_id)

        max_score = run.get('max_anomaly_score') or 0.0
        if run.get('anomalies_found') and max_score > self.config.ncr_threshold:
            issue_id, issue_message = self.client.create_issue(
                eng_item_id=run['eng_item_id'],
                title=(f"Inspection anomaly: {part.get('part_number')} "
                       f"{part.get('revision')}"),
                description=self._format_anomaly_summary(run))
            if issue_id:
                self.db.log_sync('issue_created', run['eng_item_id'], issue_id)
                message = f'{message} Raised issue {issue_id}.'
            else:
                message = f'{message} Issue creation failed: {issue_message}'
        return doc_id, message

    def _format_anomaly_summary(self, run):
        """Human-readable anomaly summary for an Issue description."""
        return (
            f"Automated surface inspection on cell {self.config.cell_id} "
            f"found {run.get('anomalies_found')} anomaly/anomalies "
            f"(max score {run.get('max_anomaly_score')}). "
            f"Run {run['run_id']} started {run.get('started_at')}, "
            f"result {run.get('overall_result')}. "
            f"Full result bundle: document {run.get('result_doc_id')}.")

    # --- status -------------------------------------------------------------

    def status(self):
        """Catalog and sync status for the picker UI and ROS services.

        Returns:
            dict: counts, last sync entry, and whether a sync is in flight.
        """
        last = self.db.last_sync()
        return {
            'counts': self.db.counts(),
            'syncing': self.syncing,
            'last_sync': last,
            'last_result': self.last_result.to_dict() if self.last_result else None,
            'bookmark_scope': self.config.bookmark_scope,
            'collab_space_filter': self.config.collab_space_filter,
        }


def build_sync(dx_config=None, sync_config=None, paths=None):
    """Assemble a `CatalogSync` from environment-derived configuration.

    Returns:
        CatalogSync
    """
    dx_config = dx_config or DXConfig.from_env()
    sync_config = sync_config or SyncConfig.from_env()
    paths = paths or CatalogPaths.from_env()
    client = DXClient(dx_config)
    return CatalogSync(client, paths.db_path, sync_config, paths)


def main(argv=None):
    """Command-line entry point: `python -m viewpoint_generation.catalog.sync`."""
    parser = argparse.ArgumentParser(
        prog='viewpoint_generation.catalog.sync',
        description='Synchronize the local parts catalog with 3DEXPERIENCE.')
    parser.add_argument('--full', action='store_true',
                        help='Run a full sync (enumerate and reconcile everything).')
    parser.add_argument('--incremental', action='store_true',
                        help='Run a lightweight incremental sync.')
    parser.add_argument('--status', action='store_true',
                        help='Print catalog status and exit.')
    parser.add_argument('--list', action='store_true',
                        help='List catalogued parts and exit.')
    parser.add_argument('--contexts', action='store_true',
                        help='List the security contexts available to the user.')
    parser.add_argument('--bookmarks', action='store_true',
                        help='List bookmarks visible to the user.')
    parser.add_argument('--sync-item', metavar='ENG_ITEM_ID',
                        help='Add or refresh a single item by id, regardless of '
                             'the configured scope. Useful when a part sits '
                             'outside the search scope or beyond the scan budget.')
    parser.add_argument('--fetch-step', metavar='ENG_ITEM_ID',
                        help='Fetch the STEP file for one part.')
    parser.add_argument('--scope', metavar='SEARCH',
                        help='Override DX_BOOKMARK_SCOPE for this run.')
    parser.add_argument('--collab-space', metavar='NAME',
                        help='Override the collaborative-space filter for this run.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable debug logging.')
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)-7s %(name)s: %(message)s')

    sync_config = SyncConfig.from_env()
    if args.scope:
        sync_config.bookmark_scope = args.scope
    if args.collab_space:
        sync_config.collab_space_filter = args.collab_space

    dx_config = DXConfig.from_env()
    configured, message = dx_config.is_configured()
    needs_remote = (args.full or args.incremental or args.contexts
                    or args.bookmarks or args.fetch_step or args.sync_item)
    if not configured and needs_remote:
        logger.error(message)
        return 2

    catalog = build_sync(dx_config, sync_config)

    if args.contexts:
        contexts = catalog.client.get_security_contexts()
        if not contexts:
            print('No security contexts reported.')
        for context in contexts:
            print(context)
        return 0

    if args.bookmarks:
        for bookmark in catalog.client.list_bookmarks():
            print(f"{bookmark.get('id')}  {bookmark.get('title')!r}  "
                  f"[{bookmark.get('collabspace')}]")
        return 0

    if args.sync_item:
        success, message = catalog.sync_single_item(args.sync_item)
        print(message)
        if not success:
            return 1

    if args.fetch_step:
        path, message = catalog.fetch_step(args.fetch_step)
        print(message)
        return 0 if path else 1

    result = None
    if args.full:
        result = catalog.run_full_sync()
    elif args.incremental:
        result = catalog.run_incremental_sync()

    if args.list or (result is None and not args.status):
        parts = catalog.db.list_parts()
        print(f'{len(parts)} part(s) in catalog:')
        for part in parts:
            step = 'cached' if part.get('step_path') else (
                'available' if part.get('step_available') else 'none')
            print(f"  {part['eng_item_id']}  {part['title']!r:40} "
                  f"rev {part.get('revision')}  {part.get('maturity')}  "
                  f"[{part.get('collab_space')}]  step={step}")

    if args.status or result is not None:
        status = catalog.status()
        counts = status['counts']
        print(f"Catalog: {counts['total']} parts, {counts['ready']} with a cached STEP")
        print(f"Status breakdown: {counts['by_status']}")
        if status['last_sync']:
            print(f"Last sync: {status['last_sync']['timestamp']} "
                  f"({status['last_sync']['action']}) {status['last_sync']['detail']}")

    if result is not None and not result.success:
        for detail in result.error_details:
            logger.error(detail)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
