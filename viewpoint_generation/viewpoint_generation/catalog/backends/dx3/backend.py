"""The 3DEXPERIENCE backend: `PLMBackend` over `DXClient`.

Identity and change detection are unchanged from the original catalog:
`part_id` is the engineering item id and `change_token` is its **cestamp**,
the optimistic-concurrency stamp 3DX mutates on every modification.

Scoping follows the tenant findings (inspection-docker
`docs/3dx_integration/TENANT_API_FINDINGS.md`):

* A scope naming a 3DX bookmark is enumerated through the bookmark
  (`dsbks:BksMask.Items`), filtered to engineering items, with no search.
* Otherwise the scope is a search string. `collab_space_filter` is pushed to
  the server as `[ds6w:project]:"<space>"`; if that predicate returns nothing
  on the first page the search falls back to the plain scope and filters
  locally. The search's `totalItems` is the page size, not the result size,
  so paging continues until a short page.

STEP comes from derived outputs (`dsdo:DerivedOutputs/Locate`). The tenant
publishes no per-object preview images, so THUMBNAILS only means "look for
one"; the core renders previews from STEP regardless.
"""

import logging
from pathlib import Path

from viewpoint_generation.catalog.backends.base import (BackendAuthError, BackendError,
                                                        Capability, RemotePart,
                                                        RemoteRef, ScopeTruncated)
from viewpoint_generation.catalog.backends.dx3.client import (DXAPIError, DXAuthError,
                                                              DXClient)
from viewpoint_generation.catalog.backends.dx3.config import DX3Settings, DXConfig

logger = logging.getLogger(__name__)

NAME = 'dx3'

# Document title prefixes by attachment kind, as uploads have always used.
TITLE_PREFIXES = {'plan': 'PLAN_', 'result_bundle': 'RESULT_'}


def _to_remote(item):
    """A normalized DXClient item as a RemotePart."""
    extra = {key: str(item[key]) for key in ('type', 'owner', 'modified', 'created')
             if item.get(key)}
    return RemotePart(
        backend=NAME, part_id=item['eng_item_id'], title=item.get('title') or '',
        part_number=item.get('part_number'), revision=item.get('revision'),
        change_token=item.get('cestamp') or '', maturity=item.get('maturity'),
        container=item.get('collab_space'), geometry_available=None,
        description=item.get('description') or '', extra=extra)


class DX3Backend:
    """3DEXPERIENCE engineering items as a PLM backend.

    Args:
        config: Tenant URLs and credentials (`DXConfig`).
        settings: Scope and behaviour (`DX3Settings`).
        client: An existing `DXClient` (tests inject one).
    """

    name = NAME

    def __init__(self, config=None, settings=None, client=None):
        self.config = config or DXConfig.from_env()
        self.settings = settings or DX3Settings.from_env()
        self.client = client or DXClient(self.config)
        # Set by the last enumeration when the scope resolved to a bookmark.
        self.active_bookmark = None
        self._listener = None

    # --- identity and capabilities -------------------------------------------

    @property
    def scope(self):
        return self.settings.bookmark_scope

    @scope.setter
    def scope(self, value):
        self.settings.bookmark_scope = value

    @property
    def capabilities(self):
        """Computed from configuration, which ROS parameters can change live."""
        capabilities = {Capability.GEOMETRY, Capability.ATTACH, Capability.ISSUES}
        if self.settings.fetch_thumbnails:
            capabilities.add(Capability.THUMBNAILS)
        if self.settings.document_links:
            capabilities.add(Capability.LINK_ATTACHMENT)
        if self.config.jms_broker_url:
            capabilities.add(Capability.EVENTS)
        return frozenset(capabilities)

    def is_configured(self):
        return self.config.is_configured()

    def connect(self):
        try:
            success, message = self.client.ensure_login()
        except DXAPIError as e:
            raise BackendError(str(e)) from e
        if not success:
            raise BackendAuthError(message)

    # --- enumeration -----------------------------------------------------------

    def _in_space(self, item):
        space = self.settings.collab_space_filter
        return not space or (item.get('collab_space') or '') == space

    def _resolve_bookmark(self, scope):
        """The 3DX bookmark the scope names, or None when it is a search string."""
        scope = (scope or '').strip()
        if not scope or scope == '*' or not self.settings.use_bookmark_scope:
            return None
        try:
            return self.client.find_bookmark(scope)
        except (DXAuthError, DXAPIError) as e:
            logger.debug('Bookmark lookup for %r failed: %s', scope, e)
            return None

    def _scope_query(self, scope):
        """(query, fallback): the space predicate ANDed with the scope, and the
        plain scope for tenants whose index lacks the [ds6w:project] tag."""
        scope = (scope or '*').strip()
        space = (self.settings.collab_space_filter or '').strip()
        terms = []
        if space:
            terms.append(f'[ds6w:project]:"{space}"')
        if scope and scope != '*':
            terms.append(scope)
        if not terms:
            return '*', '*'
        return ' AND '.join(terms), (scope or '*')

    def _bookmark_parts(self, bookmark):
        members = self.client.list_bookmark_items(bookmark['id'])
        parts = [m for m in members if self.client.is_eng_item(m)]
        budget = self.settings.max_items
        logger.info('Bookmark %r holds %d member(s), %d of them engineering items.',
                    bookmark.get('title'), len(members), len(parts))
        if len(parts) > budget:
            logger.warning('Bookmark %r holds %d engineering items, more than the '
                           'CATALOG_MAX_ITEMS budget of %d; only the first %d are '
                           'catalogued.', bookmark.get('title'), len(parts), budget, budget)
        seen = set()
        for member in parts[:budget]:
            try:
                item = self.client.get_eng_item(member['id'])
            except (DXAuthError, DXAPIError) as e:
                logger.warning('Could not read bookmarked item %s: %s', member['id'], e)
                continue
            if item and item.get('eng_item_id') and item['eng_item_id'] not in seen:
                seen.add(item['eng_item_id'])
                yield _to_remote(item)
        if len(members) >= budget:
            raise ScopeTruncated(len(members), budget)

    def _search_parts(self, scope):
        seen = set()
        skip = scanned = 0
        budget, page = self.settings.max_items, self.settings.page_size
        query, fallback = self._scope_query(scope)
        while scanned < budget:
            page_size = min(page, budget - scanned)
            items, _ = self.client.search_eng_items(query=query, top=page_size, skip=skip)
            if not items and skip == 0 and query != fallback:
                # The space predicate matched nothing on the very first page.
                # A tenant whose index lacks the tag answers 200 with no
                # results rather than an error, so retry without it and
                # filter by space locally.
                logger.info('Scope query %r returned nothing; retrying as %r '
                            'with client-side space filtering.', query, fallback)
                query = fallback
                items, _ = self.client.search_eng_items(query=query, top=page_size, skip=skip)
            if not items:
                break
            for item in items:
                item_id = item.get('eng_item_id')
                if item_id and item_id not in seen and self._in_space(item):
                    seen.add(item_id)
                    yield _to_remote(item)
            scanned += len(items)
            skip += len(items)
            if len(items) < page_size:
                break
        if scanned >= budget:
            raise ScopeTruncated(
                scanned, budget,
                f'Scan stopped at the {budget}-item budget. Narrow DX_BOOKMARK_SCOPE '
                'or raise CATALOG_MAX_ITEMS to enumerate the whole scope.')

    def list_scope(self, scope):
        try:
            bookmark = self._resolve_bookmark(scope)
            self.active_bookmark = bookmark
            if bookmark is not None:
                yield from self._bookmark_parts(bookmark)
            else:
                yield from self._search_parts(scope)
        except DXAuthError as e:
            raise BackendAuthError(str(e)) from e
        except DXAPIError as e:
            raise BackendError(str(e)) from e

    def list_recent(self, scope):
        """One search page, for an incremental pass.

        The dseng search has no modified-after filter, so this is the first
        page of the scope. A bookmark scope is small and exact, and a full
        page probably hides more behind it; both return None, which makes the
        core run a full sync instead.
        """
        try:
            if self._resolve_bookmark(scope) is not None:
                return None
            query, _ = self._scope_query(scope)
            items, _ = self.client.search_eng_items(query=query, top=self.settings.page_size)
        except DXAuthError as e:
            raise BackendAuthError(str(e)) from e
        except DXAPIError as e:
            raise BackendError(str(e)) from e
        if len(items) >= self.settings.page_size:
            logger.info('Incremental sync filled its %d-item page; escalating to a '
                        'full sync.', self.settings.page_size)
            return None
        return iter([_to_remote(item) for item in items
                     if item.get('eng_item_id') and self._in_space(item)])

    def get_part(self, part_id):
        try:
            item = self.client.get_eng_item(part_id)
        except DXAuthError as e:
            raise BackendAuthError(str(e)) from e
        except DXAPIError as e:
            raise BackendError(str(e)) from e
        return _to_remote(item) if item and item.get('eng_item_id') else None

    # --- files -------------------------------------------------------------------

    def check_geometry(self, part):
        try:
            derived_id, _ = self.client.find_step_output(part.part_id)
        except (DXAPIError, DXAuthError) as e:
            logger.debug('Derived-output check failed for %s: %s', part.part_id, e)
            return False
        return derived_id is not None

    def fetch_geometry(self, part, dest_dir):
        try:
            derived_id, step_file = self.client.find_step_output(part.part_id)
            if derived_id is None:
                return None
            dest = Path(dest_dir) / 'download.stp'
            path, message = self.client.download_derived_output(
                derived_id, step_file['id'], dest)
        except (DXAuthError, DXAPIError) as e:
            logger.warning('STEP fetch failed for %s: %s', part.part_id, e)
            return None
        if path is None:
            logger.warning('STEP download failed for %s: %s', part.part_id, message)
            return None
        logger.info('Downloaded %s STEP for %s.', step_file.get('format'), part.part_id)
        return Path(path)

    def fetch_thumbnail(self, part, dest_dir):
        """A real preview image, if the tenant publishes one (generic per-type
        icons are refused: a grid of identical icons tells the operator nothing)."""
        try:
            data, _url, is_generic = self.client.get_thumbnail(part.part_id)
        except (DXAPIError, DXAuthError) as e:
            logger.debug('Thumbnail unavailable for %s: %s', part.part_id, e)
            return None
        if not data or is_generic:
            return None
        dest = Path(dest_dir) / 'remote_thumbnail.png'
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
        except OSError as e:
            logger.warning('Could not write thumbnail for %s: %s', part.part_id, e)
            return None
        return dest

    def read_attributes(self, part):
        """Not available: no route to material/finish has been confirmed on
        this tenant's REST surface, so ATTRIBUTES is not declared."""
        return {}

    def attach_file(self, part, path, kind, title):
        """Upload a plan file or a result bundle directory as one Document."""
        link = self.settings.document_links
        try:
            if kind == 'result_bundle':
                doc_id, message = self.client.upload_result_bundle(
                    eng_item_id=part.part_id, bundle_path=Path(path), title=title,
                    collab_space=part.container, link_to_item=link)
            else:
                doc_id, message = self.client.upload_inspection_plan(
                    part.part_id, path, title, collab_space=part.container,
                    link_to_item=link)
        except (DXAuthError, DXAPIError) as e:
            raise BackendError(f'Upload failed: {e}') from e
        if doc_id is None:
            raise BackendError(message)
        return RemoteRef(backend=NAME, ref_id=doc_id, title=title, note=message)

    def list_attachments(self, part, kind):
        if not self.settings.document_links:
            return []
        try:
            documents = self.client.list_related_documents(
                part.part_id, title_prefix=TITLE_PREFIXES.get(kind))
        except (DXAuthError, DXAPIError) as e:
            logger.debug('Attachment lookup failed for %s: %s', part.part_id, e)
            return []
        return [RemoteRef(backend=NAME, ref_id=doc['id'], title=doc.get('title') or '',
                          modified=doc.get('modified') or '')
                for doc in documents if doc.get('id')]

    def fetch_attachment(self, ref, dest_dir):
        dest = Path(dest_dir) / f'{ref.ref_id}.json'
        try:
            path, message = self.client.download_document_file(ref.ref_id, dest)
        except (DXAuthError, DXAPIError) as e:
            logger.debug('Document %s not downloadable: %s', ref.ref_id, e)
            return None
        if path is None:
            logger.debug('Document %s not downloadable: %s', ref.ref_id, message)
        return path

    def raise_issue(self, part, title, body):
        try:
            issue_id, message = self.client.create_issue(
                eng_item_id=part.part_id, title=title, description=body)
        except (DXAuthError, DXAPIError) as e:
            raise BackendError(f'Issue creation failed: {e}') from e
        if not issue_id:
            raise BackendError(message)
        return RemoteRef(backend=NAME, ref_id=issue_id, title=title, note=message)

    # --- events ------------------------------------------------------------------

    def subscribe(self, on_change):
        """JMS events relayed by an EIF broker, when DX_JMS_BROKER_URL is set."""
        if not self.config.jms_broker_url:
            return None
        from viewpoint_generation.catalog.backends.dx3.jms import JMSEventListener

        listener = JMSEventListener(self.config.jms_broker_url, on_change)
        started, message = listener.start()
        if not started:
            logger.warning(message)
            listener.stop()
            return None
        logger.info(message)
        self._listener = listener

        def stop():
            listener.stop()
            self._listener = None
        return stop

    # --- status ------------------------------------------------------------------

    def describe(self):
        bookmark = self.active_bookmark or {}
        return {
            'scope': self.settings.bookmark_scope,
            'bookmark': bookmark.get('title') or '',
            'collab_space_filter': self.settings.collab_space_filter,
            'user': self.config.username,
            'events': self._listener.status() if self._listener else None,
        }

    def close(self):
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        self.client.close()

    # --- CLI hooks ---------------------------------------------------------------

    @staticmethod
    def add_cli_arguments(parser):
        """3DX-only CLI options, registered by the backend."""
        group = parser.add_argument_group('dx3 (3DEXPERIENCE)')
        group.add_argument('--contexts', action='store_true',
                           help='List the security contexts available to the user.')
        group.add_argument('--bookmarks', action='store_true',
                           help='List bookmarks visible to the user.')
        group.add_argument('--collab-space', metavar='NAME',
                           help='Override DX_COLLAB_SPACE for this run.')

    def apply_cli_overrides(self, args):
        """Apply this backend's CLI overrides before anything runs."""
        if getattr(args, 'collab_space', None):
            self.settings.collab_space_filter = args.collab_space

    def handle_cli(self, args):
        """Run a 3DX-only CLI command. Returns an exit code, or None."""
        if not (getattr(args, 'contexts', False) or getattr(args, 'bookmarks', False)):
            return None
        configured, message = self.is_configured()
        if not configured:
            logger.error(message)
            return 2
        if args.contexts:
            contexts = self.client.get_security_contexts()
            if not contexts:
                print('No security contexts reported.')
            for context in contexts:
                print(context)
            return 0
        if args.bookmarks:
            for bookmark in self.client.list_bookmarks():
                members = self.client.list_bookmark_items(bookmark['id'])
                parts = [m for m in members if self.client.is_eng_item(m)]
                print(f"{bookmark.get('id')}  {bookmark.get('title')!r:32} "
                      f"[{bookmark.get('collabspace')}]  "
                      f'{len(members)} member(s), {len(parts)} engineering item(s)')
            return 0
        return None


BACKEND_CLASS = DX3Backend


def from_env():
    """The registry factory."""
    return DX3Backend(DXConfig.from_env(), DX3Settings.from_env())
