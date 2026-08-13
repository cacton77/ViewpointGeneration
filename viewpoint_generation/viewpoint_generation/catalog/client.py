"""Authenticated REST client for a 3DEXPERIENCE (ENOVIA) tenant.

`DXClient` wraps a `requests.Session` and hides three pieces of 3DX protocol
awkwardness from the rest of the catalog subsystem:

**CAS login across two regions.** Cloud tenants routinely split the IAM
("passport") host from the data services ("3DSpace") host -- this deployment
authenticates against `eu1` and calls APIs on `usw2`. `login()` fetches a
login ticket from the passport, POSTs the credentials to obtain the CASTGC
cookie, then requests a 3DSpace resource and *follows the CAS redirect chain*
(space -> passport -> space with `?ticket=ST-...`). That final hop is what
mints the space-domain `JSESSIONID`; simply copying cookies between hosts does
not work, because the space never issues a session without a service ticket.

**Per-request boilerplate.** Every data call needs the `SecurityContext`
header (including its `ctx::` prefix), the `ENO_CSRF_TOKEN` header, and, on
cloud tenants, a `tenant=` query parameter. `_request()` adds all three.

**Silent session expiry.** An expired 3DX session does not return 401 -- it
answers with a 302 to the passport login page, so a JSON call comes back as
HTML with status 200. `_looks_like_login_redirect()` detects that and
`_request()` re-authenticates once and retries.

Endpoint availability varies by tenant and API version. Methods that depend on
services this tenant does not expose (notably `dsdo` derived outputs) degrade
to an empty result and log a warning rather than raising, so a catalog sync
still completes.
"""

import logging
import urllib.parse
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# Mask accepted by this tenant's dseng service. Richer masks (Mask.Details,
# Mask.All, Mask.SpecificationRelation) are rejected with "Mask does not
# exist", so Default is the only portable choice.
DEFAULT_MASK = 'dskern:Mask.Default'


class DXAuthError(RuntimeError):
    """Raised when authentication against the 3DX passport fails."""


class DXAPIError(RuntimeError):
    """Raised when a 3DX API call fails in a way the caller cannot recover from."""


class DXClient:
    """Authenticated session against a 3DEXPERIENCE tenant.

    Args:
        config: A `DXConfig` carrying tenant URLs and credentials.
    """

    def __init__(self, config):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            # 3DPassport's login form rejects some non-browser agents.
            'User-Agent': 'ViewpointGeneration-Catalog/1.0',
        })
        self.csrf_token = None
        self.logged_in = False

    # --- authentication -------------------------------------------------

    def login(self):
        """Authenticate against the passport and establish a 3DSpace session.

        Performs the full CAS dance: login ticket -> credential POST ->
        service-ticket redirect to the space -> CSRF token.

        Returns:
            tuple: (bool, str) success flag and message.
        """
        configured, message = self.config.is_configured()
        if not configured:
            return False, message

        passport = self.config.passport_url.rstrip('/')
        try:
            # 1. Login ticket. The passport answers with {"response","lt"}.
            response = self.session.get(
                f'{passport}/login', params={'action': 'get_auth_params'},
                timeout=self.config.request_timeout)
            response.raise_for_status()
            payload = response.json()
            login_ticket = payload.get('lt')
            if not login_ticket:
                return False, f'Passport did not return a login ticket: {payload}'

            # 2. Credentials. Success leaves CASTGC_* cookies on the passport
            #    domain; failure re-renders the login form (still HTTP 200), so
            #    the CASTGC cookie -- not the status code -- is the real signal.
            response = self.session.post(
                f'{passport}/login',
                data={
                    'lt': login_ticket,
                    'username': self.config.username,
                    'password': self.config.password,
                    'rememberMe': 'false',
                },
                timeout=self.config.request_timeout)
            response.raise_for_status()
            if not any(name.startswith('CASTGC') for name in
                       (cookie.name for cookie in self.session.cookies)):
                return False, ('Passport rejected the credentials '
                               f'for user {self.config.username!r}.')

            # 3. Space session + CSRF token. requests follows the
            #    space -> passport -> space?ticket=ST-... redirect chain, and
            #    the final hop sets the space-domain JSESSIONID.
            success, message = self._fetch_csrf()
            if not success:
                return False, message
        except requests.RequestException as e:
            return False, f'3DX login failed: {e}'
        except ValueError as e:
            return False, f'3DX login returned an unparseable response: {e}'

        self.logged_in = True
        return True, f'Authenticated to 3DX as {self.config.username}.'

    def _fetch_csrf(self):
        """Fetch and store the ENO_CSRF_TOKEN required for write operations."""
        url = f"{self.config.space_url.rstrip('/')}/resources/v1/application/CSRF"
        response = self.session.get(url, params=self._tenant_params(),
                                    timeout=self.config.request_timeout)
        response.raise_for_status()
        if self._looks_like_login_redirect(response):
            return False, 'CSRF request was redirected to the login page.'
        try:
            self.csrf_token = response.json()['csrf']['value']
        except (ValueError, KeyError) as e:
            return False, f'Could not read CSRF token from 3DX response: {e}'
        return True, 'CSRF token acquired.'

    def ensure_login(self):
        """Log in if this client has not authenticated yet.

        Returns:
            tuple: (bool, str) success flag and message.
        """
        if self.logged_in:
            return True, 'Already authenticated.'
        return self.login()

    def _tenant_params(self, extra=None):
        """Query parameters with the tenant id appended when configured."""
        params = dict(extra or {})
        if self.config.tenant:
            params['tenant'] = self.config.tenant
        return params

    def _headers(self, extra=None):
        """Standard 3DX headers: security context, CSRF token, JSON accept."""
        headers = {'Accept': 'application/json'}
        if self.config.security_context:
            # Sent unencoded; the ctx:: prefix is part of the value.
            headers['SecurityContext'] = self.config.security_context
        if self.csrf_token:
            headers['ENO_CSRF_TOKEN'] = self.csrf_token
        headers.update(extra or {})
        return headers

    @staticmethod
    def _looks_like_login_redirect(response):
        """True when a data call was answered by the passport login page.

        An expired 3DX session does not produce a 401; the request is redirected
        to the passport, which returns the login form as HTML with status 200.
        """
        content_type = response.headers.get('content-type', '')
        if 'text/html' in content_type and 'login' in response.url.lower():
            return True
        return 'action=get_auth_params' in response.text[:2000]

    # --- request plumbing -----------------------------------------------

    def _request(self, method, url, params=None, headers=None, retry=True, **kwargs):
        """Issue a request with 3DX headers, retrying once after re-login.

        Args:
            method: HTTP verb.
            url: Absolute URL.
            params: Query parameters; the tenant id is added automatically.
            headers: Extra headers merged over the standard set.
            retry: Re-authenticate and retry once on an expired session.

        Returns:
            requests.Response

        Raises:
            DXAPIError: The request could not be issued at all.
        """
        timeout = kwargs.pop('timeout', self.config.request_timeout)
        try:
            response = self.session.request(
                method, url, params=self._tenant_params(params),
                headers=self._headers(headers), timeout=timeout, **kwargs)
        except requests.RequestException as e:
            raise DXAPIError(f'{method} {url} failed: {e}') from e

        expired = (response.status_code in (401, 403)
                   or self._looks_like_login_redirect(response))
        if expired and retry:
            logger.info('3DX session expired; re-authenticating.')
            self.logged_in = False
            success, message = self.login()
            if not success:
                raise DXAuthError(message)
            return self._request(method, url, params=params, headers=headers,
                                 retry=False, timeout=timeout, **kwargs)
        return response

    def _get_json(self, url, params=None, headers=None):
        """GET a URL and parse its JSON body.

        Returns:
            dict: Parsed body, or {} when the endpoint is absent (404/405) or
            returns something that is not JSON.
        """
        response = self._request('GET', url, params=params, headers=headers)
        if response.status_code in (404, 405):
            logger.debug('3DX endpoint unavailable (%s): %s',
                         response.status_code, url)
            return {}
        if response.status_code >= 400:
            logger.warning('3DX call failed (%s): %s -> %s', response.status_code,
                           url, response.text[:200])
            return {}
        try:
            return response.json()
        except ValueError:
            logger.warning('3DX returned a non-JSON body for %s', url)
            return {}

    # --- identity --------------------------------------------------------

    def get_security_contexts(self):
        """Discover the security contexts available to the authenticated user.

        The `e6wCurrentUser` endpoint named in the integration checklist is not
        present on every tenant (it 404s here), so this falls back to the
        People & Organization service, which reports the preferred credential
        and every collaborative space / role couple the user holds.

        Returns:
            list: `ctx::Role.Organization.CollabSpace` strings, preferred first.
        """
        success, message = self.ensure_login()
        if not success:
            logger.error(message)
            return []

        space = self.config.space_url.rstrip('/')
        contexts = []

        payload = self._get_json(
            f'{space}/resources/v1/application/e6w/api/v1/e6wCurrentUser')
        for entry in payload.get('securityContexts', []) or []:
            if isinstance(entry, str):
                contexts.append(entry)

        person_url = f'{space}/resources/modeler/pno/person'
        preferred = self._get_json(
            person_url, params={'current': 'true', 'select': 'preferredcredentials'})
        credentials = preferred.get('preferredcredentials') or {}
        if credentials:
            triplet = self._format_context(
                credentials.get('role', {}).get('name'),
                credentials.get('organization', {}).get('name'),
                credentials.get('collabspace', {}).get('name'))
            if triplet and triplet not in contexts:
                contexts.append(triplet)

        spaces = self._get_json(
            person_url, params={'current': 'true', 'select': 'collabspaces'})
        for collab_space in spaces.get('collabspaces', []) or []:
            for couple in collab_space.get('couples', []) or []:
                triplet = self._format_context(
                    couple.get('role', {}).get('name'),
                    couple.get('organization', {}).get('name'),
                    collab_space.get('name'))
                if triplet and triplet not in contexts:
                    contexts.append(triplet)
        return contexts

    @staticmethod
    def _format_context(role, organization, collab_space):
        """Assemble a `ctx::Role.Organization.CollabSpace` string."""
        if not (role and organization and collab_space):
            return None
        return f'ctx::{role}.{organization}.{collab_space}'

    # --- engineering items ------------------------------------------------

    def search_eng_items(self, query='*', mask=DEFAULT_MASK, top=100, skip=0):
        """Search engineering items visible to the current security context.

        Args:
            query: 3DX search string. Plain text matches names/titles;
                simple predicates such as `owner:can28` are also honoured.
                `'*'` matches everything visible.
            mask: Response mask. Only `dskern:Mask.Default` is available here.
            top: Maximum items to return.
            skip: Items to skip, for paging.

        Returns:
            tuple: (items, total) where items is a list of normalized dicts and
            total is the remote total item count.
        """
        success, message = self.ensure_login()
        if not success:
            raise DXAuthError(message)

        url = (f"{self.config.space_url.rstrip('/')}"
               '/resources/v1/modeler/dseng/dseng:EngItem/search')
        params = {'$searchStr': query or '*', '$mask': mask, '$top': str(top)}
        if skip:
            params['$skip'] = str(skip)
        payload = self._get_json(url, params=params)
        members = payload.get('member', []) or []
        total = payload.get('totalItems', len(members))
        return [self._normalize_item(member) for member in members], total

    def get_eng_item(self, item_id, mask=DEFAULT_MASK):
        """Fetch a single engineering item by id.

        Returns:
            dict: Normalized item fields, or None when not found.
        """
        success, message = self.ensure_login()
        if not success:
            raise DXAuthError(message)

        url = (f"{self.config.space_url.rstrip('/')}"
               f'/resources/v1/modeler/dseng/dseng:EngItem/{item_id}')
        payload = self._get_json(url, params={'$mask': mask})
        members = payload.get('member', []) or []
        if not members:
            return None
        return self._normalize_item(members[0])

    @staticmethod
    def _normalize_item(member):
        """Map a raw dseng search member onto the `parts` table's field names."""
        return {
            'eng_item_id': member.get('id'),
            'title': member.get('title') or member.get('name') or '',
            'part_number': member.get('name'),
            'revision': member.get('revision'),
            'cestamp': member.get('cestamp') or '',
            'maturity': member.get('state'),
            'type': member.get('type'),
            'collab_space': member.get('collabspace'),
            'description': member.get('description') or '',
            'owner': member.get('owner'),
            'modified': member.get('modified'),
            'created': member.get('created'),
            'thumbnail_url': None,
        }

    def get_object_details(self, object_id):
        """Fetch the 3DSpace `documents` view of any object.

        This service carries fields the `dseng` mask omits -- notably the
        preview image URLs and the attached file list -- and is the same
        service used for FCS check-in/check-out.

        Returns:
            dict: The object's entry, or {} when unavailable.
        """
        url = (f"{self.config.space_url.rstrip('/')}"
               f'/resources/v1/modeler/documents/{object_id}')
        payload = self._get_json(url)
        entries = payload.get('data', []) or []
        return entries[0] if entries else {}

    def list_files(self, object_id):
        """List files attached to an object via the 3DSpace documents service.

        Returns:
            list: File descriptor dicts (empty when the object carries none).
        """
        url = (f"{self.config.space_url.rstrip('/')}"
               f'/resources/v1/modeler/documents/{object_id}/files')
        payload = self._get_json(url)
        return payload.get('data', []) or []

    # --- derived outputs (STEP) -------------------------------------------

    def list_derived_outputs(self, item_id):
        """List derived outputs (converted formats) generated for an item.

        The `dsdo` service is not exposed on every tenant -- it is absent here,
        in which case this returns an empty list and the catalog records the
        item as having no STEP available rather than failing the sync.

        Returns:
            list: Derived output dicts, each with at least `id`, `name`, and
            `format` when the service is available.
        """
        success, message = self.ensure_login()
        if not success:
            raise DXAuthError(message)

        space = self.config.space_url.rstrip('/')
        candidates = (
            (f'{space}/resources/v1/modeler/dsdo/dsdo:DerivedOutput',
             {'$ids': item_id, '$mask': DEFAULT_MASK}),
            (f'{space}/resources/v1/modeler/dsdo/dsdo:DerivedOutput/search',
             {'$searchStr': item_id, '$mask': DEFAULT_MASK}),
        )
        for url, params in candidates:
            payload = self._get_json(url, params=params)
            members = payload.get('member') or payload.get('data') or []
            if members:
                return members
        logger.debug('No derived-output service response for item %s; '
                     'treating the item as having no STEP output.', item_id)
        return []

    def find_step_output(self, item_id):
        """Locate a STEP derived output for an item, if one exists.

        Returns:
            dict: The matching derived output, or None.
        """
        for output in self.list_derived_outputs(item_id):
            haystack = ' '.join(str(output.get(key, '')) for key in
                                ('name', 'title', 'format', 'type')).lower()
            if 'step' in haystack or 'stp' in haystack:
                return output
        return None

    # --- files -------------------------------------------------------------

    def get_download_ticket(self, object_id, file_id=None):
        """Request an FCS download ticket for an object's file.

        Like the check-in ticket, this is a PUT naming the files wanted -- a
        GET is answered with an empty result rather than an error. The reply
        carries a fully-signed URL, so unlike check-in there is no separate
        ticket parameter to post.

        Returns:
            dict: {'url': ticket URL, 'name': suggested filename}, or {} when
            no downloadable file is exposed.
        """
        space = self.config.space_url.rstrip('/')
        url = f'{space}/resources/v1/modeler/documents/{object_id}/files/DownloadTicket'

        file_ids = [file_id] if file_id else [
            entry.get('id') for entry in self.list_files(object_id) if entry.get('id')]
        body = {'data': [{'id': fid} for fid in file_ids if fid]} if file_ids else {'data': [{}]}

        response = self._request('PUT', url, json=body,
                                 headers={'Content-Type': 'application/json'})
        if response.status_code >= 400:
            logger.warning('Download ticket request failed (%s): %s',
                           response.status_code, response.text[:200])
            return {}
        try:
            entries = response.json().get('data', []) or []
        except ValueError:
            return {}
        for entry in entries:
            elements = entry.get('dataelements', {}) or {}
            ticket_url = elements.get('ticketURL') or elements.get('ticketUrl')
            if ticket_url:
                return {'url': ticket_url,
                        'name': elements.get('title') or elements.get('fileName') or ''}
        return {}

    def download_file(self, ticket_url, dest):
        """Stream a ticketed FCS URL to a local path.

        Args:
            ticket_url: A signed FCS URL from `get_download_ticket()`.
            dest: Destination path; parent directories are created.

        Returns:
            tuple: (bool, str) success flag and message.
        """
        dest = Path(dest)
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            # The ticket carries its own authorization; 3DX headers are
            # unnecessary and the FCS host rejects some of them.
            with self.session.get(ticket_url, stream=True,
                                  timeout=self.config.download_timeout) as response:
                if response.status_code >= 400:
                    return False, (f'File download failed ({response.status_code}): '
                                   f'{response.text[:200]}')
                partial = dest.with_suffix(dest.suffix + '.part')
                with open(partial, 'wb') as handle:
                    for chunk in response.iter_content(chunk_size=65536):
                        if chunk:
                            handle.write(chunk)
                partial.replace(dest)
        except (requests.RequestException, OSError) as e:
            return False, f'File download failed: {e}'
        return True, f'Downloaded {dest}.'

    # --- thumbnails --------------------------------------------------------

    # Platform type icons live under this path. They are per-*type*, not
    # per-object, so every Physical Product returns byte-identical artwork --
    # useless for telling parts apart in a picker.
    GENERIC_ICON_MARKER = '/snresources/images/icons/'

    def get_thumbnail(self, item_id):
        """Fetch an item's thumbnail image.

        A rendered CAD preview is published only for objects that expose
        geometry. This tenant does not: the engineering items carry no files,
        their representations are not reachable, and the `documents` service
        offers only the platform's type icon. The returned `is_generic` flag
        says which of the two arrived, so callers can decline to cache
        artwork that is identical for every part.

        Returns:
            tuple: (bytes, str, bool) image data, its source URL, and whether
            it is a generic platform icon rather than a real preview.
            (None, '', False) when no image is published at all.
        """
        details = self.get_object_details(item_id)
        elements = details.get('dataelements', {}) or {}
        for key in ('thumbnail', 'thumbnailUrl', 'image', 'typeicon'):
            image_url = elements.get(key)
            if not image_url:
                continue
            try:
                response = self._request('GET', image_url)
            except DXAPIError as e:
                logger.debug('Thumbnail fetch failed for %s: %s', item_id, e)
                continue
            content_type = response.headers.get('content-type', '')
            if response.status_code == 200 and content_type.startswith('image/'):
                is_generic = self.GENERIC_ICON_MARKER in image_url
                return response.content, image_url, is_generic
        return None, '', False

    # --- bookmarks ---------------------------------------------------------

    def list_bookmarks(self, query='*', top=50):
        """List bookmarks (workspaces) visible to the current context.

        Returns:
            list: Bookmark dicts with `id`, `title`, and `collabspace`.
        """
        success, message = self.ensure_login()
        if not success:
            raise DXAuthError(message)

        url = (f"{self.config.space_url.rstrip('/')}"
               '/resources/v1/modeler/dsbks/dsbks:Bookmark/search')
        payload = self._get_json(url, params={'$searchStr': query or '*',
                                              '$mask': DEFAULT_MASK,
                                              '$top': str(top)})
        return payload.get('member', []) or []

    def find_bookmark(self, title):
        """Find a bookmark by exact (case-insensitive) title.

        Returns:
            dict: The bookmark, or None.
        """
        if not title:
            return None
        for bookmark in self.list_bookmarks(query=title):
            if (bookmark.get('title') or '').strip().lower() == title.strip().lower():
                return bookmark
        return None

    def list_bookmark_items(self, bookmark_id, recursive=True, page_size=1000,
                            _seen=None):
        """List the objects a bookmark contains.

        The mask family here is `dsbks:BksMask.*`, NOT the `dskern:Mask.*` /
        `dsbks:Mask.*` spelling used elsewhere -- `dsbks:Mask.Items` is
        rejected with "Mask does not exist" while `dsbks:BksMask.Items` works.
        That one naming difference is the whole reason bookmark contents look
        unreachable if you guess at the mask name.

        Unlike the search endpoint, the `totalItems` reported inside `items` is
        a genuine total, so it can be trusted to drive paging ($top caps at
        1000 per the service documentation).

        Args:
            bookmark_id: The bookmark (or bookmark folder) object id.
            recursive: Also descend into sub-bookmarks (folders).
            page_size: Items per request, capped at 1000 by the service.
            _seen: Internal guard against cycles in the folder graph.

        Returns:
            list: dicts with `id`, `type`, and `relative_path` for each member.
        """
        success, message = self.ensure_login()
        if not success:
            raise DXAuthError(message)

        seen = _seen if _seen is not None else set()
        if bookmark_id in seen:
            return []
        seen.add(bookmark_id)

        url = (f"{self.config.space_url.rstrip('/')}"
               f'/resources/v1/modeler/dsbks/dsbks:Bookmark/{bookmark_id}')
        collected = []
        skip = 0
        top = min(page_size, 1000)
        while True:
            payload = self._get_json(url, params={'$mask': 'dsbks:BksMask.Items',
                                                  '$top': str(top),
                                                  '$skip': str(skip)})
            members = payload.get('member') or []
            if not members:
                break
            items = members[0].get('items') or {}
            entries = items.get('member') or []
            for entry in entries:
                referenced = entry.get('referencedObject') or {}
                if referenced.get('identifier'):
                    collected.append({
                        'id': referenced['identifier'],
                        'type': referenced.get('type'),
                        'relative_path': referenced.get('relativePath') or '',
                    })
            total = items.get('totalItems')
            skip += len(entries)
            if not entries or (total is not None and skip >= total):
                break

        if recursive:
            for child in self.list_sub_bookmarks(bookmark_id):
                collected.extend(self.list_bookmark_items(
                    child['id'], recursive=True, page_size=page_size, _seen=seen))
        return collected

    def list_sub_bookmarks(self, bookmark_id):
        """List a bookmark's child bookmarks (folders).

        Returns:
            list: dicts with `id` and `type`.
        """
        url = (f"{self.config.space_url.rstrip('/')}"
               f'/resources/v1/modeler/dsbks/dsbks:Bookmark/{bookmark_id}')
        payload = self._get_json(url, params={'$mask': 'dsbks:BksMask.Bookmarks',
                                              '$top': '1000'})
        members = payload.get('member') or []
        if not members:
            return []
        children = []
        for entry in (members[0].get('bookmarks') or {}).get('member') or []:
            referenced = entry.get('referencedObject') or {}
            if referenced.get('identifier'):
                children.append({'id': referenced['identifier'],
                                 'type': referenced.get('type')})
        return children

    @staticmethod
    def is_eng_item(item):
        """True when a bookmark member is an engineering item.

        Bookmarks hold anything the operator dragged in -- Documents,
        requirement groups, electrical systems -- so members have to be
        filtered before being treated as parts.
        """
        return ('dseng:EngItem' in (item.get('relative_path') or '')
                or item.get('type') == 'VPMReference')

    # --- documents: creation, file transfer, relationships -----------------
    #
    # The 3DSpace "documents" service is the tenant's file API. Creation is a
    # POST to the collection; file transfer is a two-step FCS dance (ask 3DSpace
    # for a signed ticket, then move the bytes to/from the FCS host directly,
    # never through 3DSpace). Endpoint paths vary by platform release more than
    # anything else in this client, so each method reports failures as
    # (None/False, message) rather than raising, and logs the exact response.

    def create_document(self, title, description='', collab_space=None,
                        doc_type='Document'):
        """Create a Document object in 3DSpace.

        Args:
            title: Document title, e.g. 'PLAN_BRK-1042_A.3'.
            description: Free-text description.
            collab_space: Collaborative space to create it in. Defaults to the
                one named in the security context.
            doc_type: 3DX type name.

        Returns:
            tuple: (doc_id, message); doc_id is None on failure.
        """
        success, message = self.ensure_login()
        if not success:
            return None, message

        url = f"{self.config.space_url.rstrip('/')}/resources/v1/modeler/documents"
        elements = {'title': title, 'description': description}
        if collab_space:
            elements['collabspace'] = collab_space
        body = {'data': [{'type': doc_type, 'dataelements': elements}]}

        response = self._request(
            'POST', url, json=body,
            headers={'Content-Type': 'application/json'})
        if response.status_code >= 400:
            return None, (f'Document creation failed ({response.status_code}): '
                          f'{response.text[:300]}')
        try:
            entries = response.json().get('data', []) or []
        except ValueError:
            return None, f'Document creation returned a non-JSON body: {response.text[:200]}'
        if not entries:
            return None, f'Document creation returned no object: {response.text[:200]}'
        doc_id = entries[0].get('id')
        return doc_id, f'Created document {doc_id}.'

    def get_checkin_ticket(self, doc_id, file_count=1):
        """Request an FCS upload ticket for a document.

        Returns:
            tuple: (ticket, message) where ticket is a dict carrying at least
            'ticketURL'; None on failure.
        """
        url = (f"{self.config.space_url.rstrip('/')}"
               f'/resources/v1/modeler/documents/{doc_id}/files/CheckinTicket')
        body = {'data': [{'dataelements': {'numberOfFiles': str(file_count)}}]}
        response = self._request('PUT', url, json=body,
                                 headers={'Content-Type': 'application/json'})
        if response.status_code >= 400:
            return None, (f'Check-in ticket request failed ({response.status_code}): '
                          f'{response.text[:300]}')
        try:
            entries = response.json().get('data', []) or []
        except ValueError:
            return None, f'Check-in ticket returned a non-JSON body: {response.text[:200]}'
        for entry in entries:
            elements = entry.get('dataelements', {}) or {}
            if elements.get('ticketURL') or elements.get('ticket'):
                return elements, 'Check-in ticket acquired.'
        return None, f'No check-in ticket in response: {response.text[:300]}'

    def upload_to_fcs(self, ticket, file_path):
        """Upload one file to the FCS host using a check-in ticket.

        3DSpace hands back three things: the FCS endpoint (`ticketURL`), an
        opaque base64 job ticket (`ticket`), and the *name of the form field*
        that ticket must be submitted under (`ticketparamname`, in practice
        `__fcs__jobTicket`). The job ticket is what authorizes the upload and
        tells FCS which store and object the bytes belong to -- posting the
        file without it is accepted at the HTTP level but stores nothing, so
        the field name is read from the ticket rather than hard-coded.

        Returns:
            tuple: (receipt, message); receipt is the FCS response used to
            complete the check-in, or None on failure.
        """
        file_path = Path(file_path)
        ticket_url = ticket.get('ticketURL')
        job_ticket = ticket.get('ticket')
        param_name = ticket.get('ticketparamname') or '__fcs__jobTicket'
        if not ticket_url:
            return None, 'Check-in ticket carried no upload URL.'
        if not job_ticket:
            return None, 'Check-in ticket carried no job ticket.'
        try:
            with open(file_path, 'rb') as handle:
                # The FCS host authenticates by job ticket, not by session, so
                # the 3DSpace headers are deliberately not sent here.
                response = requests.post(
                    ticket_url,
                    data={param_name: job_ticket},
                    files={'file': (file_path.name, handle,
                                    'application/octet-stream')},
                    timeout=self.config.download_timeout)
        except (requests.RequestException, OSError) as e:
            return None, f'FCS upload failed: {e}'
        if response.status_code >= 400:
            return None, (f'FCS upload failed ({response.status_code}): '
                          f'{response.text[:300]}')
        receipt = response.text.strip()
        if not receipt:
            return None, 'FCS accepted the upload but returned no receipt.'
        return receipt, f'Uploaded {file_path.name}.'

    def complete_checkin(self, doc_id, receipt, file_name):
        """Register an FCS-uploaded file against its document.

        Must be POSTed: the same payload sent as PUT is answered with HTTP 200
        and an empty `data` array, having silently done nothing. That is why
        success is judged on the returned file object rather than on the status
        code -- a 200 here does not by itself mean the file was attached.

        Returns:
            tuple: (bool, str) success flag and message.
        """
        url = (f"{self.config.space_url.rstrip('/')}"
               f'/resources/v1/modeler/documents/{doc_id}/files')
        body = {'data': [{'dataelements': {
            'title': file_name,
            'receipt': receipt,
        }}]}
        response = self._request('POST', url, json=body,
                                 headers={'Content-Type': 'application/json'})
        if response.status_code >= 400:
            return False, (f'Check-in completion failed ({response.status_code}): '
                           f'{response.text[:300]}')
        try:
            entries = response.json().get('data', []) or []
        except ValueError:
            return False, f'Check-in returned a non-JSON body: {response.text[:200]}'
        if not entries:
            return False, ('3DX accepted the check-in request but registered no '
                           f'file for {file_name}.')
        return True, f'Checked in {file_name} as {entries[0].get("id")}.'

    def upload_file_to_document(self, doc_id, file_path):
        """Run the full FCS check-in cycle for one file.

        Returns:
            tuple: (bool, str) success flag and message.
        """
        ticket, message = self.get_checkin_ticket(doc_id)
        if ticket is None:
            return False, message
        receipt, message = self.upload_to_fcs(ticket, file_path)
        if receipt is None:
            return False, message
        return self.complete_checkin(doc_id, receipt, Path(file_path).name)

    def attach_document_to_item(self, doc_id, eng_item_id):
        """Attach a Document to an EngItem as a specification relationship.

        Returns:
            tuple: (bool, str) success flag and message.
        """
        url = (f"{self.config.space_url.rstrip('/')}"
               f'/resources/v1/modeler/dseng/dseng:EngItem/{eng_item_id}/dseng:SpecificationDocument')
        body = {'data': [{'id': doc_id, 'type': 'Document'}]}
        response = self._request('POST', url, json=body,
                                 headers={'Content-Type': 'application/json'})
        if response.status_code >= 400:
            return False, (f'Attaching document {doc_id} to {eng_item_id} failed '
                           f'({response.status_code}): {response.text[:300]}')
        return True, f'Attached document {doc_id} to {eng_item_id}.'

    def list_related_documents(self, eng_item_id, title_prefix=None):
        """List Documents attached to an EngItem.

        Args:
            eng_item_id: The engineering item.
            title_prefix: Keep only documents whose title starts with this.

        Returns:
            list: Document dicts with at least 'id' and 'title'.
        """
        success, message = self.ensure_login()
        if not success:
            raise DXAuthError(message)

        space = self.config.space_url.rstrip('/')
        documents = []
        for path in (f'/resources/v1/modeler/dseng/dseng:EngItem/{eng_item_id}'
                     '/dseng:SpecificationDocument',
                     f'/resources/v1/modeler/documents/{eng_item_id}/relateddata'):
            payload = self._get_json(f'{space}{path}')
            members = payload.get('member') or payload.get('data') or []
            for member in members:
                elements = member.get('dataelements', {}) or {}
                documents.append({
                    'id': member.get('id'),
                    'title': member.get('title') or elements.get('title') or '',
                    'revision': member.get('revision') or elements.get('revision'),
                    'modified': member.get('modified') or elements.get('modified'),
                })
            if documents:
                break

        if title_prefix:
            documents = [doc for doc in documents
                         if (doc.get('title') or '').startswith(title_prefix)]
        return documents

    def download_document_file(self, doc_id, dest):
        """Download a Document's primary file via an FCS download ticket.

        Returns:
            tuple: (Path or None, str) the downloaded path and a message.
        """
        ticket = self.get_download_ticket(doc_id)
        if not ticket.get('url'):
            return None, f'No downloadable file exposed on document {doc_id}.'
        success, message = self.download_file(ticket['url'], dest)
        return (Path(dest) if success else None), message

    def revise_document(self, doc_id):
        """Create a new revision of an existing Document.

        Returns:
            tuple: (new_doc_id, message); new_doc_id is None on failure.
        """
        url = (f"{self.config.space_url.rstrip('/')}"
               f'/resources/v1/modeler/documents/{doc_id}/revise')
        response = self._request('POST', url, json={'data': [{}]},
                                 headers={'Content-Type': 'application/json'})
        if response.status_code >= 400:
            return None, (f'Revising document {doc_id} failed '
                          f'({response.status_code}): {response.text[:300]}')
        try:
            entries = response.json().get('data', []) or []
        except ValueError:
            return None, f'Revise returned a non-JSON body: {response.text[:200]}'
        if not entries:
            return None, 'Revise returned no new object.'
        return entries[0].get('id'), f'Revised document {doc_id}.'

    def upload_inspection_plan(self, eng_item_id, plan_path, title,
                               collab_space=None):
        """Upload an inspection plan JSON and attach it to its EngItem.

        Orchestrates the whole sequence: create the Document, run the FCS
        check-in, then relate it to the engineering item.

        Returns:
            tuple: (doc_id, message); doc_id is None on failure.
        """
        doc_id, message = self.create_document(
            title, description='ViewpointGeneration inspection plan',
            collab_space=collab_space)
        if doc_id is None:
            return None, message

        success, upload_message = self.upload_file_to_document(doc_id, plan_path)
        if not success:
            # The Document exists but carries no payload; say so plainly rather
            # than reporting a success the operator cannot use.
            return None, (f'Created document {doc_id} but the file check-in '
                          f'failed: {upload_message}')

        attached, attach_message = self.attach_document_to_item(doc_id, eng_item_id)
        if not attached:
            return doc_id, (f'Uploaded plan as {doc_id}, but attaching it to '
                            f'{eng_item_id} failed: {attach_message}')
        return doc_id, f'Uploaded inspection plan as document {doc_id}.'

    def upload_result_bundle(self, eng_item_id, bundle_path, title,
                             collab_space=None):
        """Upload an inspection result bundle as a single Document.

        Every file in the bundle directory is checked in against one Document,
        with the manifest first so it is the primary file.

        Returns:
            tuple: (doc_id, message); doc_id is None on failure.
        """
        bundle_path = Path(bundle_path)
        if not bundle_path.exists():
            return None, f'Result bundle not found: {bundle_path}.'

        files = sorted(path for path in bundle_path.rglob('*') if path.is_file())
        manifest = bundle_path / 'result_manifest.json'
        if manifest in files:
            files.remove(manifest)
            files.insert(0, manifest)
        if not files:
            return None, f'Result bundle is empty: {bundle_path}.'

        doc_id, message = self.create_document(
            title, description='ViewpointGeneration inspection result',
            collab_space=collab_space)
        if doc_id is None:
            return None, message

        failures = []
        for path in files:
            success, upload_message = self.upload_file_to_document(doc_id, path)
            if not success:
                failures.append(f'{path.name}: {upload_message}')

        attached, attach_message = self.attach_document_to_item(doc_id, eng_item_id)
        if not attached:
            failures.append(attach_message)

        if failures:
            return doc_id, (f'Uploaded result bundle as {doc_id} with '
                            f'{len(failures)} problem(s): {"; ".join(failures[:3])}')
        return doc_id, (f'Uploaded {len(files)} result file(s) as document {doc_id}.')

    def create_issue(self, eng_item_id, title, description, severity='Medium'):
        """Create an Issue against an EngItem for the NCR workflow.

        Returns:
            tuple: (issue_id, message); issue_id is None on failure.
        """
        success, message = self.ensure_login()
        if not success:
            return None, message

        url = (f"{self.config.space_url.rstrip('/')}"
               '/resources/v1/modeler/dsiss/dsiss:Issue')
        body = {'data': [{'dataelements': {
            'title': title,
            'description': description,
            'severity': severity,
        }}]}
        response = self._request('POST', url, json=body,
                                 headers={'Content-Type': 'application/json'})
        if response.status_code >= 400:
            return None, (f'Issue creation failed ({response.status_code}): '
                          f'{response.text[:300]}')
        try:
            entries = response.json().get('data') or response.json().get('member') or []
        except ValueError:
            return None, f'Issue creation returned a non-JSON body: {response.text[:200]}'
        if not entries:
            return None, 'Issue creation returned no object.'
        issue_id = entries[0].get('id')
        return issue_id, f'Created issue {issue_id} for {eng_item_id}.'

    @staticmethod
    def encode_security_context(security_context):
        """URL-encode a security context for use in a query string.

        `ctx::Role.Org.Space` becomes `ctx%3A%3ARole.Org.Space` with spaces
        percent-encoded. Header values must stay unencoded.
        """
        return urllib.parse.quote(security_context or '', safe='')

    def close(self):
        """Release the underlying HTTP session."""
        self.session.close()
        self.logged_in = False
