"""Configuration for the 3DEXPERIENCE (dx3) backend.

Two dataclasses, both loadable from the `DX_*` environment variables the cell
has always used (the container gets them from the host's `.env`):

    DXConfig    -- tenant URLs and credentials for the REST client. Holds a
                   password, so it deliberately has NO `to_dict()`: it must
                   never become a ROS parameter or be written into a plan,
                   result or log. Use `redacted()` when logging.
    DX3Settings -- scope and behaviour, no secrets. Follows the config
                   dataclass convention (`to_dict()` returning
                   `{field: {value, type, description, control, range}}`), so
                   the node declares each field as a `dx3.<field>` parameter.
"""

import os
from dataclasses import dataclass


def _env(name, default=''):
    """Read an environment variable, treating unset and empty as the default."""
    value = os.environ.get(name, '')
    return value if value != '' else default


def _env_int(name, default):
    """Read an integer environment variable, falling back on a bad value."""
    try:
        return int(_env(name, default))
    except ValueError:
        return default


def _env_float(name, default):
    """Read a float environment variable, falling back on a bad value."""
    try:
        return float(_env(name, default))
    except ValueError:
        return default


def _env_flag(name, default):
    """Read a boolean environment variable ('0'/'false' are false)."""
    raw = _env(name, '')
    if raw == '':
        return default
    return raw not in ('0', 'false', 'False', 'no')


@dataclass
class DXConfig:
    """Connection parameters for a 3DEXPERIENCE tenant.

    The passport (IAM) and space (3DSpace) hosts are frequently in *different*
    regions on cloud tenants -- one deployment authenticates against `eu1` and
    calls data services on `usw2`. Authentication is a CAS service-ticket
    redirect, so both URLs are needed and neither can be derived from the other.
    """

    passport_url: str = ''
    space_url: str = ''
    tenant: str = ''
    username: str = ''
    password: str = ''
    # Full SecurityContext triplet INCLUDING the 'ctx::' prefix, e.g.
    # 'ctx::VPLMProjectLeader.Organization.Collab Space'. Sent unencoded as an
    # HTTP header; url-encoded only when placed in a query string.
    security_context: str = ''
    # Optional STOMP/AMQP bridge for event-driven sync. When empty, the
    # catalog falls back to timer-based polling.
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
class DX3Settings:
    """Scope and behaviour of the dx3 backend. No secrets."""

    # Identifies the parts in scope. When it matches the title of a 3DX
    # bookmark, that bookmark's contents ARE the catalog -- the curated set the
    # operator maintains in 3DX. Otherwise it is used as a search string
    # ($searchStr), where '*' matches everything the security context can see.
    bookmark_scope: str = '*'
    # Resolve bookmark_scope against 3DX bookmarks before treating it as a
    # search string. Turn off to force search behaviour even when a bookmark of
    # that name exists.
    use_bookmark_scope: bool = True
    # Restrict search results to one collaborative space. Pushed to the
    # server as the [ds6w:project] predicate. Not applied to a bookmark scope,
    # which may legitimately span spaces.
    collab_space_filter: str = ''
    # Maximum items pulled per search request, and overall.
    page_size: int = 100
    max_items: int = 1000
    # Look for a real per-object preview image on new or modified parts. Only
    # real previews are kept; the platform's generic per-type icons are
    # skipped, since they are identical for every part.
    fetch_thumbnails: bool = True
    # Relate uploaded Documents to their engineering item, and look for plans
    # other cells uploaded through that relationship. The relationship
    # service is not exposed on every tenant (the UW tenant answers 404), so
    # this is off unless the tenant is known to support it.
    document_links: bool = False

    @classmethod
    def from_env(cls):
        """Build DX3Settings from the DX_*/CATALOG_* environment variables."""
        return cls(
            bookmark_scope=_env('DX_BOOKMARK_SCOPE', '*'),
            use_bookmark_scope=_env_flag('DX_USE_BOOKMARK_SCOPE', True),
            collab_space_filter=_env('DX_COLLAB_SPACE', ''),
            page_size=_env_int('CATALOG_PAGE_SIZE', 100),
            max_items=_env_int('CATALOG_MAX_ITEMS', 1000),
            fetch_thumbnails=_env_flag('CATALOG_FETCH_THUMBNAILS', True),
            document_links=_env_flag('DX_DOCUMENT_LINKS', False),
        )

    def to_dict(self):
        return {
            'bookmark_scope': {
                'value': self.bookmark_scope,
                'type': 'string',
                'description': ('Catalog scope: the title of a 3DX bookmark whose contents '
                                "are the catalog, or a search string ('*' = everything visible)"),
                'control': 'text',
            },
            'use_bookmark_scope': {
                'value': self.use_bookmark_scope,
                'type': 'boolean',
                'description': ('Resolve the scope against 3DX bookmarks before falling back '
                                'to treating it as a search string'),
                'control': 'toggle',
            },
            'collab_space_filter': {
                'value': self.collab_space_filter,
                'type': 'string',
                'description': ('Keep only search results in this collaborative space '
                                '(empty = all readable spaces)'),
                'control': 'text',
            },
            'page_size': {
                'value': self.page_size,
                'type': 'integer',
                'description': 'Items requested per 3DX search page',
                'control': 'slider',
                'range': [1, 500],
            },
            'max_items': {
                'value': self.max_items,
                'type': 'integer',
                'description': 'Maximum items pulled in one full sync',
                'control': 'slider',
                'range': [1, 10000],
            },
            'fetch_thumbnails': {
                'value': self.fetch_thumbnails,
                'type': 'boolean',
                'description': ('Look for real preview images on 3DX during sync (generic '
                                'per-type icons are always skipped)'),
                'control': 'toggle',
            },
            'document_links': {
                'value': self.document_links,
                'type': 'boolean',
                'description': ('Relate uploaded Documents to their engineering item and '
                                'discover other cells\' plans through it (needs a tenant '
                                'that exposes the relationship service)'),
                'control': 'toggle',
            },
        }
