"""The protocol every PLM backend implements, and the types it trades in.

The catalog core talks to PLM systems only through `PLMBackend`. It never
imports a concrete backend: backends are looked up by name in
`backends.registry`.

Conventions every backend follows:

* `part_id` is the backend's own identifier, opaque outside the backend. The
  core never uses it as a file path component (it may contain '/'); it
  derives a filesystem-safe `storage_key` instead.
* `change_token` is opaque and compared for equality only. When it moves, the
  part's cached geometry and the plans generated against it are stale.
* The core calls a method only when the backend declares the matching
  `Capability`, and otherwise falls back (thumbnails are rendered locally,
  plans stay on the cell, and so on).
* Reads that depend on a service the PLM may not expose degrade to None or an
  empty result and log; a missing service must not fail a whole sync.
* Writes the operator asked for (`attach_file`, `raise_issue`) raise
  `BackendError` with a readable message when they fail, so the reason
  reaches the operator. They return None only when the backend does not
  support the operation at all.
* Credentials live in the backend's own configuration, which never has a
  `to_dict()`, is never a ROS parameter, and is never written into a plan,
  result or log.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Iterator, Mapping, Optional, Protocol, runtime_checkable


class BackendError(RuntimeError):
    """A PLM call failed in a way the caller should report."""


class BackendAuthError(BackendError):
    """The backend could not authenticate."""


class ScopeTruncated(Exception):
    """Raised by `list_scope()` after it has yielded every part it could, when
    it stopped at a budget rather than at the end of the scope.

    The core then reconciles what it saw but archives nothing: parts it did
    not see were never looked at, not removed.
    """

    def __init__(self, scanned, budget, message=''):
        super().__init__(message or f'Scope scan stopped at its {budget}-item budget.')
        self.scanned = scanned
        self.budget = budget


class Capability(str, Enum):
    """Optional things a backend can do."""

    GEOMETRY = 'geometry'                 # produce a STEP file for a part
    ATTRIBUTES = 'attributes'             # engineering attributes (material, ...)
    ATTACH = 'attach'                     # store a file remotely
    LINK_ATTACHMENT = 'link_attachment'   # a stored file is discoverable from the part
    ISSUES = 'issues'                     # raise an issue / NCR
    EVENTS = 'events'                     # push change notifications
    THUMBNAILS = 'thumbnails'             # a real per-part preview image


@dataclass(frozen=True)
class RemotePart:
    """One part as a backend reports it."""

    backend: str                  # registry name of the backend
    part_id: str                  # backend-native id, opaque outside the backend
    title: str
    part_number: Optional[str]
    revision: Optional[str]       # human-facing; may be None
    change_token: str             # opaque; compared for equality only
    maturity: Optional[str]       # IN_WORK, RELEASED, ...; None if no such concept
    container: Optional[str]      # 3DX collaborative space, a folder, ...
    # Whether the backend can produce STEP for this part. None means the
    # listing did not say; the core asks `check_geometry()` when it matters.
    geometry_available: Optional[bool]
    description: str = ''
    extra: Mapping[str, str] = field(default_factory=dict)  # never interpreted by the core


@dataclass(frozen=True)
class RemoteRef:
    """Something a backend created or found remotely (document, file, issue)."""

    backend: str
    ref_id: str
    url: Optional[str] = None
    title: str = ''
    modified: str = ''            # ISO-8601 when known; used to order attachments
    note: str = ''                # partial-success detail worth showing the operator


@runtime_checkable
class PLMBackend(Protocol):
    """What the catalog core needs from a PLM system."""

    name: str
    capabilities: frozenset
    scope: str                    # the scope this backend catalogues by default

    def is_configured(self) -> tuple:
        """(bool, str): whether enough is configured to try connecting."""

    def connect(self) -> None:
        """Authenticate, raising BackendAuthError on failure."""

    def list_scope(self, scope: str) -> Iterator[RemotePart]:
        """Every part in a scope. Raises ScopeTruncated at the end when the
        listing stopped at a budget."""

    def list_recent(self, scope: str) -> Optional[Iterator[RemotePart]]:
        """A cheap listing of recent changes for an incremental pass, or None
        when only a full listing is meaningful."""

    def get_part(self, part_id: str) -> Optional[RemotePart]:
        """One part, or None when it is no longer visible."""

    def check_geometry(self, part: RemotePart) -> bool:
        """Whether STEP can be produced for the part (asked when the listing
        did not say)."""

    def fetch_geometry(self, part: RemotePart, dest_dir: Path) -> Optional[Path]:
        """Write the part's STEP into dest_dir and return its path."""

    def fetch_thumbnail(self, part: RemotePart, dest_dir: Path) -> Optional[Path]:
        """Write a real preview image into dest_dir and return its path."""

    def read_attributes(self, part: RemotePart) -> dict:
        """Engineering attributes as strings."""

    def attach_file(self, part: RemotePart, path: Path, kind: str,
                    title: str) -> Optional[RemoteRef]:
        """Store a file (or every file in a directory, as one attachment)."""

    def list_attachments(self, part: RemotePart, kind: str) -> list:
        """RemoteRefs of attachments of a kind that belong to the part."""

    def fetch_attachment(self, ref: RemoteRef, dest_dir: Path) -> Optional[Path]:
        """Download an attachment into dest_dir and return its path."""

    def raise_issue(self, part: RemotePart, title: str, body: str) -> Optional[RemoteRef]:
        """Raise an issue against the part."""

    def subscribe(self, on_change: Callable[[str], None]) -> Optional[Callable[[], None]]:
        """Start delivering part_ids that changed; returns a stop callable."""

    def describe(self) -> dict:
        """Display-only status (scope details, connection state)."""

    def close(self) -> None:
        """Release network resources."""


def part_from_row(row):
    """A RemotePart rebuilt from a catalog `parts` row, for backend calls."""
    try:
        extra = json.loads(row.get('extra_json') or '{}')
    except ValueError:
        extra = {}
    available = row.get('geometry_available')
    return RemotePart(
        backend=row['backend'], part_id=row['part_id'], title=row.get('title') or '',
        part_number=row.get('part_number'), revision=row.get('revision'),
        change_token=row.get('change_token') or '', maturity=row.get('maturity'),
        container=row.get('container'),
        geometry_available=None if available is None else bool(available),
        description=row.get('description') or '', extra=extra)
