"""An in-memory backend for tests and development.

Holds parts, geometry bytes, attachments and issues in dictionaries, and can
be told to declare any subset of capabilities, so the core's capability
fallbacks and the backend contract tests have something deterministic to run
against. Nothing here touches the network.
"""

import itertools
import logging
from dataclasses import replace
from pathlib import Path

from viewpoint_generation.catalog.backends.base import (BackendError, Capability,
                                                        RemotePart, RemoteRef,
                                                        ScopeTruncated)

logger = logging.getLogger(__name__)

ALL_CAPABILITIES = frozenset(Capability)


class FakeBackend:
    """A PLM that lives in memory.

    Args:
        name: Registry name to report (lets a test run two at once).
        capabilities: Capabilities to declare; default all.
        scope_budget: Stop list_scope after this many parts and raise
            ScopeTruncated, as a budgeted remote scan would.
    """

    def __init__(self, name='fake', capabilities=ALL_CAPABILITIES, scope_budget=None):
        self.name = name
        self.capabilities = frozenset(capabilities)
        self.scope = '*'
        self.scope_budget = scope_budget
        self.parts = {}
        self.geometry = {}
        self.thumbnails = {}
        self.attributes = {}
        self.attachments = []
        self.issues = []
        self.fail_writes = False
        self.connected = False
        self.configured = True
        self.listener = None
        self._ids = itertools.count(1)

    # --- seeding (tests) -------------------------------------------------------

    def add_part(self, part_id, change_token='t1', geometry=b'ISO-10303-21;\nEND-ISO-10303-21;\n',
                 **fields):
        """Add or replace a part; returns the RemotePart."""
        part = RemotePart(
            backend=self.name, part_id=part_id, title=fields.pop('title', f'Part {part_id}'),
            part_number=fields.pop('part_number', f'PN-{part_id}'),
            revision=fields.pop('revision', 'A'), change_token=change_token,
            maturity=fields.pop('maturity', 'IN_WORK'),
            container=fields.pop('container', 'Fake Space'),
            geometry_available=fields.pop('geometry_available', geometry is not None),
            description=fields.pop('description', ''), extra=fields.pop('extra', {}))
        if fields:
            raise TypeError(f'Unknown part fields: {sorted(fields)}')
        self.parts[part_id] = part
        if geometry is None:
            self.geometry.pop(part_id, None)
        else:
            self.geometry[part_id] = geometry
        return part

    def touch(self, part_id, change_token):
        """Change a part remotely (a new CAD revision)."""
        self.parts[part_id] = replace(self.parts[part_id], change_token=change_token)

    def emit(self, part_id):
        """Deliver a change event, as a push-capable PLM would."""
        if self.listener is not None:
            self.listener(part_id)

    # --- protocol -------------------------------------------------------------

    def is_configured(self):
        return (True, 'Fake backend.') if self.configured else (False, 'Fake backend disabled.')

    def connect(self):
        self.connected = True

    def list_scope(self, scope):
        for scanned, part in enumerate(list(self.parts.values()), 1):
            yield part
            if self.scope_budget is not None and scanned >= self.scope_budget \
                    and scanned < len(self.parts):
                raise ScopeTruncated(scanned, self.scope_budget)

    def list_recent(self, scope):
        return None

    def get_part(self, part_id):
        return self.parts.get(part_id)

    def check_geometry(self, part):
        return part.part_id in self.geometry

    def fetch_geometry(self, part, dest_dir):
        data = self.geometry.get(part.part_id)
        if data is None:
            return None
        dest = Path(dest_dir) / 'download.stp'
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return dest

    def fetch_thumbnail(self, part, dest_dir):
        data = self.thumbnails.get(part.part_id)
        if data is None:
            return None
        dest = Path(dest_dir) / 'remote_thumbnail.png'
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return dest

    def read_attributes(self, part):
        return dict(self.attributes.get(part.part_id, {}))

    def attach_file(self, part, path, kind, title):
        if Capability.ATTACH not in self.capabilities:
            return None
        if self.fail_writes:
            raise BackendError('Fake backend refused the upload.')
        path = Path(path)
        files = sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else [path]
        ref = RemoteRef(backend=self.name, ref_id=f'ATT{next(self._ids)}', title=title,
                        modified=f'2026-01-01T00:00:{len(self.attachments):02d}Z')
        self.attachments.append({'part_id': part.part_id, 'kind': kind, 'ref': ref,
                                 'files': {p.name: p.read_bytes() for p in files}})
        return ref

    def list_attachments(self, part, kind):
        if Capability.LINK_ATTACHMENT not in self.capabilities:
            return []
        return [a['ref'] for a in self.attachments
                if a['part_id'] == part.part_id and a['kind'] == kind]

    def fetch_attachment(self, ref, dest_dir):
        for attachment in self.attachments:
            if attachment['ref'].ref_id == ref.ref_id:
                name, data = next(iter(attachment['files'].items()))
                dest = Path(dest_dir) / f'{ref.ref_id}_{name}'
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                return dest
        return None

    def raise_issue(self, part, title, body):
        if Capability.ISSUES not in self.capabilities:
            return None
        if self.fail_writes:
            raise BackendError('Fake backend refused the issue.')
        ref = RemoteRef(backend=self.name, ref_id=f'ISS{next(self._ids)}', title=title)
        self.issues.append({'part_id': part.part_id, 'title': title, 'body': body, 'ref': ref})
        return ref

    def subscribe(self, on_change):
        if Capability.EVENTS not in self.capabilities:
            return None
        self.listener = on_change

        def stop():
            self.listener = None
        return stop

    def describe(self):
        return {'scope': self.scope, 'parts': len(self.parts)}

    def close(self):
        self.connected = False


BACKEND_CLASS = FakeBackend


def from_env():
    """The registry factory: an empty fake backend."""
    return FakeBackend()
