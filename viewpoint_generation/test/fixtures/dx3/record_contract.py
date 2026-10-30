"""Record the dx3 backend contract fixture from a live 3DX tenant.

    PYTHONPATH=.:test python3 test/fixtures/dx3/record_contract.py

Needs the DX_* environment variables and network access. Exercises the
backend protocol read paths over the configured scope, uploads ONE small
Document titled TEST_PLM_BRIDGE_CONTRACT_PLAN and downloads it back. It never
creates Issues and never relates documents to items. The fixture is
sanitized as it is recorded (see http_replay.Sanitizer); the id of the
created Document is printed so it can be logged, and is not stored.
"""

import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from http_replay import RecordingAdapter, Sanitizer

from viewpoint_generation.catalog.backends.dx3.backend import DX3Backend
from viewpoint_generation.catalog.backends.dx3 import client as client_module
from viewpoint_generation.catalog.backends.dx3.client import DXClient
from viewpoint_generation.catalog.backends.dx3.config import DX3Settings, DXConfig

FIXTURE = Path(__file__).with_name('contract.json')
TITLE = 'TEST_PLM_BRIDGE_CONTRACT_PLAN'
UNKNOWN_ID = 'F' * 32
CONTENT = '{"contract_test": true}\n'


def literals(config, settings):
    """Identifying values from the environment, mapped to placeholders."""
    values = {}
    for url, placeholder in ((config.space_url, 'space.example.invalid'),
                             (config.passport_url, 'passport.example.invalid')):
        host = urlsplit(url).hostname
        if host:
            values[host] = placeholder
    context = config.security_context.replace('ctx::', '')
    role, _, rest = context.partition('.')
    organization, _, space = rest.partition('.')
    values[organization] = 'Test Org'
    for name in {space, settings.collab_space_filter}:
        if name:
            values[name] = 'Test Space'
            # A personal collaborative space is usually '<owner> Space'; its
            # owner's names also appear on their own in ownerInfo blocks.
            if name.endswith(' Space'):
                owner = name[:-len(' Space')]
                values[owner] = 'Test User'
                for token in owner.split():
                    if len(token) > 2:
                        values[token] = 'Test'
    if config.username:
        values[config.username] = 'tester'
    if config.tenant:
        values[config.tenant] = 'R0000000000000'
        values[config.tenant.lower()] = 'r0000000000000'
    return values


def main():
    config, settings = DXConfig.from_env(), DX3Settings.from_env()
    configured, message = config.is_configured()
    if not configured:
        print(message)
        return 2
    client = DXClient(config)
    ok, message = client.ensure_login()      # before recording: never captured
    if not ok:
        print(message)
        return 1

    sanitizer = Sanitizer(literals(config, settings))
    adapter = RecordingAdapter(sanitizer)
    client.session.mount('https://', adapter)
    # The FCS upload is sent with requests.post, outside the session (the FCS
    # host authenticates by job ticket); route it through the recorder too.
    client_module.requests.post = client.session.post
    backend = DX3Backend(config, settings, client)

    parts = list(backend.list_scope(backend.scope))
    backend.list_recent(backend.scope)
    with_geometry = next(p for p in parts if backend.check_geometry(p))
    without_geometry = next((p for p in parts if not backend.check_geometry(p)), None)
    backend.get_part(with_geometry.part_id)
    backend.get_part(UNKNOWN_ID)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        backend.fetch_geometry(with_geometry, tmp / 'geometry')
        if without_geometry is not None:
            backend.fetch_geometry(without_geometry, tmp / 'none')
        backend.fetch_thumbnail(with_geometry, tmp / 'thumb')
        plan = tmp / 'contract_plan.json'
        plan.write_text(CONTENT)
        ref = backend.attach_file(with_geometry, plan, 'plan', TITLE)
        print(f'CREATED Document {ref.ref_id} titled {TITLE!r} '
              f'at {datetime.now(timezone.utc).isoformat()}')
        backend.fetch_attachment(ref, tmp / 'download')
        settings.document_links = True
        backend.list_attachments(with_geometry, 'plan')
        settings.document_links = False

    fixture = {
        'recorded_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'notes': 'Recorded by record_contract.py against a live tenant; sanitized '
                 'at record time. Object ids are consistent fakes.',
        'scope': sanitizer.text(backend.scope),
        'space_url': 'https://space.example.invalid/enovia',
        'parts': [sanitizer.text(p.part_id) for p in parts],
        'change_tokens': {sanitizer.text(p.part_id): sanitizer.text(p.change_token)
                          for p in parts},
        'with_geometry': sanitizer.text(with_geometry.part_id),
        'without_geometry': sanitizer.text(without_geometry.part_id)
        if without_geometry else None,
        'unknown': sanitizer.text(UNKNOWN_ID),
        'attachment_ref': sanitizer.text(ref.ref_id),
        'attachment_title': TITLE,
        'attachment_content': CONTENT,
        'interactions': adapter.interactions,
    }
    FIXTURE.write_text(json.dumps(fixture, indent=1, sort_keys=True))
    print(f'Wrote {FIXTURE} ({len(adapter.interactions)} interactions).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
