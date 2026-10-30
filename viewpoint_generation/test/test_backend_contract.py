"""The backend contract: every PLMBackend must pass these.

Parametrized over the in-memory fake and the dx3 backend replaying HTTP
traffic recorded from a live 3DX tenant (`fixtures/dx3/contract.json`,
sanitized at record time; re-record with `fixtures/dx3/record_contract.py`).
A new backend joins by adding a case to `contract_case`.

dx3 Issue creation is not in the recording: creating an Issue on the tenant
is outside what the test runs may do, so that case is skipped for dx3.
"""

import json
from pathlib import Path

import pytest

from http_replay import ReplayAdapter

from viewpoint_generation.catalog.backends.base import (Capability, PLMBackend, RemotePart,
                                                        RemoteRef)
from viewpoint_generation.catalog.backends.dx3 import client as client_module
from viewpoint_generation.catalog.backends.dx3.backend import DX3Backend
from viewpoint_generation.catalog.backends.dx3.client import DXClient
from viewpoint_generation.catalog.backends.dx3.config import DX3Settings, DXConfig
from viewpoint_generation.catalog.backends.fake import FakeBackend

FIXTURE = Path(__file__).parent / 'fixtures' / 'dx3' / 'contract.json'
CONTENT = '{"contract_test": true}\n'


def fake_case():
    backend = FakeBackend()
    backend.add_part('P-STEP', change_token='t1')
    backend.add_part('P-NONE', change_token='t2', geometry=None)
    return backend, {'scope': '*', 'parts': ['P-STEP', 'P-NONE'], 'with_geometry': 'P-STEP',
                     'without_geometry': 'P-NONE', 'unknown': 'missing',
                     'attach_title': 'TEST_PLM_BRIDGE_CONTRACT_PLAN', 'issues': True}


def dx3_case(monkeypatch):
    fixture = json.loads(FIXTURE.read_text())
    config = DXConfig(passport_url='https://passport.example.invalid',
                      space_url=fixture['space_url'], tenant='R0000000000000',
                      username='tester', password='not-a-secret',
                      security_context='ctx::VPLMProjectLeader.Test Org.Test Space')
    client = DXClient(config)
    adapter = ReplayAdapter(fixture['interactions'])
    client.session.mount('https://', adapter)
    client.logged_in, client.csrf_token = True, 'CSRF-REDACTED'
    # The FCS upload is sent with requests.post outside the session.
    monkeypatch.setattr(client_module.requests, 'post', client.session.post)
    settings = DX3Settings(bookmark_scope=fixture['scope'], collab_space_filter='Test Space')
    backend = DX3Backend(config, settings, client)
    backend.replay = adapter
    return backend, {'scope': fixture['scope'], 'parts': fixture['parts'],
                     'with_geometry': fixture['with_geometry'],
                     'without_geometry': fixture['without_geometry'],
                     'unknown': fixture['unknown'],
                     'attach_title': fixture['attachment_title'], 'issues': False}


@pytest.fixture(params=['fake', 'dx3'])
def contract_case(request, monkeypatch):
    if request.param == 'dx3':
        return dx3_case(monkeypatch)
    return fake_case()


def _part(backend, sample, key='with_geometry'):
    return next(p for p in backend.list_scope(sample['scope']) if p.part_id == sample[key])


def test_satisfies_the_protocol(contract_case):
    backend, _ = contract_case
    assert isinstance(backend, PLMBackend)
    assert isinstance(backend.name, str) and backend.name
    assert all(isinstance(c, Capability) for c in backend.capabilities)
    assert backend.is_configured()[0] is True
    backend.connect()
    assert isinstance(backend.describe(), dict)


def test_list_scope_yields_complete_remote_parts(contract_case):
    backend, sample = contract_case
    parts = list(backend.list_scope(sample['scope']))
    assert sorted(p.part_id for p in parts) == sorted(sample['parts'])
    for part in parts:
        assert isinstance(part, RemotePart)
        assert part.backend == backend.name
        assert isinstance(part.title, str) and part.title
        assert isinstance(part.change_token, str) and part.change_token
        assert part.geometry_available in (True, False, None)


def test_get_part_matches_the_listing(contract_case):
    backend, sample = contract_case
    listed = _part(backend, sample)
    fetched = backend.get_part(sample['with_geometry'])
    assert fetched.part_id == listed.part_id
    assert fetched.change_token == listed.change_token
    assert backend.get_part(sample['unknown']) is None


def test_geometry(contract_case, tmp_path):
    backend, sample = contract_case
    assert Capability.GEOMETRY in backend.capabilities
    part = _part(backend, sample)
    assert backend.check_geometry(part) is True
    path = backend.fetch_geometry(part, tmp_path / 'with')
    assert path is not None and path.read_bytes().startswith(b'ISO-10303-21')
    if sample['without_geometry']:
        none = _part(backend, sample, 'without_geometry')
        assert backend.check_geometry(none) is False
        assert backend.fetch_geometry(none, tmp_path / 'none') is None


def test_optional_reads_degrade_to_empty(contract_case, tmp_path):
    backend, sample = contract_case
    part = _part(backend, sample)
    thumbnail = backend.fetch_thumbnail(part, tmp_path)
    assert thumbnail is None or thumbnail.exists()
    assert isinstance(backend.read_attributes(part), dict)
    assert isinstance(backend.list_attachments(part, 'plan'), list)


def test_attach_and_read_back(contract_case, tmp_path):
    backend, sample = contract_case
    assert Capability.ATTACH in backend.capabilities
    plan = tmp_path / 'contract_plan.json'
    plan.write_text(CONTENT)
    ref = backend.attach_file(_part(backend, sample), plan, 'plan', sample['attach_title'])
    assert isinstance(ref, RemoteRef) and ref.backend == backend.name and ref.ref_id
    path = backend.fetch_attachment(ref, tmp_path / 'download')
    assert path is not None and path.read_text() == CONTENT


def test_link_lookup_on_a_tenant_without_it_returns_nothing(contract_case):
    backend, sample = contract_case
    if backend.name == 'dx3':
        # The relationship service answers 404 on the recorded tenant; with
        # document_links on, the lookup must still degrade to an empty list.
        backend.settings.document_links = True
    assert isinstance(backend.list_attachments(_part(backend, sample), 'plan'), list)


def test_issues(contract_case):
    backend, sample = contract_case
    if not sample['issues']:
        pytest.skip('Issue creation is not recorded (tenant hygiene: no Issues).')
    ref = backend.raise_issue(_part(backend, sample), 'TEST_PLM_BRIDGE issue', 'body')
    assert isinstance(ref, RemoteRef) and ref.ref_id


def test_events_only_with_the_capability(contract_case):
    backend, _ = contract_case
    stop = backend.subscribe(lambda part_id: None)
    if Capability.EVENTS in backend.capabilities:
        assert callable(stop)
        stop()
    else:
        assert stop is None


def test_replay_served_every_request(contract_case):
    """Guards the dx3 case against silently answering from nothing."""
    backend, sample = contract_case
    list(backend.list_scope(sample['scope']))
    if backend.name == 'dx3':
        assert backend.replay.unmatched == []
