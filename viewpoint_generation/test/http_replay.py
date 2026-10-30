"""Record and replay HTTP traffic for backend contract tests.

`RecordingAdapter` sits under a live `requests.Session` and writes every
interaction to a fixture, **sanitized as it is recorded**: tenant ids and
hostnames, collaborative-space, organization and user names, object ids
(mapped consistently to fake ones), FCS tickets, receipts and e-mail
addresses are replaced, request and response headers are dropped (so no
cookie, CSRF token or Authorization header is ever written), and binary
downloads are replaced by a placeholder. Authentication must happen *before*
the adapter is mounted, so the login exchange is never recorded.

`ReplayAdapter` serves a fixture back to a `requests.Session` in tests,
matching on method, path, query (without the tenant parameter) and JSON body.
"""

import io
import json
import re
from collections import defaultdict, deque
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit

import requests
from requests.adapters import BaseAdapter, HTTPAdapter

PLACEHOLDER_STEP = (b'ISO-10303-21;\nHEADER;\n'
                    b"FILE_DESCRIPTION(('replay placeholder'),'2;1');\n"
                    b'ENDSEC;\nDATA;\nENDSEC;\nEND-ISO-10303-21;\n')
FAKE_TENANT = 'R0000000000000'
_HEX32 = re.compile(r'\b[0-9A-F]{32}\b')
_TENANT = re.compile(r'\b[Rr]1\d{11,}\b')
_DS_HOST = re.compile(r'\b[a-z0-9.-]+\.3dexperience\.3ds\.com\b', re.I)
_EMAIL = re.compile(r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b')
_SECRET_KEYS = re.compile(r'"(ticket|receipt|csrf|token)"(\s*:\s*)"([^"]*)"', re.I)
# 3DX echoes the CSRF token in document responses as {"name": ..., "value": ...}.
_CSRF_OBJECT = re.compile(r'"csrf"\s*:\s*\{[^{}]*\}', re.I)
_CSRF_REDACTED = '"csrf": {"name": "ENO_CSRF_TOKEN", "value": "CSRF-REDACTED"}'
# Person-name fields (3DX ownerInfo blocks carry them).
_PERSON = re.compile(r'"(first_?name|last_?name|full_?name|display_?name|given_?name|'
                     r'family_?name)"(\s*:\s*)"([^"]*)"', re.I)
_JOB_TICKET = re.compile(r'(__fcs__jobTicket=)([^&"\s]+)')


class Sanitizer:
    """Consistent replacement of identifying values in recorded traffic."""

    def __init__(self, literals=None):
        # literal -> placeholder, longest first so substrings never win;
        # matched case-insensitively (3DX lower-cases user names in places).
        self.literals = [(re.compile(re.escape(k), re.I), v) for k, v in sorted(
            ((k, v) for k, v in (literals or {}).items() if k), key=lambda kv: -len(kv[0]))]
        self.ids = {}
        self.hosts = {}
        self.secrets = {}

    def _id(self, match):
        value = match.group(0)
        if value not in self.ids:
            self.ids[value] = f'{len(self.ids) + 1:032X}'
        return self.ids[value]

    def _host(self, match):
        value = match.group(0).lower()
        if value not in self.hosts:
            self.hosts[value] = f'host{len(self.hosts) + 1}.example.invalid'
        return self.hosts[value]

    def _secret(self, kind, value):
        key = (kind, value)
        if key not in self.secrets:
            self.secrets[key] = f'{kind.upper()}-{len(self.secrets) + 1}'
        return self.secrets[key]

    def text(self, value):
        if not value:
            return value
        for pattern, placeholder in self.literals:
            value = pattern.sub(placeholder, value)
        value = _CSRF_OBJECT.sub(_CSRF_REDACTED, value)
        value = _PERSON.sub(lambda m: f'"{m.group(1)}"{m.group(2)}"Test User"', value)
        value = _DS_HOST.sub(self._host, value)
        value = _TENANT.sub(FAKE_TENANT, value)
        value = _EMAIL.sub('user@example.invalid', value)
        value = _SECRET_KEYS.sub(
            lambda m: f'"{m.group(1)}"{m.group(2)}"{self._secret(m.group(1), m.group(3))}"',
            value)
        # The same ticket travels raw in JSON and URL-encoded in a query.
        value = _JOB_TICKET.sub(
            lambda m: m.group(1) + self._secret('ticket', unquote(m.group(2))), value)
        return _HEX32.sub(self._id, value)


def _key(method, url, body):
    """Match key: method, path, sorted query without tenant, canonical JSON body."""
    parts = urlsplit(url)
    query = sorted((k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
                   if k != 'tenant')
    canonical = ''
    if body:
        try:
            canonical = json.dumps(json.loads(body), sort_keys=True)
        except (ValueError, TypeError):
            canonical = ''  # multipart and other non-JSON bodies are not matched on
    return f'{method.upper()} {parts.path}?{urlencode(query)} {canonical}'


def _body_text(body):
    if body is None:
        return ''
    if isinstance(body, bytes):
        try:
            return body.decode('utf-8')
        except UnicodeDecodeError:
            return ''
    return str(body)


class RecordingAdapter(HTTPAdapter):
    """Pass requests through to the network and record them, sanitized."""

    def __init__(self, sanitizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sanitizer = sanitizer
        self.interactions = []

    def send(self, request, **kwargs):
        response = super().send(request, **kwargs)
        content_type = response.headers.get('content-type', '')
        content = response.content
        if urlsplit(request.url).path.endswith('/fcs/checkin'):
            # The FCS check-in answers with a bare receipt; it travels on in
            # the completion request's JSON, where it maps to the same value.
            body = self.sanitizer._secret('receipt', content.decode('utf-8', 'replace').strip())
            binary = False
        elif 'json' in content_type or 'text' in content_type or 'html' in content_type:
            body, binary = self.sanitizer.text(content.decode('utf-8', 'replace')), False
        elif content[:12] == b'ISO-10303-21':
            body, binary = PLACEHOLDER_STEP.decode(), True
        elif content:
            body, binary = '', True
        else:
            body, binary = '', False
        request_body = _body_text(request.body)
        is_json = 'json' in (request.headers.get('Content-Type') or '')
        self.interactions.append({
            'method': request.method,
            'url': self.sanitizer.text(request.url),
            'request_json': self.sanitizer.text(request_body) if is_json else '',
            'status': response.status_code,
            'content_type': content_type.split(';')[0],
            'binary': binary,
            'body': body,
        })
        return response


class ReplayAdapter(BaseAdapter):
    """Serve recorded interactions; unmatched requests get a 599."""

    def __init__(self, interactions):
        super().__init__()
        self.queues = defaultdict(deque)
        self.unmatched = []
        for interaction in interactions:
            key = _key(interaction['method'], interaction['url'],
                       interaction.get('request_json'))
            self.queues[key].append(interaction)

    def send(self, request, **kwargs):
        body = _body_text(request.body)
        is_json = 'json' in (request.headers.get('Content-Type') or '')
        key = _key(request.method, request.url, body if is_json else '')
        queue = self.queues.get(key)
        response = requests.Response()
        response.request = request
        response.url = request.url
        if not queue:
            self.unmatched.append(key)
            response.status_code = 599
            response._content = b'no recorded interaction'
            response.headers['content-type'] = 'text/plain'
            return response
        interaction = queue[0] if len(queue) == 1 else queue.popleft()
        response.status_code = interaction['status']
        response.headers['content-type'] = interaction['content_type'] or 'application/json'
        body = interaction['body']
        content = body.encode('utf-8') if isinstance(body, str) else body
        response._content = content
        # Streamed downloads read from raw; give them the same bytes.
        response._content_consumed = True
        response.raw = io.BytesIO(content)
        return response

    def close(self):
        pass
