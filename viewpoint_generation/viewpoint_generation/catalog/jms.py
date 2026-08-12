"""Event-driven catalog sync over a JMS/STOMP bridge.

3DEXPERIENCE publishes platform events (object modified, maturity changed,
bookmark content added, derived output created) on an internal JMS bus. Cloud
tenants cannot reach that bus directly: Dassault's Enterprise Integration
Framework (EIF) relays the events to an external broker -- RabbitMQ or ActiveMQ
-- which this listener subscribes to over STOMP.

The payoff is latency. Polling on `CATALOG_SYNC_INTERVAL` means a part added to
the catalog in 3DX shows up in the picker minutes later, and each poll
re-enumerates the whole scope. An event names the object that changed, so
`sync_single_item()` touches exactly one row.

This is strictly an optimization: when `DX_JMS_BROKER_URL` is unset, or the
broker is unreachable, the catalog node keeps its polling timer and loses
nothing but freshness. `JMSEventListener.start()` reports whether it connected
so the caller can decide which mode it is in.

Broker-side configuration needed in production is documented in
`docs/3dx_integration/README_3dx_integration.md`.
"""

import json
import logging
import threading
import time
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Platform event topics this listener cares about, mapped to the handler names
# used below. Destination names are broker-side configuration -- EIF is free to
# publish under any prefix -- so they are overridable per deployment.
DEFAULT_TOPICS = {
    'engitem.modified': '/topic/ds.enovia.engitem.modified',
    'maturity.changed': '/topic/ds.enovia.engitem.maturity.changed',
    'bookmark.content.added': '/topic/ds.enovia.bookmark.content.added',
    'derivedoutput.created': '/topic/ds.enovia.derivedoutput.created',
}


class JMSEventListener:
    """Subscribes to 3DX platform events and drives targeted catalog syncs.

    Args:
        broker_url: STOMP URL, e.g. `stomp://user:pass@broker:61613`.
        catalog_sync: The `CatalogSync` to drive.
        topics: Optional mapping of event name -> broker destination,
            overriding DEFAULT_TOPICS.
        on_event: Optional callable invoked as `on_event(event_name, body)`
            after each handled event, used by the picker's live updates.
    """

    def __init__(self, broker_url, catalog_sync, topics=None, on_event=None):
        self.broker_url = broker_url
        self.catalog = catalog_sync
        self.topics = dict(topics or DEFAULT_TOPICS)
        self.on_event = on_event
        self.connection = None
        self.connected = False
        self.running = False
        self._thread = None
        self._stop = threading.Event()
        # Reconnect backoff, doubling to a ceiling so a broker outage does not
        # turn into a reconnect storm.
        self._backoff = 1.0
        self._max_backoff = 60.0
        self.events_handled = 0
        self.last_error = ''

    # --- lifecycle ---------------------------------------------------------

    def start(self):
        """Connect and begin listening on a background thread.

        Returns:
            tuple: (bool, str) whether the listener started, and a message.
        """
        if not self.broker_url:
            return False, 'No JMS broker configured; catalog will poll instead.'
        try:
            import stomp  # noqa: F401 - probed here so the caller gets a clear reason
        except ImportError:
            return False, ('The stomp.py package is not installed; '
                           'catalog will poll instead.')

        self.running = True
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_forever, name='catalog-jms', daemon=True)
        self._thread.start()

        # Give the first connection attempt a moment so start() can report
        # honestly whether event-driven sync is actually live.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not self.connected and self.running:
            time.sleep(0.1)
        if self.connected:
            return True, f'Listening for 3DX events on {self._safe_url()}.'
        return False, (f'Could not reach the JMS broker at {self._safe_url()}'
                       f'{": " + self.last_error if self.last_error else ""}; '
                       'catalog will poll instead.')

    def stop(self):
        """Disconnect and stop the listener thread."""
        self.running = False
        self._stop.set()
        if self.connection is not None:
            try:
                self.connection.disconnect()
            except Exception:  # noqa: BLE001 - disconnect races are not actionable
                pass
        self.connected = False

    def _safe_url(self):
        """The broker URL with any password removed, for logging."""
        parsed = urlparse(self.broker_url)
        if parsed.password:
            netloc = parsed.netloc.replace(f':{parsed.password}@', ':***@')
            return f'{parsed.scheme}://{netloc}'
        return self.broker_url

    def _parse_broker_url(self):
        """Split the broker URL into host, port, and credentials."""
        parsed = urlparse(self.broker_url)
        return {
            'host': parsed.hostname or 'localhost',
            'port': parsed.port or 61613,
            'username': parsed.username,
            'password': parsed.password,
        }

    def _run_forever(self):
        """Connect, and keep reconnecting for as long as the listener runs."""
        while self.running and not self._stop.is_set():
            try:
                self.connect()
                self._backoff = 1.0
                # stomp.py delivers messages on its own receiver thread, so
                # this one only has to stay alive and watch the connection.
                while self.running and not self._stop.is_set():
                    if self.connection is not None and not self.connection.is_connected():
                        raise ConnectionError('STOMP connection dropped.')
                    self._stop.wait(2.0)
            except Exception as e:  # noqa: BLE001 - any failure means reconnect
                self.connected = False
                self.last_error = str(e)
                if not self.running:
                    break
                logger.warning('JMS listener error (%s); reconnecting in %.0fs.',
                               e, self._backoff)
                self._stop.wait(self._backoff)
                self._backoff = min(self._backoff * 2, self._max_backoff)

    def connect(self):
        """Open the STOMP connection and subscribe to the event topics."""
        import stomp

        settings = self._parse_broker_url()
        self.connection = stomp.Connection(
            [(settings['host'], settings['port'])],
            heartbeats=(10000, 10000))
        self.connection.set_listener('catalog', _StompListener(self))
        self.connection.connect(
            settings['username'], settings['password'], wait=True)
        for index, (event, destination) in enumerate(self.topics.items()):
            self.connection.subscribe(destination=destination, id=str(index + 1),
                                      ack='auto')
            logger.debug('Subscribed to %s for %s', destination, event)
        self.connected = True
        logger.info('JMS listener connected to %s', self._safe_url())

    # --- event handling ----------------------------------------------------

    def _event_for_destination(self, destination):
        """Map a broker destination back to the event name it carries."""
        for event, topic in self.topics.items():
            if topic == destination:
                return event
        # Some brokers rewrite destinations; fall back to a suffix match.
        for event in self.topics:
            if destination and destination.endswith(event):
                return event
        return destination or 'unknown'

    def on_message(self, destination, body):
        """Handle one platform event.

        Args:
            destination: The broker destination the message arrived on.
            body: The raw message body (JSON text or an already-parsed dict).

        Returns:
            tuple: (bool, str) whether the event was acted on, and a message.
        """
        if isinstance(body, (str, bytes)):
            try:
                payload = json.loads(body)
            except (ValueError, TypeError):
                return False, f'Ignoring non-JSON event body on {destination}.'
        else:
            payload = body or {}

        event = self._event_for_destination(destination)
        object_id = (payload.get('objectId') or payload.get('physicalid')
                     or payload.get('id'))

        handled, message = self._dispatch(event, object_id, payload)
        if handled:
            self.events_handled += 1
            if self.on_event is not None:
                try:
                    self.on_event(event, payload)
                except Exception as e:  # noqa: BLE001 - a bad hook must not kill the listener
                    logger.warning('JMS event hook failed: %s', e)
        logger.info('JMS %s (%s): %s', event, object_id, message)
        return handled, message

    def _dispatch(self, event, object_id, payload):
        """Route one event to the catalog action it implies."""
        if event in ('engitem.modified', 'bookmark.content.added'):
            if not object_id:
                return False, 'Event carried no object id.'
            return self.catalog.sync_single_item(object_id)

        if event == 'maturity.changed':
            if not object_id:
                return False, 'Event carried no object id.'
            new_state = (payload.get('newState') or payload.get('state') or '').upper()
            states = self.catalog.config.maturity_states()
            if states and new_state and new_state not in states:
                # The part left the configured maturity scope; reconciling it
                # marks it archived rather than leaving a stale row behind.
                return self.catalog.sync_single_item(object_id)
            return self.catalog.sync_single_item(object_id)

        if event == 'derivedoutput.created':
            parent_id = (payload.get('parentId') or payload.get('parentObjectId')
                         or object_id)
            if not parent_id:
                return False, 'Derived-output event carried no parent id.'
            if self.catalog.db.get_part(parent_id) is None:
                return False, f'{parent_id} is not in the catalog; ignoring.'
            self.catalog.db.set_step_available(parent_id, True)
            self.catalog.db.log_sync('derived_output', parent_id,
                                     payload.get('format', ''))
            return True, f'Marked {parent_id} as having a derived output.'

        return False, f'No handler for event {event!r}.'

    def status(self):
        """Listener state, for the catalog status endpoint."""
        return {
            'enabled': bool(self.broker_url),
            'connected': self.connected,
            'broker': self._safe_url() if self.broker_url else '',
            'events_handled': self.events_handled,
            'last_error': self.last_error,
        }


class _StompListener:
    """Adapts stomp.py's listener callbacks onto JMSEventListener."""

    def __init__(self, owner):
        self.owner = owner

    def on_message(self, frame):
        destination = (frame.headers or {}).get('destination', '')
        try:
            self.owner.on_message(destination, frame.body)
        except Exception as e:  # noqa: BLE001 - never let a bad event kill the receiver
            logger.error('Failed to handle JMS message on %s: %s', destination, e)

    def on_error(self, frame):
        self.owner.last_error = str(getattr(frame, 'body', frame))
        logger.error('JMS broker error: %s', self.owner.last_error)

    def on_disconnected(self):
        self.owner.connected = False
        logger.warning('JMS broker disconnected.')
