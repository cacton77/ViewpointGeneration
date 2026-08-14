"""Flask part picker UI for the 3DEXPERIENCE catalog.

Serves a card grid of catalogued parts with thumbnails, cache state, and plan
state, and turns a click into the full selection flow: fetch the STEP, resolve
the inspection plan, and announce the part on `/catalog/part_selected` so the
viewpoint generation node loads it.

**How this talks to ROS.** The app owns a small rclpy node of its own and
publishes `PartSelected` directly, rather than calling the catalog node's
`SelectPart` service. Two reasons: a Flask worker thread calling a ROS service
synchronously is a deadlock waiting to happen, and the catalog itself is a
plain SQLite database plus an HTTP client -- the picker can use `CatalogSync`
in-process and get the same result with far less machinery. When rclpy is
unavailable or ROS is not running, the UI still works completely; it just
reports that the selection was not announced.

Run standalone (no ROS):

    python -m viewpoint_generation.picker.app --port 5050

or as the `picker_node` executable inside the ROS graph.
"""

import argparse
import logging
import os
import threading

from flask import Flask, abort, jsonify, render_template, request, send_file

from viewpoint_generation.catalog.client import DXClient
from viewpoint_generation.catalog.config import CatalogPaths, DXConfig, SyncConfig
from viewpoint_generation.catalog.sync import CatalogSync, PlanStatus

logger = logging.getLogger(__name__)


class PickerBridge:
    """Owns the catalog and, when available, a ROS publisher for selections."""

    def __init__(self, catalog, enable_ros=True):
        self.catalog = catalog
        self.node = None
        self.publisher = None
        self._executor_thread = None
        self._sync_lock = threading.Lock()
        self.last_error = ''
        if enable_ros:
            self._start_ros()

    def _start_ros(self):
        """Bring up a minimal rclpy node for publishing part selections.

        Failure here is never fatal -- the picker is useful for browsing and
        caching even with no ROS graph running.
        """
        try:
            import rclpy
            from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                                   ReliabilityPolicy)
            from viewpoint_generation_interfaces.msg import PartSelected

            if not rclpy.ok():
                rclpy.init()
            self.node = rclpy.create_node('picker')
            self._part_selected_type = PartSelected
            self.publisher = self.node.create_publisher(
                PartSelected, '/catalog/part_selected',
                QoSProfile(depth=1,
                           history=HistoryPolicy.KEEP_LAST,
                           reliability=ReliabilityPolicy.RELIABLE,
                           durability=DurabilityPolicy.TRANSIENT_LOCAL))

            def spin():
                try:
                    rclpy.spin(self.node)
                except Exception:  # noqa: BLE001 - shutdown races are expected
                    pass

            self._executor_thread = threading.Thread(
                target=spin, name='picker-ros', daemon=True)
            self._executor_thread.start()
            logger.info('Picker connected to ROS; selections will be published.')
        except Exception as e:  # noqa: BLE001 - any ROS problem is non-fatal here
            logger.warning('ROS unavailable (%s); the picker will run '
                           'without publishing selections.', e)
            self.node = None
            self.publisher = None

    def publish_selection(self, part, step_path, plan_status, plan_path, units='mm'):
        """Announce a part selection on /catalog/part_selected.

        Returns:
            tuple: (bool, str) whether the message was published, and why not.
        """
        if self.publisher is None:
            return False, 'ROS is not available; selection was not announced.'
        msg = self._part_selected_type()
        msg.eng_item_id = part['eng_item_id']
        msg.title = part.get('title') or ''
        msg.part_number = part.get('part_number') or ''
        msg.revision = part.get('revision') or ''
        msg.cestamp = part.get('cestamp') or ''
        msg.step_file_path = str(step_path)
        msg.mesh_units = units
        msg.plan_status = plan_status
        msg.plan_file_path = plan_path or ''
        self.publisher.publish(msg)
        return True, 'Selection published.'

    def shutdown(self):
        """Tear down the ROS node, if one was created."""
        if self.node is not None:
            try:
                self.node.destroy_node()
            except Exception:  # noqa: BLE001
                pass


def card_state(part, catalog):
    """Classify a part into the picker's card state.

    Args:
        part: A catalog row.
        catalog: The `CatalogSync`, used to resolve cache paths against this
            process's mount point.

    Returns:
        tuple: (key, label) where key drives the CSS class.
    """
    if part.get('sync_status') == 'archived':
        return 'archived', 'Archived'
    if catalog.resolved_step_path(part) is not None:
        if part.get('step_cestamp') and part['step_cestamp'] != part.get('cestamp'):
            return 'stale', 'Stale'
        return 'ready', 'Ready'
    if part.get('step_available'):
        return 'fetch', 'Fetch'
    if part.get('sync_status') == 'new':
        return 'new', 'New'
    return 'nostp', 'No STP'


def create_app(catalog=None, bridge=None, mesh_units='mm'):
    """Build the Flask application.

    Args:
        catalog: A `CatalogSync`; one is constructed from the environment
            when omitted.
        bridge: A `PickerBridge`; one is constructed when omitted.
        mesh_units: Units announced with selections, matching how the STEP
            files on this tenant are authored.

    Returns:
        Flask: The configured application.
    """
    if catalog is None:
        paths = CatalogPaths.from_env()
        catalog = CatalogSync(DXClient(DXConfig.from_env()), paths.db_path,
                              SyncConfig.from_env(), paths)
    if bridge is None:
        bridge = PickerBridge(catalog)

    app = Flask(__name__)
    app.config['CATALOG'] = catalog
    app.config['BRIDGE'] = bridge
    app.config['MESH_UNITS'] = mesh_units

    def part_payload(part):
        """Serialize one catalog row for the front end."""
        state, label = card_state(part, catalog)
        # Local-only: the grid renders every part, and a 3DX round trip per
        # card would stall the UI. The detail panel does the remote check.
        plan_status, plan_path = catalog.ensure_plan(part['eng_item_id'],
                                                     check_remote=False)
        return {
            'eng_item_id': part['eng_item_id'],
            'title': part.get('title') or '(untitled)',
            'part_number': part.get('part_number') or '',
            'revision': part.get('revision') or '',
            'maturity': part.get('maturity') or '',
            'collab_space': part.get('collab_space') or '',
            'description': part.get('description') or '',
            'sync_status': part.get('sync_status') or '',
            'state': state,
            'state_label': label,
            'step_cached': catalog.resolved_step_path(part) is not None,
            'step_available': bool(part.get('step_available')),
            'has_thumbnail': catalog.resolved_thumb_path(part) is not None,
            'plan_status': plan_status.value,
            'plan_file_path': plan_path or '',
            'last_synced': part.get('last_synced') or '',
        }

    @app.route('/')
    def index():
        """Render the picker shell; the grid is filled by /api/parts."""
        return render_template('index.html')

    @app.route('/api/parts')
    def api_parts():
        """The catalog as JSON, honouring search and maturity filters."""
        parts = catalog.db.list_parts(
            search=request.args.get('search') or None,
            maturity=request.args.get('maturity') or None,
            collab_space=catalog.config.collab_space_filter or None,
            include_archived=request.args.get('archived') == '1')
        return jsonify({'parts': [part_payload(part) for part in parts],
                        'total': len(parts)})

    @app.route('/api/status')
    def api_status():
        """Catalog counts, sync state, and the last sync's outcome."""
        status = catalog.status()
        status['ros'] = bridge.publisher is not None
        status['configured'] = catalog.client.config.is_configured()[0]
        return jsonify(status)

    @app.route('/api/select', methods=['POST'])
    def api_select():
        """Fetch a part's STEP, resolve its plan, and announce the selection.

        An optional `plan_id` loads that exact plan rather than whichever one
        is current -- including one generated against an earlier CAD revision,
        which comes back as SELECTED so the pipeline honours the choice and
        warns. `plan_id: null` explicitly loads the model with no plan.
        """
        payload = request.get_json(silent=True) or {}
        eng_item_id = payload.get('eng_item_id') or request.form.get('eng_item_id')
        if not eng_item_id:
            return jsonify({'success': False, 'message': 'eng_item_id is required.'}), 400

        part = catalog.db.get_part(eng_item_id)
        if part is None:
            return jsonify({'success': False,
                            'message': f'Unknown part: {eng_item_id}.'}), 404

        step_path, message = catalog.fetch_step(eng_item_id)
        if step_path is None:
            return jsonify({'success': False, 'message': message}), 409

        plan_id = payload.get('plan_id')
        if plan_id:
            plan_status, plan_path = catalog.select_plan(eng_item_id, plan_id)
            if plan_status == PlanStatus.NONE:
                return jsonify({'success': False,
                                'message': f'Plan is unavailable: {plan_id}.'}), 404
        elif payload.get('model_only'):
            plan_status, plan_path = PlanStatus.NONE, ''
        else:
            plan_status, plan_path = catalog.ensure_plan(eng_item_id)

        published, publish_message = bridge.publish_selection(
            part, step_path, plan_status.value, plan_path,
            units=app.config['MESH_UNITS'])

        return jsonify({
            'success': True,
            'eng_item_id': eng_item_id,
            'title': part.get('title'),
            'step_file_path': str(step_path),
            'plan_status': plan_status.value,
            'plan_file_path': plan_path or '',
            'published': published,
            'message': f'{message} {publish_message}'.strip(),
        })

    @app.route('/api/sync', methods=['POST'])
    def api_sync():
        """Run a catalog sync and report what changed."""
        payload = request.get_json(silent=True) or {}
        full = bool(payload.get('full', True))
        configured, message = catalog.client.config.is_configured()
        if not configured:
            return jsonify({'success': False, 'message': message}), 409
        if catalog.syncing:
            return jsonify({'success': False,
                            'message': 'A sync is already running.'}), 409
        result = (catalog.run_full_sync() if full
                  else catalog.run_incremental_sync())
        return jsonify({'success': result.success, **result.to_dict()})

    @app.route('/thumb/<eng_item_id>')
    def thumb(eng_item_id):
        """Serve a cached thumbnail image."""
        part = catalog.db.get_part(eng_item_id)
        if part is None:
            abort(404)
        path = catalog.resolved_thumb_path(part)
        if path is None:
            abort(404)
        return send_file(str(path), mimetype='image/png')

    @app.route('/api/parts/<eng_item_id>/plan')
    def api_part_plan(eng_item_id):
        """Plan status and summary for one part."""
        part = catalog.db.get_part(eng_item_id)
        if part is None:
            abort(404)
        status, path = catalog.ensure_plan(eng_item_id)
        plan = catalog.db.get_current_plan(eng_item_id)
        return jsonify({
            'eng_item_id': eng_item_id,
            'status': status.value,
            'file_path': path or '',
            'stale_reason': ('Plan targets CAD cestamp '
                             f"{plan['cestamp']}, current is {part.get('cestamp')}."
                             if plan and status == PlanStatus.STALE else ''),
            'plan': plan,
        })

    @app.route('/api/parts/<eng_item_id>/plans')
    def api_part_plans(eng_item_id):
        """Every plan recorded for a part, newest first.

        This is what fills the picker's plan strip, so it must stay purely
        local -- no 3DX round trip.
        """
        if catalog.db.get_part(eng_item_id) is None:
            abort(404)
        plans = catalog.list_plans(eng_item_id)
        current = next((p['plan_id'] for p in plans if p['is_current']), '')
        return jsonify({'plans': plans, 'current_plan_id': current,
                        'total': len(plans)})

    @app.route('/api/parts/<eng_item_id>/plans/adopt', methods=['POST'])
    def api_adopt_plans(eng_item_id):
        """Record plan files already on disk that the catalog never captured."""
        if catalog.db.get_part(eng_item_id) is None:
            abort(404)
        adopted, skipped = catalog.adopt_orphan_plans(eng_item_id)
        return jsonify({
            'success': True,
            'adopted': adopted,
            'skipped': skipped,
            'message': (f'Adopted {adopted} plan(s).' if adopted
                        else 'No unrecorded plan files found.'),
        })

    @app.route('/api/plans/<path:plan_id>/upload', methods=['POST'])
    def api_upload_plan(plan_id):
        """Upload one recorded plan to 3DX as a Document."""
        if catalog.db.get_plan(plan_id) is None:
            abort(404)
        doc_id, message = catalog.upload_recorded_plan(plan_id)
        return jsonify({'success': doc_id is not None,
                        'plan_doc_id': doc_id or '',
                        'message': message}), (200 if doc_id else 409)

    @app.route('/api/parts/<eng_item_id>/runs')
    def api_part_runs(eng_item_id):
        """Inspection history for one part."""
        if catalog.db.get_part(eng_item_id) is None:
            abort(404)
        return jsonify({'runs': catalog.db.list_runs(eng_item_id)})

    return app


def main(args=None):
    """Entry point for both the ROS executable and standalone use."""
    parser = argparse.ArgumentParser(
        prog='viewpoint_generation.picker.app',
        description='Part picker UI for the 3DEXPERIENCE catalog.')
    parser.add_argument('--port', type=int,
                        default=int(os.environ.get('PICKER_PORT', '5050')),
                        help='Port to listen on (default: $PICKER_PORT or 5050).')
    parser.add_argument('--host', default='0.0.0.0',
                        help='Address to bind (default: all interfaces).')
    parser.add_argument('--units', default='mm',
                        help='Units used to load selected STEP files.')
    parser.add_argument('--no-ros', action='store_true',
                        help='Do not attempt to publish selections over ROS.')
    parser.add_argument('--debug', action='store_true', help='Flask debug mode.')
    parsed, _ = parser.parse_known_args(args)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-7s %(name)s: %(message)s')

    paths = CatalogPaths.from_env()
    success, message = paths.ensure()
    if not success:
        logger.error(message)
        return 1
    catalog = CatalogSync(DXClient(DXConfig.from_env()), paths.db_path,
                          SyncConfig.from_env(), paths)
    bridge = PickerBridge(catalog, enable_ros=not parsed.no_ros)
    app = create_app(catalog, bridge, mesh_units=parsed.units)

    logger.info('Part picker listening on http://%s:%d', parsed.host, parsed.port)
    try:
        # threaded so a slow STEP fetch cannot block the whole UI.
        app.run(host=parsed.host, port=parsed.port, debug=parsed.debug,
                threaded=True, use_reloader=False)
    finally:
        bridge.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
