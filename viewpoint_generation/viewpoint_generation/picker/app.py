"""Flask part picker UI for the PLM parts catalog.

Serves a card grid of catalogued parts -- labelled by the backend they come
from -- with thumbnails, cache state, and plan state, and turns a click into
the full selection flow: fetch the STEP, resolve the inspection plan, and
announce the part so the plan producer loads it.

Parts are addressed by their `storage_key` in URLs: a backend's own part_id
is opaque and may contain characters a URL path should not.

**How this talks to ROS: it does not.** The catalog node runs the picker in its
own process and hands `create_app()` three callbacks -- `on_select` to
announce a selection, `on_sync` to run a sync under the node's sync lock, and
`on_adopt` to have the plan producer describe plan files found on disk. The
picker imports no ROS message types. Without callbacks it still works for
browsing and caching, and says that selections were not announced.

Run standalone (no ROS), e.g. for UI work:

    python -m viewpoint_generation.picker.app --port 5050
"""

import argparse
import logging
import os
import threading

from flask import Flask, abort, jsonify, render_template, request, send_file

from viewpoint_generation.catalog.backends.base import Capability
from viewpoint_generation.catalog.cli import describe_with_command
from viewpoint_generation.catalog.core.catalog import PlanStatus
from viewpoint_generation.catalog.factory import build_catalog

logger = logging.getLogger(__name__)


def card_state(part, catalog):
    """Classify a part into the picker's card state.

    Returns:
        tuple: (key, label) where key drives the CSS class.
    """
    if part.get('sync_status') == 'archived':
        return 'archived', 'Archived'
    if catalog.resolved_geometry_path(part) is not None:
        if part.get('geometry_token') and part['geometry_token'] != part.get('change_token'):
            return 'stale', 'Stale'
        return 'ready', 'Ready'
    if part.get('geometry_available'):
        return 'fetch', 'Fetch'
    if part.get('sync_status') == 'new':
        return 'new', 'New'
    return 'nostp', 'No STP'


def create_app(catalog=None, on_select=None, on_sync=None, on_adopt=None):
    """Build the Flask application.

    Args:
        catalog: A `Catalog`; built from the environment when omitted.
        on_select: Callable taking the selection dict from `Catalog.select()`
            and returning (published, message). None: not announced anywhere.
        on_sync: Callable taking `full` and returning (list of SyncResult,
            message). None runs the sync in the request thread.
        on_adopt: Callable taking (backend, part_id) and returning (adopted,
            skipped, message). None asks the producer's describe command.

    Returns:
        Flask: The configured application.
    """
    if catalog is None:
        catalog, _errors = build_catalog()
    sync_lock = threading.Lock()

    app = Flask(__name__)
    app.config['CATALOG'] = catalog

    def part_or_404(key):
        part = catalog.db.get_part_by_key(key)
        if part is None:
            abort(404)
        return part

    def part_payload(part):
        """Serialize one catalog row for the front end."""
        state, label = card_state(part, catalog)
        # Local-only: the grid renders every part, and a remote round trip
        # per card would stall the UI. The detail panel does the remote check.
        plan_status, plan_path = catalog.ensure_plan(part['backend'], part['part_id'],
                                                     check_remote=False)
        return {
            'key': part['storage_key'],
            'backend': part['backend'],
            'part_id': part['part_id'],
            'title': part.get('title') or '(untitled)',
            'part_number': part.get('part_number') or '',
            'revision': part.get('revision') or '',
            'maturity': part.get('maturity') or '',
            'container': part.get('container') or '',
            'description': part.get('description') or '',
            'sync_status': part.get('sync_status') or '',
            'state': state,
            'state_label': label,
            'geometry_cached': catalog.resolved_geometry_path(part) is not None,
            'geometry_available': bool(part.get('geometry_available')),
            'has_thumbnail': catalog.resolved_thumb_path(part) is not None,
            'can_upload': catalog.has(part['backend'], Capability.ATTACH),
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
        """The catalog as JSON, honouring search, maturity and backend filters."""
        parts = catalog.db.list_parts(
            search=request.args.get('search') or None,
            maturity=request.args.get('maturity') or None,
            backend=request.args.get('backend') or None,
            include_archived=request.args.get('archived') == '1')
        return jsonify({'parts': [part_payload(part) for part in parts],
                        'total': len(parts)})

    @app.route('/api/status')
    def api_status():
        """Catalog counts, sync state, backends, and the last syncs' outcomes."""
        status = catalog.status()
        status['ros'] = on_select is not None
        status['configured'] = any(b['configured'] for b in status['backends'])
        return jsonify(status)

    @app.route('/api/select', methods=['POST'])
    def api_select():
        """Fetch a part's STEP, resolve its plan, and announce the selection.

        The body names the part by `key` (its storage_key) or by `backend` and
        `part_id`. An optional `plan_id` loads that exact plan -- including one
        generated against an earlier change token, which comes back SELECTED
        so the pipeline honours the choice and warns. `model_only` loads the
        geometry with no plan.
        """
        payload = request.get_json(silent=True) or {}
        part = None
        if payload.get('key'):
            part = catalog.db.get_part_by_key(payload['key'])
        elif payload.get('part_id'):
            part = catalog.db.get_part(payload.get('backend', ''), payload['part_id'])
        else:
            return jsonify({'success': False, 'message': 'key or part_id is required.'}), 400
        if part is None:
            return jsonify({'success': False, 'message': 'Unknown part.'}), 404

        selection, message = catalog.select(
            part['backend'], part['part_id'], plan_id=payload.get('plan_id'),
            model_only=bool(payload.get('model_only')))
        if selection is None:
            status = 404 if message.startswith('Plan is unavailable') else 409
            return jsonify({'success': False, 'message': message}), status

        if on_select is None:
            published, publish_message = (
                False, 'ROS is not available; selection was not announced.')
        else:
            published, publish_message = on_select(selection)

        return jsonify({
            'success': True,
            'key': part['storage_key'],
            'backend': part['backend'],
            'part_id': part['part_id'],
            'title': part.get('title'),
            'geometry_path': selection['geometry_path'],
            'plan_status': selection['plan_status'].value,
            'plan_file_path': selection['plan_path'],
            'plan_dir': selection['plan_dir'],
            'published': published,
            'message': f'{message} {publish_message}'.strip(),
        })

    @app.route('/api/sync', methods=['POST'])
    def api_sync():
        """Run a catalog sync and report what changed, per backend."""
        payload = request.get_json(silent=True) or {}
        full = bool(payload.get('full', True))
        backend = payload.get('backend') or None
        if on_sync is not None:
            results, message = on_sync(full, backend)
        else:
            if not sync_lock.acquire(blocking=False) or catalog.syncing:
                return jsonify({'success': False,
                                'message': 'A sync is already running.'}), 409
            try:
                run = catalog.run_full_sync if full else catalog.run_incremental_sync
                results = run(backend)
            finally:
                sync_lock.release()
            message = '; '.join(f'{r.backend}: {r.message}' for r in results)
        if not results:
            return jsonify({'success': False, 'message': message}), 409
        totals = {field: sum(getattr(r, field) for r in results)
                  for field in ('added', 'modified', 'unchanged', 'archived', 'errors')}
        return jsonify({'success': all(r.success for r in results), 'message': message,
                        'results': [r.to_dict() for r in results], **totals})

    @app.route('/thumb/<key>')
    def thumb(key):
        """Serve a cached thumbnail image."""
        path = catalog.resolved_thumb_path(part_or_404(key))
        if path is None:
            abort(404)
        return send_file(str(path), mimetype='image/png')

    @app.route('/api/parts/<key>/plan')
    def api_part_plan(key):
        """Plan status and summary for one part."""
        part = part_or_404(key)
        status, path = catalog.ensure_plan(part['backend'], part['part_id'])
        plan = catalog.db.get_current_plan(part['backend'], part['part_id'])
        return jsonify({
            'key': key,
            'status': status.value,
            'file_path': path or '',
            'stale_reason': ('Plan targets change token '
                             f"{plan['change_token']}, current is {part.get('change_token')}."
                             if plan and status == PlanStatus.STALE else ''),
            'plan': plan,
        })

    @app.route('/api/parts/<key>/plans')
    def api_part_plans(key):
        """Every plan recorded for a part, newest first (purely local)."""
        part = part_or_404(key)
        plans = catalog.list_plans(part['backend'], part['part_id'])
        current = next((p['plan_id'] for p in plans if p['is_current']), '')
        return jsonify({'plans': plans, 'current_plan_id': current, 'total': len(plans),
                        'can_upload': catalog.has(part['backend'], Capability.ATTACH)})

    @app.route('/api/parts/<key>/plans/adopt', methods=['POST'])
    def api_adopt_plans(key):
        """Record plan files on disk that the catalog never captured.

        The catalog only lists the files; the plan producer describes them.
        """
        part = part_or_404(key)
        if on_adopt is not None:
            adopted, skipped, message = on_adopt(part['backend'], part['part_id'])
        else:
            adopted, skipped = catalog.adopt_orphan_plans(
                part['backend'], part['part_id'], describe=describe_with_command)
            message = (f'Adopted {adopted} plan(s).' if adopted
                       else 'No unrecorded plan files found.')
        return jsonify({'success': True, 'adopted': adopted, 'skipped': skipped,
                        'message': message})

    @app.route('/api/plans/<path:plan_id>/upload', methods=['POST'])
    def api_upload_plan(plan_id):
        """Upload one recorded plan through its part's backend."""
        if catalog.db.get_plan(plan_id) is None:
            abort(404)
        ref, message = catalog.upload_recorded_plan(plan_id)
        return jsonify({'success': ref is not None, 'remote_ref': ref or '',
                        'message': message}), (200 if ref else 409)

    @app.route('/api/parts/<key>/runs')
    def api_part_runs(key):
        """Inspection history for one part."""
        part = part_or_404(key)
        return jsonify({'runs': catalog.db.list_runs(part['backend'], part['part_id'])})

    return app


def main(args=None):
    """Standalone entry point (no ROS): browse, cache, and sync only.

    Inside the cell the catalog node serves the picker itself; see
    `catalog.picker_enabled`.
    """
    parser = argparse.ArgumentParser(
        prog='viewpoint_generation.picker.app',
        description='Part picker UI for the PLM parts catalog (standalone, no ROS).')
    parser.add_argument('--port', type=int,
                        default=int(os.environ.get('PICKER_PORT', '5050')),
                        help='Port to listen on (default: $PICKER_PORT or 5050).')
    parser.add_argument('--host', default='0.0.0.0',
                        help='Address to bind (default: all interfaces).')
    parser.add_argument('--debug', action='store_true', help='Flask debug mode.')
    parsed, _ = parser.parse_known_args(args)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-7s %(name)s: %(message)s')
    catalog, _errors = build_catalog()
    app = create_app(catalog)
    logger.info('Part picker listening on http://%s:%d', parsed.host, parsed.port)
    try:
        # threaded so a slow STEP fetch cannot block the whole UI.
        app.run(host=parsed.host, port=parsed.port, debug=parsed.debug,
                threaded=True, use_reloader=False)
    finally:
        catalog.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
