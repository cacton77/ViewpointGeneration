"""Flask part picker UI for the 3DEXPERIENCE catalog.

Serves a card grid of catalogued parts with thumbnails, cache state, and plan
state, and turns a click into the full selection flow: fetch the STEP, resolve
the inspection plan, and announce the part so the viewpoint generation node
loads it.

**How this talks to ROS: it does not.** The catalog node runs the picker in its
own process (`catalog.picker_enabled`) and hands `create_app()` three
callbacks -- `on_select` to announce a selection on `/catalog/part_selected`,
`on_sync` to run a sync under the node's sync lock, and `on_adopt` to have the
plan producer describe plan files found on disk. The picker imports no ROS
message types. Without callbacks it still works completely for browsing and
caching; it reports that selections were not announced.

Run standalone (no ROS), e.g. for UI work:

    python -m viewpoint_generation.picker.app --port 5050
"""

import argparse
import logging
import os
import threading

from flask import Flask, abort, jsonify, render_template, request, send_file

from viewpoint_generation.catalog.client import DXClient
from viewpoint_generation.catalog.config import CatalogPaths, DXConfig, SyncConfig
from viewpoint_generation.catalog.sync import (CatalogSync, PlanStatus,
                                               describe_with_command)

logger = logging.getLogger(__name__)


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


def create_app(catalog=None, on_select=None, on_sync=None, on_adopt=None):
    """Build the Flask application.

    Args:
        catalog: A `CatalogSync`; one is constructed from the environment
            when omitted.
        on_select: Callable taking the selection dict from
            `CatalogSync.select()` and returning (published, message). None
            means selections are not announced anywhere.
        on_sync: Callable taking `full` and returning (SyncResult or None,
            message). None runs the sync directly in the request thread.
        on_adopt: Callable taking an eng_item_id and returning (adopted,
            skipped, message). None asks the producer's describe command
            (`describe_with_command`).

    Returns:
        Flask: The configured application.
    """
    if catalog is None:
        paths = CatalogPaths.from_env()
        catalog = CatalogSync(DXClient(DXConfig.from_env()), paths.db_path,
                              SyncConfig.from_env(), paths)
    sync_lock = threading.Lock()

    app = Flask(__name__)
    app.config['CATALOG'] = catalog

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
        status['ros'] = on_select is not None
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

        selection, message = catalog.select(
            eng_item_id, plan_id=payload.get('plan_id'),
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
            'eng_item_id': eng_item_id,
            'title': part.get('title'),
            'step_file_path': selection['step_path'],
            'plan_status': selection['plan_status'].value,
            'plan_file_path': selection['plan_path'],
            'plan_dir': selection['plan_dir'],
            'published': published,
            'message': f'{message} {publish_message}'.strip(),
        })

    @app.route('/api/sync', methods=['POST'])
    def api_sync():
        """Run a catalog sync and report what changed."""
        payload = request.get_json(silent=True) or {}
        full = bool(payload.get('full', True))
        if on_sync is not None:
            result, message = on_sync(full)
            if result is None:
                return jsonify({'success': False, 'message': message}), 409
            return jsonify({'success': result.success, **result.to_dict()})

        configured, message = catalog.client.config.is_configured()
        if not configured:
            return jsonify({'success': False, 'message': message}), 409
        if not sync_lock.acquire(blocking=False) or catalog.syncing:
            return jsonify({'success': False,
                            'message': 'A sync is already running.'}), 409
        try:
            result = (catalog.run_full_sync() if full
                      else catalog.run_incremental_sync())
        finally:
            sync_lock.release()
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
        """Record plan files already on disk that the catalog never captured.

        The catalog only lists the files; the plan producer describes them.
        """
        if catalog.db.get_part(eng_item_id) is None:
            abort(404)
        if on_adopt is not None:
            adopted, skipped, message = on_adopt(eng_item_id)
        else:
            adopted, skipped = catalog.adopt_orphan_plans(
                eng_item_id, describe=describe_with_command)
            message = (f'Adopted {adopted} plan(s).' if adopted
                       else 'No unrecorded plan files found.')
        return jsonify({
            'success': True,
            'adopted': adopted,
            'skipped': skipped,
            'message': message,
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
    """Standalone entry point (no ROS): browse, cache, and sync only.

    Inside the cell the catalog node serves the picker itself; see
    `catalog.picker_enabled`.
    """
    parser = argparse.ArgumentParser(
        prog='viewpoint_generation.picker.app',
        description='Part picker UI for the 3DEXPERIENCE catalog (standalone, no ROS).')
    parser.add_argument('--port', type=int,
                        default=int(os.environ.get('PICKER_PORT', '5050')),
                        help='Port to listen on (default: $PICKER_PORT or 5050).')
    parser.add_argument('--host', default='0.0.0.0',
                        help='Address to bind (default: all interfaces).')
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
    app = create_app(catalog)

    logger.info('Part picker listening on http://%s:%d', parsed.host, parsed.port)
    # threaded so a slow STEP fetch cannot block the whole UI.
    app.run(host=parsed.host, port=parsed.port, debug=parsed.debug,
            threaded=True, use_reloader=False)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
