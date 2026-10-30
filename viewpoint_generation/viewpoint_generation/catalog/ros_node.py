"""ROS 2 node exposing the PLM parts catalog to the inspection cell.

One node hosts every configured backend (`catalog.backends`, from
`PLM_BACKENDS`). Services and topics live under `catalog.namespace` (default
`catalog`), so their names are unchanged from the single-backend catalog:

    <ns>/list_backends    -- hosted backends, capabilities, connection state
    <ns>/list_parts       -- browse the local catalog
    <ns>/select_part      -- fetch the geometry, resolve the plan, announce it
    <ns>/sync_now         -- run a sync on demand (one backend or all)
    <ns>/ensure_plan      -- report plan validity for one part
    <ns>/record_plan      -- record a plan the producer wrote (plan contract)
    <ns>/upload_plan      -- upload a recorded plan on demand
    <ns>/upload_results   -- upload an inspection run's result bundle

    /<ns>/part_selected   -- PartSelected, latched for late joiners
    /<ns>/sync_complete   -- SyncResult after every sync pass, per backend

As a client it calls the plan producer's `DescribePlans` service
(`catalog.describe_plans_service`, default `viewpoint_generation/describe_plans`)
when the picker's "Scan for files" finds plan files it never recorded.

Parameters: every `CoreConfig` field as `catalog.<field>`, and every field of
a backend's non-secret settings as `<backend>.<field>` (e.g.
`dx3.bookmark_scope`). Credentials are never parameters.

The browser part picker runs in this process when `catalog.picker_enabled`
is true, sharing the node's catalog, sync lock and publisher through
callbacks.

Network work never runs on the executor thread that serves listings: syncs
run on a background worker, and service handlers that touch the network are
in a reentrant callback group.
"""

import threading

import rclpy
import rclpy.node
from rcl_interfaces.msg import (FloatingPointRange, IntegerRange, ParameterDescriptor,
                                SetParametersResult)
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from viewpoint_generation_interfaces.msg import (BackendInfo, PartSelected, PartSummary)
from viewpoint_generation_interfaces.msg import SyncResult as SyncResultMsg
from viewpoint_generation_interfaces.srv import (DescribePlans, EnsurePlan, ListBackends,
                                                 ListParts, RecordPlan, SelectPart, SyncNow,
                                                 UploadPlan, UploadResults)

from viewpoint_generation.catalog.core.catalog import PlanStatus
from viewpoint_generation.catalog.core.config import CoreConfig
from viewpoint_generation.catalog.factory import build_catalog

# Core settings that only take effect at startup.
STARTUP_ONLY = {'backends', 'root', 'picker_enabled', 'picker_port', 'namespace'}

PLAN_STATUS_MESSAGES = {
    PlanStatus.CURRENT: 'Plan matches the current CAD revision.',
    PlanStatus.DOWNLOADED: 'Plan downloaded from the PLM.',
    PlanStatus.UPDATED_FROM_REMOTE: 'A newer plan was downloaded from the PLM.',
    PlanStatus.SELECTED: 'Plan was chosen explicitly and targets an earlier CAD revision.',
    PlanStatus.STALE: ('Plan was generated for an earlier CAD revision; '
                       're-planning is recommended.'),
    PlanStatus.NONE: 'No inspection plan exists for this part.',
}


def _make_descriptor(field_info):
    """Build a ParameterDescriptor from one config.to_dict() entry."""
    descriptor = ParameterDescriptor()
    descriptor.description = field_info.get('description', '')
    descriptor.additional_constraints = field_info.get('control', '')
    range_val = field_info.get('range')
    if range_val is not None:
        if field_info['type'] == 'float':
            fp = FloatingPointRange()
            fp.from_value = float(range_val[0])
            fp.to_value = float(range_val[1])
            fp.step = 0.0
            descriptor.floating_point_range = [fp]
        elif field_info['type'] == 'integer':
            ir = IntegerRange()
            ir.from_value = int(range_val[0])
            ir.to_value = int(range_val[1])
            ir.step = 0
            descriptor.integer_range = [ir]
    return descriptor


class CatalogNode(rclpy.node.Node):
    """ROS 2 front end for the local parts catalog."""

    def __init__(self, node_name='catalog'):
        super().__init__(node_name)

        # CoreConfig follows the repo's to_dict() convention, so its fields
        # become parameters automatically; adding a field is enough.
        self.config = CoreConfig.from_env()
        self._declare_from(self.config, 'catalog')
        self.declare_parameter('catalog.describe_plans_service',
                               'viewpoint_generation/describe_plans')

        self.catalog, errors = build_catalog(self.config)
        for message in errors:
            self.get_logger().error(message)
        # Each backend's non-secret settings become <backend>.<field>.
        for name, backend in self.catalog.backends.items():
            settings = getattr(backend, 'settings', None)
            if settings is not None and hasattr(settings, 'to_dict'):
                self._declare_from(settings, name)
            configured, message = backend.is_configured()
            if configured:
                self.get_logger().info(f'{name}: {message}')
            else:
                # Not fatal: everything already cached still browses and
                # selects, and the operator gets a clear reason syncing won't.
                self.get_logger().warn(f'{name}: {message} Serving cached parts only.')

        # Serialize sync passes; a second request while one runs is rejected
        # rather than queued, so a PLM never sees overlapping enumerations.
        self._sync_lock = threading.Lock()
        self._sync_thread = None

        ns = self.config.namespace.strip('/') or 'catalog'
        services = ReentrantCallbackGroup()
        for srv_type, name, callback in (
                (ListBackends, 'list_backends', self.list_backends_callback),
                (ListParts, 'list_parts', self.list_parts_callback),
                (SelectPart, 'select_part', self.select_part_callback),
                (SyncNow, 'sync_now', self.sync_now_callback),
                (EnsurePlan, 'ensure_plan', self.ensure_plan_callback),
                (RecordPlan, 'record_plan', self.record_plan_callback),
                (UploadPlan, 'upload_plan', self.upload_plan_callback),
                (UploadResults, 'upload_results', self.upload_results_callback)):
            self.create_service(srv_type, f'{ns}/{name}', callback, callback_group=services)

        # Latched: a picker or GUI that starts after a selection still learns
        # which part is loaded.
        latched = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                             reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.part_selected_pub = self.create_publisher(
            PartSelected, f'/{ns}/part_selected', latched)
        self.sync_complete_pub = self.create_publisher(
            SyncResultMsg, f'/{ns}/sync_complete', 10)
        self.describe_plans_client = self.create_client(
            DescribePlans, self.get_parameter('catalog.describe_plans_service').value,
            callback_group=services)

        # Event-driven sync where a backend can push changes and actually
        # connected; polling otherwise.
        self.event_driven = self.catalog.start_events(on_synced=self._on_event_synced)
        if self.event_driven:
            self.get_logger().info(f"Event-driven sync for: {', '.join(self.event_driven)}")

        self.sync_timer = None
        self._timer_cb_group = MutuallyExclusiveCallbackGroup()
        self._restart_sync_timer()
        self.add_on_set_parameters_callback(self.parameter_callback)

        self.picker_server = None
        if self.config.picker_enabled:
            self._start_picker()

        if self.config.sync_on_startup and self._any_configured():
            self.get_logger().info('Running startup catalog sync...')
            self._start_sync(full=True)

        self.get_logger().info(
            f"Catalog node ready: {self.catalog.db.counts()['total']} parts in "
            f'{self.catalog.paths.db_path}; backends: '
            f"{', '.join(sorted(self.catalog.backends)) or 'none'}")

    # --- parameters ---------------------------------------------------------------

    def _declare_from(self, config, prefix):
        """Declare a config dataclass's fields as `<prefix>.<field>` and apply values."""
        params = [(f'{prefix}.{name}', info['value'], _make_descriptor(info))
                  for name, info in config.to_dict().items()]
        self.declare_parameters(namespace='', parameters=params)
        for ros_name, _, _ in params:
            setattr(config, ros_name[len(prefix) + 1:], self.get_parameter(ros_name).value)

    def parameter_callback(self, params):
        """Apply parameter changes to the live configuration."""
        for param in params:
            prefix, _, field = param.name.partition('.')
            if prefix == 'catalog' and field in self.config.to_dict():
                setattr(self.config, field, param.value)
                if field in STARTUP_ONLY:
                    self.get_logger().warn(f'{param.name} takes effect at the next start.')
                elif field == 'sync_interval':
                    self._restart_sync_timer()
            elif prefix in self.catalog.backends:
                settings = getattr(self.catalog.backends[prefix], 'settings', None)
                if settings is not None and field in settings.to_dict():
                    setattr(settings, field, param.value)
            else:
                continue
            self.get_logger().info(f'{param.name} set to {param.value}.')
        return SetParametersResult(successful=True)

    def _restart_sync_timer(self):
        """(Re)arm the background sync timer from the configured interval.

        With an event-driven backend the timer becomes a slow safety net:
        events cover ordinary changes, and an occasional full pass catches
        anything the broker dropped.
        """
        if self.sync_timer is not None:
            self.sync_timer.cancel()
            self.destroy_timer(self.sync_timer)
            self.sync_timer = None
        interval = self.config.sync_interval
        if self.event_driven and interval and interval > 0:
            interval = max(interval, 1800)
        if interval and interval > 0:
            self.sync_timer = self.create_timer(float(interval), self._timer_sync,
                                                callback_group=self._timer_cb_group)
            self.get_logger().info(
                f'Background sync every {interval}s'
                f'{" (safety net; events drive updates)" if self.event_driven else ""}.')
        else:
            self.get_logger().info('Background sync disabled (interval = 0).')

    # --- helpers -------------------------------------------------------------------

    def _any_configured(self):
        return any(b.is_configured()[0] for b in self.catalog.backends.values())

    def _backend_name(self, requested):
        """The backend a request means: as given, or the only one hosted."""
        if requested:
            return requested
        if len(self.catalog.backends) == 1:
            return next(iter(self.catalog.backends))
        return ''

    # --- picker and plan producer ------------------------------------------------

    def _start_picker(self):
        """Serve the browser part picker from a thread in this process."""
        from werkzeug.serving import make_server

        from viewpoint_generation.picker.app import create_app

        app = create_app(self.catalog, on_select=self._publish_selection,
                         on_sync=self.sync_blocking, on_adopt=self.adopt_plans)
        port = int(self.config.picker_port)
        try:
            self.picker_server = make_server('0.0.0.0', port, app, threaded=True)
        except OSError as e:
            self.get_logger().error(f'Part picker could not bind port {port}: {e}')
            return
        threading.Thread(target=self.picker_server.serve_forever,
                         name='catalog-picker', daemon=True).start()
        self.get_logger().info(f'Part picker listening on http://0.0.0.0:{port}')

    def describe_plans(self, paths, timeout=120.0):
        """Ask the plan producer to describe plan files (DescribePlans).

        Returns:
            list: description dicts; empty when the producer is not running.
        """
        if not self.describe_plans_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn(
                f'Plan producer ({self.describe_plans_client.srv_name}) is not running; '
                'cannot describe plan files.')
            return []
        request = DescribePlans.Request()
        request.paths = [str(path) for path in paths]
        done = threading.Event()
        future = self.describe_plans_client.call_async(request)
        future.add_done_callback(lambda _: done.set())
        if not done.wait(timeout) or future.result() is None:
            self.get_logger().warn('Plan producer did not answer describe_plans.')
            return []
        return [{'path': plan.path, 'valid': plan.valid, 'stage': plan.stage,
                 'stage_rank': plan.stage_rank, 'summary_json': plan.summary_json,
                 'summary_text': plan.summary_text, 'change_token': plan.change_token,
                 'generated_at': plan.generated_at}
                for plan in future.result().plans]

    def adopt_plans(self, backend_name, part_id):
        """Record plan files on disk that the catalog never captured.

        Returns:
            tuple: (int, int, str) adopted and skipped counts, and a message.
        """
        if not self.catalog.orphan_plan_candidates(backend_name, part_id):
            return 0, 0, 'No unrecorded plan files found.'
        if not self.describe_plans_client.wait_for_service(timeout_sec=2.0):
            return 0, 0, ('Found unrecorded plan files, but the plan producer is not '
                          'running to describe them.')
        adopted, skipped = self.catalog.adopt_orphan_plans(
            backend_name, part_id, describe=self.describe_plans)
        return adopted, skipped, (f'Adopted {adopted} plan(s).' if adopted
                                  else 'No unrecorded plan files found.')

    def _publish_selection(self, selection):
        """Announce a selection made by the service or the picker.

        Returns:
            tuple: (bool, str) whether it was published, and a message.
        """
        part = selection['part']
        msg = PartSelected()
        msg.backend = part['backend']
        msg.part_id = part['part_id']
        msg.title = part.get('title') or ''
        msg.part_number = part.get('part_number') or ''
        msg.revision = part.get('revision') or ''
        msg.change_token = part.get('change_token') or ''
        msg.geometry_path = selection['geometry_path']
        msg.mesh_units = selection['mesh_units']
        msg.plan_status = selection['plan_status'].value
        msg.plan_file_path = selection['plan_path']
        msg.plan_dir = selection['plan_dir']
        msg.plm_context_json = selection['plm_context_json']
        self.part_selected_pub.publish(msg)
        self.get_logger().info(
            f"Selected {part.get('title')} ({part['backend']}) -> {msg.geometry_path} "
            f'(plan: {msg.plan_status})')
        return True, 'Selection published.'

    # --- sync -------------------------------------------------------------------------

    def _timer_sync(self):
        """Timer tick: start an incremental sync if none is running."""
        if self._any_configured():
            self._start_sync(full=False)

    def _start_sync(self, full, backend=None):
        """Run a sync on a background thread.

        Returns:
            bool: False when a sync is already in flight.
        """
        if not self._sync_lock.acquire(blocking=False):
            self.get_logger().debug('Sync already running; skipping this trigger.')
            return False

        def worker():
            try:
                run = self.catalog.run_full_sync if full else self.catalog.run_incremental_sync
                for result in run(backend or None):
                    self._publish_sync_result(result)
                    log = self.get_logger().info if result.success else self.get_logger().error
                    log(f"{result.backend} {'full' if full else 'incremental'} sync: "
                        f'{result.message}')
            except Exception as e:  # noqa: BLE001 - a worker must never die silently
                self.get_logger().error(f'Sync failed: {e}')
            finally:
                self._sync_lock.release()

        self._sync_thread = threading.Thread(target=worker, name='catalog-sync', daemon=True)
        self._sync_thread.start()
        return True

    def sync_blocking(self, full, backend=None):
        """Run a sync and wait for it, for callers that want real counts.

        Returns:
            tuple: (list of SyncResult, str) the results (empty when the sync
            could not start) and a message.
        """
        if backend and self.catalog.backend(backend) is None:
            return [], f'Backend {backend!r} is not hosted by this catalog.'
        if not self._any_configured():
            return [], 'No configured backend to sync.'
        before = dict(self.catalog.last_results)
        if not self._start_sync(full=full, backend=backend):
            return [], 'A catalog sync is already running.'
        # The worker owns the lock for the duration of the pass; joining keeps
        # the caller synchronous. Handlers run in a reentrant group, so other
        # services stay live meanwhile.
        if self._sync_thread is not None:
            self._sync_thread.join()
        names = [backend] if backend else sorted(self.catalog.backends)
        results = [self.catalog.last_results[name] for name in names
                   if name in self.catalog.last_results
                   and self.catalog.last_results[name] is not before.get(name)]
        if not results:
            return [], 'Sync produced no result.'
        return results, '; '.join(f'{r.backend}: {r.message}' for r in results)

    def _on_event_synced(self, backend, part_id, outcome):
        success, message = outcome
        self.get_logger().info(f'{backend} event for {part_id}: {message}')

    @staticmethod
    def _sync_msg(result):
        msg = SyncResultMsg()
        msg.backend = result.backend
        msg.success = bool(result.success)
        msg.added = int(result.added)
        msg.modified = int(result.modified)
        msg.unchanged = int(result.unchanged)
        msg.archived = int(result.archived)
        msg.errors = int(result.errors)
        msg.duration_sec = float(result.duration_sec)
        msg.started_at = result.started_at or ''
        msg.completed_at = result.completed_at or ''
        msg.message = result.message or ''
        return msg

    def _publish_sync_result(self, result):
        self.sync_complete_pub.publish(self._sync_msg(result))

    # --- service handlers -----------------------------------------------------------

    def list_backends_callback(self, request, response):
        for info in self.catalog.backend_info():
            msg = BackendInfo()
            msg.name = info['name']
            msg.capabilities = list(info['capabilities'])
            msg.connected = bool(info['connected'])
            msg.message = info['message'] or ''
            response.backends.append(msg)
        return response

    def list_parts_callback(self, request, response):
        """Return catalogued parts matching the request's filters."""
        parts = self.catalog.db.list_parts(
            search=request.filter or None, maturity=request.maturity_filter or None,
            include_archived=request.include_archived, backend=request.backend or None)
        response.parts = [self._to_summary(part) for part in parts]
        response.total_count = len(response.parts)
        return response

    def _to_summary(self, part):
        """Convert a catalog row into a PartSummary message."""
        summary = PartSummary()
        summary.backend = part['backend']
        summary.part_id = part['part_id']
        summary.title = part.get('title') or ''
        summary.part_number = part.get('part_number') or ''
        summary.revision = part.get('revision') or ''
        summary.maturity = part.get('maturity') or ''
        summary.container = part.get('container') or ''
        summary.sync_status = part.get('sync_status') or ''
        # Resolved for this process's mount point.
        geometry = self.catalog.resolved_geometry_path(part)
        thumb = self.catalog.resolved_thumb_path(part)
        summary.geometry_path = str(geometry) if geometry else ''
        summary.geometry_cached = geometry is not None
        summary.geometry_available = bool(part.get('geometry_available'))
        summary.thumbnail_path = str(thumb) if thumb else ''
        # Local-only: this runs for every row.
        status, _ = self.catalog.ensure_plan(part['backend'], part['part_id'],
                                             check_remote=False)
        summary.plan_status = status.value
        summary.last_synced = part.get('last_synced') or ''
        return summary

    def select_part_callback(self, request, response):
        """Fetch a part's geometry, resolve its plan, and announce the selection."""
        backend = self._backend_name(request.backend)
        self.get_logger().info(f'SelectPart: {backend}/{request.part_id}')
        selection, message = self.catalog.select(
            backend, request.part_id, skip_plan_check=request.skip_plan_check)
        if selection is None:
            response.success = False
            response.error_message = message
            self.get_logger().error(f'SelectPart failed: {message}')
            return response
        response.success = True
        response.geometry_path = selection['geometry_path']
        response.plan_status = selection['plan_status'].value
        response.plan_file_path = selection['plan_path']
        response.plan_dir = selection['plan_dir']
        response.error_message = ''
        self._publish_selection(selection)
        return response

    def sync_now_callback(self, request, response):
        """Trigger a sync and wait for it, so the caller gets real counts."""
        results, message = self.sync_blocking(request.full, request.backend or None)
        if not results:
            failed = SyncResultMsg()
            failed.backend = request.backend
            failed.success = False
            failed.message = message
            response.results = [failed]
            return response
        response.results = [self._sync_msg(result) for result in results]
        return response

    def record_plan_callback(self, request, response):
        """Register a plan the producer just wrote.

        Uploading is decided by `auto_upload_plan_stages` unless the caller
        forces it, so a full pipeline run leaves one remote document rather
        than one per stage.
        """
        backend = self._backend_name(request.backend)
        plan_id, message = self.catalog.record_plan(
            backend, request.part_id, request.plan_path, stage=request.stage,
            stage_rank=request.stage_rank, summary_json=request.summary_json,
            summary_text=request.summary_text, change_token=request.change_token,
            generated_at=request.generated_at,
            upload=True if request.force_upload else None)
        response.success = plan_id is not None
        response.plan_id = plan_id or ''
        response.message = message
        if plan_id is None:
            self.get_logger().error(f'RecordPlan failed: {message}')
            return response
        plan = self.catalog.db.get_plan(plan_id) or {}
        response.stage = plan.get('stage') or ''
        response.upload_status = plan.get('upload_status') or 'local'
        self.get_logger().info(f'RecordPlan: {message}')
        return response

    def upload_plan_callback(self, request, response):
        """Upload a recorded plan through its part's backend."""
        self.get_logger().info(f'UploadPlan: {request.plan_id}')
        ref, message = self.catalog.upload_recorded_plan(request.plan_id)
        response.success = ref is not None
        response.remote_ref = ref or ''
        response.message = message
        (self.get_logger().info if ref else self.get_logger().error)(message)
        return response

    def upload_results_callback(self, request, response):
        """Upload an inspection run's result bundle."""
        self.get_logger().info(f'UploadResults: {request.run_id}')
        ref, message = self.catalog.upload_results(request.run_id)
        response.success = ref is not None
        response.remote_ref = ref or ''
        response.message = message
        (self.get_logger().info if ref else self.get_logger().error)(message)
        return response

    def ensure_plan_callback(self, request, response):
        """Report whether a usable inspection plan exists for a part."""
        status, plan_path = self.catalog.ensure_plan(self._backend_name(request.backend),
                                                     request.part_id)
        response.status = status.value
        response.plan_file_path = plan_path or ''
        response.message = PLAN_STATUS_MESSAGES.get(status, '')
        return response


def main(args=None):
    """Spin the catalog node with a multi-threaded executor."""
    rclpy.init(args=args)
    node = CatalogNode()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if node.picker_server is not None:
            node.picker_server.shutdown()
        node.catalog.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
