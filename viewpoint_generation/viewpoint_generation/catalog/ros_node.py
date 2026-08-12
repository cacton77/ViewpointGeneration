"""ROS 2 node exposing the 3DEXPERIENCE parts catalog to the inspection cell.

Wraps `CatalogSync` in services and topics so the rest of the cell -- the
viewpoint generation node, the GUI, the part picker -- can browse the catalog,
select a part, and trigger synchronization without knowing anything about 3DX.

Services (all namespaced under the node name, `catalog`):

    catalog/list_parts   -- browse the local catalog
    catalog/select_part  -- fetch the STEP, resolve the plan, announce it
    catalog/sync_now     -- run a sync on demand
    catalog/ensure_plan  -- report plan validity for one part

Topics:

    /catalog/part_selected  -- PartSelected, latched for late joiners
    /catalog/sync_complete  -- SyncResult after every sync pass

The node deliberately is NOT a lifecycle node: it has to interoperate with the
existing non-lifecycle nodes, and the only state worth managing (the sync
timer) is simple enough to own directly.

Network work never runs on the executor thread. Syncs execute on a background
worker so service calls and the sync timer stay responsive even when the tenant
is slow, and every service handler that touches the network is registered in a
reentrant callback group.
"""

import threading

import rclpy
import rclpy.node
from rcl_interfaces.msg import (FloatingPointRange, IntegerRange,
                                ParameterDescriptor, SetParametersResult)
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from viewpoint_generation_interfaces.msg import PartSelected, PartSummary
from viewpoint_generation_interfaces.msg import SyncResult as SyncResultMsg
from viewpoint_generation_interfaces.srv import (EnsurePlan, ListParts,
                                                 SelectPart, SyncNow,
                                                 UploadPlan, UploadResults)

from viewpoint_generation.catalog.client import DXClient
from viewpoint_generation.catalog.config import CatalogPaths, DXConfig, SyncConfig
from viewpoint_generation.catalog.jms import JMSEventListener
from viewpoint_generation.catalog.sync import CatalogSync, PlanStatus


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

    def __init__(self):
        node_name = 'catalog'
        super().__init__(node_name)

        self.dx_config = DXConfig.from_env()
        self.sync_config = SyncConfig.from_env()
        self.paths = CatalogPaths.from_env()

        # SyncConfig follows the repo's to_dict() convention, so its fields
        # become ROS parameters automatically -- adding a field to the
        # dataclass is enough, exactly as for the algorithm configs.
        self._declare_sync_parameters()

        success, message = self.paths.ensure()
        if not success:
            self.get_logger().error(message)

        self.client = DXClient(self.dx_config)
        self.catalog = CatalogSync(self.client, self.paths.db_path,
                                   self.sync_config, self.paths)

        configured, message = self.dx_config.is_configured()
        if configured:
            self.get_logger().info(
                f'3DX catalog configured for {self.dx_config.username} '
                f'at {self.dx_config.space_url}')
        else:
            # A missing tenant configuration is not fatal: everything already
            # cached locally still browses and selects fine, and the operator
            # gets a clear reason why syncing will not work.
            self.get_logger().warn(
                f'{message} The catalog will serve locally cached parts only.')

        # Serialize sync passes; a second request while one is running is
        # rejected rather than queued, so the tenant never sees overlapping
        # enumerations from this cell.
        self._sync_lock = threading.Lock()
        self._sync_thread = None

        services_cb_group = ReentrantCallbackGroup()
        timer_cb_group = MutuallyExclusiveCallbackGroup()

        self.create_service(ListParts, f'{node_name}/list_parts',
                            self.list_parts_callback, callback_group=services_cb_group)
        self.create_service(SelectPart, f'{node_name}/select_part',
                            self.select_part_callback, callback_group=services_cb_group)
        self.create_service(SyncNow, f'{node_name}/sync_now',
                            self.sync_now_callback, callback_group=services_cb_group)
        self.create_service(EnsurePlan, f'{node_name}/ensure_plan',
                            self.ensure_plan_callback, callback_group=services_cb_group)
        self.create_service(UploadPlan, f'{node_name}/upload_plan',
                            self.upload_plan_callback, callback_group=services_cb_group)
        self.create_service(UploadResults, f'{node_name}/upload_results',
                            self.upload_results_callback,
                            callback_group=services_cb_group)

        # Latched: a picker or GUI that starts after a selection still learns
        # which part is loaded.
        latched = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.part_selected_pub = self.create_publisher(
            PartSelected, '/catalog/part_selected', latched)
        self.sync_complete_pub = self.create_publisher(
            SyncResultMsg, '/catalog/sync_complete', 10)

        self.sync_timer = None
        self._timer_cb_group = timer_cb_group

        # Event-driven sync when a broker is configured and reachable, polling
        # otherwise. The listener reports whether it actually connected, so the
        # fallback is decided on fact rather than on configuration alone.
        self.jms = None
        self.event_driven = False
        self._start_jms_listener()
        self._restart_sync_timer(timer_cb_group)

        self.add_on_set_parameters_callback(self.parameter_callback)

        if self.get_parameter('catalog.sync_on_startup').value and configured:
            self.get_logger().info('Running startup catalog sync...')
            self._start_sync(full=True)

        self.get_logger().info(
            f'Catalog node ready: {self.catalog.db.counts()["total"]} parts '
            f'in {self.paths.db_path}')

    # --- parameters --------------------------------------------------------

    def _declare_sync_parameters(self):
        """Declare SyncConfig fields plus node-level knobs as ROS parameters."""
        params = [
            (f'catalog.{name}', info['value'], _make_descriptor(info))
            for name, info in self.sync_config.to_dict().items()
        ]
        self.declare_parameters(namespace='', parameters=params)
        for ros_name, _, _ in params:
            setattr(self.sync_config, ros_name[len('catalog.'):],
                    self.get_parameter(ros_name).value)

        self.declare_parameters(namespace='', parameters=[
            ('catalog.sync_on_startup', True),
            ('catalog.auto_select_units', 'mm'),
        ])

    def parameter_callback(self, params):
        """Apply parameter changes to the live sync configuration."""
        for param in params:
            if not param.name.startswith('catalog.'):
                continue
            field = param.name[len('catalog.'):]
            if field in self.sync_config.to_dict():
                setattr(self.sync_config, field, param.value)
                self.get_logger().info(f'{param.name} set to {param.value}.')
                if field == 'sync_interval':
                    self._restart_sync_timer()
        return SetParametersResult(successful=True)

    def _start_jms_listener(self):
        """Start the JMS event listener when a broker URL is configured."""
        broker_url = self.dx_config.jms_broker_url
        if not broker_url:
            self.get_logger().info(
                'No DX_JMS_BROKER_URL set; using timer-based catalog polling.')
            return

        self.jms = JMSEventListener(broker_url, self.catalog,
                                    on_event=self._on_jms_event)
        started, message = self.jms.start()
        self.event_driven = started
        if started:
            self.get_logger().info(message)
        else:
            self.get_logger().warn(message)

    def _on_jms_event(self, event, payload):
        """Publish a sync summary after an event-driven catalog change."""
        result = self.catalog.last_result
        if result is not None:
            self._publish_sync_result(result)
        self.get_logger().debug(f'Handled 3DX event {event}.')

    def _restart_sync_timer(self, callback_group=None):
        """(Re)arm the background sync timer from the configured interval.

        With a live JMS listener the timer becomes a slow safety net rather
        than the primary refresh path: events cover ordinary changes, and an
        occasional full pass catches anything the broker dropped.
        """
        if self.sync_timer is not None:
            self.sync_timer.cancel()
            self.destroy_timer(self.sync_timer)
            self.sync_timer = None
        interval = self.sync_config.sync_interval
        if self.event_driven and interval and interval > 0:
            interval = max(interval, 1800)
        if interval and interval > 0:
            self.sync_timer = self.create_timer(
                float(interval), self._timer_sync,
                callback_group=callback_group or self._timer_cb_group)
            self.get_logger().info(
                f'Background sync every {interval}s'
                f'{" (safety net; events drive updates)" if self.event_driven else ""}.')
        else:
            self.get_logger().info('Background sync disabled (interval = 0).')

    # --- sync --------------------------------------------------------------

    def _timer_sync(self):
        """Timer tick: kick off an incremental sync if none is running."""
        configured, _ = self.dx_config.is_configured()
        if configured:
            self._start_sync(full=False)

    def _start_sync(self, full):
        """Run a sync on a background thread.

        Returns:
            bool: False when a sync is already in flight.
        """
        if not self._sync_lock.acquire(blocking=False):
            self.get_logger().debug('Sync already running; skipping this trigger.')
            return False

        def worker():
            try:
                result = (self.catalog.run_full_sync() if full
                          else self.catalog.run_incremental_sync())
                self._publish_sync_result(result)
                level = self.get_logger().info if result.success else self.get_logger().error
                level(f'{"Full" if full else "Incremental"} sync: {result.message}')
            except Exception as e:  # noqa: BLE001 - a worker must never die silently
                self.get_logger().error(f'Sync failed: {e}')
            finally:
                self._sync_lock.release()

        self._sync_thread = threading.Thread(
            target=worker, name='catalog-sync', daemon=True)
        self._sync_thread.start()
        return True

    def _publish_sync_result(self, result):
        """Publish a SyncResult message describing a completed pass."""
        msg = SyncResultMsg()
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
        self.sync_complete_pub.publish(msg)

    # --- service handlers ---------------------------------------------------

    def list_parts_callback(self, request, response):
        """Return catalogued parts matching the request's filters."""
        parts = self.catalog.db.list_parts(
            search=request.filter or None,
            maturity=request.maturity_filter or None,
            collab_space=self.sync_config.collab_space_filter or None,
            include_archived=request.include_archived)
        response.parts = [self._to_summary(part) for part in parts]
        response.total_count = len(response.parts)
        return response

    def _to_summary(self, part):
        """Convert a catalog row into a PartSummary message."""
        summary = PartSummary()
        summary.eng_item_id = part['eng_item_id'] or ''
        summary.title = part.get('title') or ''
        summary.part_number = part.get('part_number') or ''
        summary.revision = part.get('revision') or ''
        summary.maturity = part.get('maturity') or ''
        summary.collab_space = part.get('collab_space') or ''
        summary.sync_status = part.get('sync_status') or ''
        # Resolved for this process's mount point: a catalog.db written by a
        # host-side sync records host paths, which do not exist in here.
        step_path = self.catalog.resolved_step_path(part)
        thumb_path = self.catalog.resolved_thumb_path(part)
        summary.step_path = str(step_path) if step_path else ''
        summary.step_cached = step_path is not None
        summary.step_available = bool(part.get('step_available'))
        summary.thumbnail_path = str(thumb_path) if thumb_path else ''
        # Local-only: this runs for every row, and a per-row 3DX round trip
        # would make listing the catalog unusably slow.
        status, _ = self.catalog.ensure_plan(part['eng_item_id'], check_remote=False)
        summary.plan_status = status.value
        summary.last_synced = part.get('last_synced') or ''
        return summary

    def select_part_callback(self, request, response):
        """Fetch a part's STEP, resolve its plan, and announce the selection."""
        eng_item_id = request.eng_item_id
        self.get_logger().info(f'SelectPart: {eng_item_id}')

        part = self.catalog.db.get_part(eng_item_id)
        if part is None:
            response.success = False
            response.error_message = f'Unknown part: {eng_item_id}.'
            self.get_logger().error(response.error_message)
            return response

        step_path, message = self.catalog.fetch_step(eng_item_id)
        if step_path is None:
            response.success = False
            response.error_message = message
            self.get_logger().error(f'SelectPart failed: {message}')
            return response

        plan_status = PlanStatus.NONE
        plan_path = ''
        if not request.skip_plan_check:
            plan_status, plan_path = self.catalog.ensure_plan(eng_item_id)

        response.success = True
        response.step_file_path = str(step_path)
        response.plan_status = plan_status.value
        response.plan_file_path = plan_path or ''
        response.error_message = ''

        msg = PartSelected()
        msg.eng_item_id = eng_item_id
        msg.title = part.get('title') or ''
        msg.part_number = part.get('part_number') or ''
        msg.revision = part.get('revision') or ''
        msg.cestamp = part.get('cestamp') or ''
        msg.step_file_path = str(step_path)
        msg.mesh_units = self.get_parameter('catalog.auto_select_units').value
        msg.plan_status = plan_status.value
        msg.plan_file_path = plan_path or ''
        self.part_selected_pub.publish(msg)

        self.get_logger().info(
            f"Selected {part.get('title')} -> {step_path} (plan: {plan_status.value})")
        return response

    def sync_now_callback(self, request, response):
        """Trigger a sync and wait for it, so the caller gets real counts."""
        configured, message = self.dx_config.is_configured()
        if not configured:
            response.result.success = False
            response.result.message = message
            return response

        if not self._start_sync(full=request.full):
            response.result.success = False
            response.result.message = 'A catalog sync is already running.'
            return response

        # The worker owns the lock for the duration of the pass; joining it
        # keeps this service synchronous, which is what the picker's Sync
        # button expects. The handler runs in a reentrant group, so other
        # services stay live meanwhile.
        if self._sync_thread is not None:
            self._sync_thread.join()

        result = self.catalog.last_result
        if result is None:
            response.result.success = False
            response.result.message = 'Sync produced no result.'
            return response

        response.result.success = bool(result.success)
        response.result.added = int(result.added)
        response.result.modified = int(result.modified)
        response.result.unchanged = int(result.unchanged)
        response.result.archived = int(result.archived)
        response.result.errors = int(result.errors)
        response.result.duration_sec = float(result.duration_sec)
        response.result.started_at = result.started_at or ''
        response.result.completed_at = result.completed_at or ''
        response.result.message = result.message or ''
        return response

    def upload_plan_callback(self, request, response):
        """Wrap a results JSON in a PLM envelope and upload it to 3DX."""
        self.get_logger().info(
            f'UploadPlan: {request.eng_item_id} <- {request.results_json_path}')
        doc_id, message = self.catalog.upload_plan(
            request.eng_item_id, request.results_json_path)
        response.success = doc_id is not None
        response.plan_doc_id = doc_id or ''
        response.message = message
        if response.success:
            self.get_logger().info(message)
        else:
            self.get_logger().error(message)
        return response

    def upload_results_callback(self, request, response):
        """Upload an inspection run's result bundle to 3DX."""
        self.get_logger().info(f'UploadResults: {request.run_id}')
        doc_id, message = self.catalog.upload_results(request.run_id)
        response.success = doc_id is not None
        response.result_doc_id = doc_id or ''
        response.message = message
        if response.success:
            self.get_logger().info(message)
        else:
            self.get_logger().error(message)
        return response

    def ensure_plan_callback(self, request, response):
        """Report whether a usable inspection plan exists for a part."""
        status, plan_path = self.catalog.ensure_plan(request.eng_item_id)
        response.status = status.value
        response.plan_file_path = plan_path or ''
        response.message = {
            PlanStatus.CURRENT: 'Plan matches the current CAD revision.',
            PlanStatus.DOWNLOADED: 'Plan downloaded from 3DX.',
            PlanStatus.UPDATED_FROM_REMOTE: 'A newer plan was downloaded from 3DX.',
            PlanStatus.STALE: ('Plan was generated for an earlier CAD revision; '
                               're-planning is recommended.'),
            PlanStatus.NONE: 'No inspection plan exists for this part.',
        }[status]
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
        if node.jms is not None:
            node.jms.stop()
        node.client.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
