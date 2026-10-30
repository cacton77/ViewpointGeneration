"""Configuration for the catalog core.

`CoreConfig` follows the repo's config-dataclass convention: `to_dict()`
returns `{field: {value, type, description, control, range}}`, and the ROS
node auto-declares every field as a `catalog.<field>` parameter, so adding a
field here is all a new setting needs. It holds no secrets: credentials
belong to each backend's own configuration, which never has a `to_dict()`.

Everything loads from the environment the cell has always used (`CATALOG_*`,
`CELL_ID`, `PICKER_PORT`), plus `PLM_BACKENDS`.
"""

import os
from dataclasses import dataclass


def env(name, default=''):
    """Read an environment variable, treating unset and empty as the default."""
    value = os.environ.get(name, '')
    return value if value != '' else default


def env_int(name, default):
    """Read an integer environment variable, falling back on a bad value."""
    try:
        return int(env(name, default))
    except ValueError:
        return default


def env_float(name, default):
    """Read a float environment variable, falling back on a bad value."""
    try:
        return float(env(name, default))
    except ValueError:
        return default


def env_flag(name, default):
    """Read a boolean environment variable ('0'/'false' are false)."""
    raw = env(name, '')
    if raw == '':
        return default
    return raw not in ('0', 'false', 'False', 'no')


@dataclass
class CoreConfig:
    """Cadence and policy of the catalog core. No secrets."""

    # Comma-separated registry names of the PLM backends to host.
    backends: str = 'dx3'
    # Seconds between background incremental syncs. 0 disables the timer.
    sync_interval: int = 300
    # Run a full sync when the node starts.
    sync_on_startup: bool = True
    # Comma-separated maturity states to keep, e.g. 'RELEASED,IN_WORK'.
    # Empty means no maturity filtering.
    maturity_filter: str = ''
    # Stage labels whose plans upload automatically when they are recorded.
    # Every recorded plan stays on the cell regardless; this controls only
    # which become remote attachments. The default uploads just the terminal
    # plan -- the one the cell executes. Empty leaves uploading manual.
    auto_upload_plan_stages: str = 'ordered'
    # Anomaly score above which an uploaded inspection result raises an issue.
    ncr_threshold: float = 0.8
    # Identity of this inspection cell, stamped into plans and results.
    cell_id: str = 'alpha'
    # Render a preview locally from cached STEP whenever geometry arrives.
    render_thumbnails: bool = True
    # Units announced with a selection. STEP files declare their own units and
    # the loader ignores this for them; it matters only for other formats.
    mesh_units: str = 'mm'
    # Prepended to the title of everything uploaded. Empty in production; a
    # test tenant run sets it so its objects are recognisable.
    title_prefix: str = ''
    # Catalog root directory (database and caches).
    root: str = '/catalog'
    # Serve the browser part picker from the node, and on which port.
    picker_enabled: bool = False
    picker_port: int = 5050
    # ROS namespace of the catalog's services and topics
    # (`/<namespace>/part_selected`, `<namespace>/record_plan`, ...).
    namespace: str = 'catalog'

    @classmethod
    def from_env(cls):
        """Build a CoreConfig from the environment."""
        return cls(
            backends=env('PLM_BACKENDS', 'dx3'),
            sync_interval=env_int('CATALOG_SYNC_INTERVAL', 300),
            sync_on_startup=env_flag('CATALOG_SYNC_ON_STARTUP', True),
            maturity_filter=env('CATALOG_MATURITY_FILTER', env('DX_MATURITY_FILTER', '')),
            auto_upload_plan_stages=env('CATALOG_AUTO_UPLOAD_PLAN_STAGES', 'ordered'),
            ncr_threshold=env_float('CATALOG_NCR_THRESHOLD', 0.8),
            cell_id=env('CELL_ID', 'alpha'),
            render_thumbnails=env_flag('CATALOG_RENDER_THUMBNAILS', True),
            mesh_units=env('CATALOG_MESH_UNITS', 'mm'),
            title_prefix=env('CATALOG_TITLE_PREFIX', ''),
            root=env('CATALOG_ROOT', '/catalog'),
            picker_enabled=env_flag('PICKER_ENABLED', False),
            picker_port=env_int('PICKER_PORT', 5050),
            namespace=env('CATALOG_NAMESPACE', 'catalog'),
        )

    def backend_names(self):
        """The configured backends, parsed into a list."""
        return [n.strip() for n in self.backends.split(',') if n.strip()]

    def maturity_states(self):
        """The maturity filter parsed into an upper-cased list (empty = all)."""
        return [s.strip().upper() for s in self.maturity_filter.split(',') if s.strip()]

    def auto_upload_stages(self):
        """Plan stages to auto-upload, parsed into a lower-cased list."""
        return [s.strip().lower()
                for s in self.auto_upload_plan_stages.split(',') if s.strip()]

    def to_dict(self):
        return {
            'backends': {
                'value': self.backends,
                'type': 'string',
                'description': "Comma-separated PLM backends to host, e.g. 'dx3'; read at startup",
                'control': 'text',
            },
            'sync_interval': {
                'value': self.sync_interval,
                'type': 'integer',
                'description': 'Seconds between background incremental syncs (0 disables)',
                'control': 'slider',
                'range': [0, 3600],
            },
            'sync_on_startup': {
                'value': self.sync_on_startup,
                'type': 'boolean',
                'description': 'Run a full sync when the node starts',
                'control': 'toggle',
            },
            'maturity_filter': {
                'value': self.maturity_filter,
                'type': 'string',
                'description': ("Comma-separated maturity states to keep (empty = all), "
                                "e.g. 'RELEASED,IN_WORK'"),
                'control': 'text',
            },
            'auto_upload_plan_stages': {
                'value': self.auto_upload_plan_stages,
                'type': 'string',
                'description': ('Comma-separated plan stages that upload automatically when '
                                "recorded (e.g. 'ordered'; empty = manual only). Every plan "
                                'is recorded locally regardless'),
                'control': 'text',
            },
            'ncr_threshold': {
                'value': self.ncr_threshold,
                'type': 'float',
                'description': 'Anomaly score above which an uploaded result raises an issue',
                'control': 'slider',
                'range': [0.0, 1.0],
            },
            'cell_id': {
                'value': self.cell_id,
                'type': 'string',
                'description': ('Identifier of this inspection cell, stamped into plans '
                                'and results'),
                'control': 'text',
            },
            'render_thumbnails': {
                'value': self.render_thumbnails,
                'type': 'boolean',
                'description': 'Render a preview locally from the STEP file when one is cached',
                'control': 'toggle',
            },
            'mesh_units': {
                'value': self.mesh_units,
                'type': 'string',
                'description': ("Units announced with a selection ('m', 'mm', 'cm', 'in'); "
                                'ignored for STEP files, which declare their own'),
                'control': 'text',
            },
            'title_prefix': {
                'value': self.title_prefix,
                'type': 'string',
                'description': 'Prepended to the title of everything uploaded (test runs)',
                'control': 'text',
            },
            'root': {
                'value': self.root,
                'type': 'string',
                'description': 'Catalog root directory (database and caches); read at startup',
                'control': 'text',
            },
            'picker_enabled': {
                'value': self.picker_enabled,
                'type': 'boolean',
                'description': 'Serve the browser part picker from this node; read at startup',
                'control': 'toggle',
            },
            'picker_port': {
                'value': self.picker_port,
                'type': 'integer',
                'description': 'Port the part picker listens on; read at startup',
                'control': 'slider',
                'range': [1, 65535],
            },
            'namespace': {
                'value': self.namespace,
                'type': 'string',
                'description': ('Namespace of the catalog services and topics; '
                                'read at startup'),
                'control': 'text',
            },
        }
