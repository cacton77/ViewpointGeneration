"""Command line for the catalog.

    python -m viewpoint_generation.catalog.cli --full
    python -m viewpoint_generation.catalog.cli --list
    python -m viewpoint_generation.catalog.cli --contexts     # a dx3 option

Parts are named by `part_id`; with more than one backend configured, pass
`--backend NAME` (default: the first configured backend). Backends register
their own options (`add_cli_arguments` / `handle_cli`).
"""

import argparse
import json
import logging
import os
import shlex
import subprocess
import sys

from viewpoint_generation.catalog.backends import registry
from viewpoint_generation.catalog.core.config import CoreConfig
from viewpoint_generation.catalog.factory import build_catalog

logger = logging.getLogger(__name__)

# How the CLI asks the plan producer to describe files during adoption: the
# command-line form of the DescribePlans service. Paths are appended, and it
# must print a JSON list of descriptions (see Catalog.adopt_orphan_plans).
DEFAULT_DESCRIBE_COMMAND = 'python3 -m viewpoint_generation.plan_io describe'


def describe_with_command(paths, command=None):
    """Describe plan files by running the producer's describe command.

    Returns:
        list: one description dict per path; empty when the command fails.
    """
    command = command or os.environ.get('PLAN_DESCRIBE_CMD') or DEFAULT_DESCRIBE_COMMAND
    try:
        completed = subprocess.run(shlex.split(command) + [str(p) for p in paths],
                                   capture_output=True, text=True, timeout=600, check=True)
        descriptions = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as e:
        logger.error('Plan producer could not describe %d file(s) with %r: %s',
                     len(paths), command, e)
        return []
    return descriptions if isinstance(descriptions, list) else []


def _parser(config):
    parser = argparse.ArgumentParser(
        prog='python -m viewpoint_generation.catalog.cli',
        description='Synchronize and inspect the local PLM parts catalog.')
    parser.add_argument('--backend', metavar='NAME',
                        help='Backend the part options refer to (default: the first of '
                             f'PLM_BACKENDS={config.backends!r}).')
    parser.add_argument('--full', action='store_true',
                        help='Run a full sync (enumerate and reconcile everything).')
    parser.add_argument('--incremental', action='store_true',
                        help='Run a lightweight incremental sync.')
    parser.add_argument('--status', action='store_true', help='Print catalog status.')
    parser.add_argument('--list', action='store_true', help='List catalogued parts.')
    parser.add_argument('--backends', action='store_true',
                        help='List hosted backends and their capabilities.')
    parser.add_argument('--render-thumbnails', action='store_true',
                        help='Render previews from cached STEP files for every part '
                             'missing a thumbnail.')
    parser.add_argument('--force', action='store_true',
                        help='With --render-thumbnails, re-render existing ones.')
    parser.add_argument('--sync-item', metavar='PART_ID',
                        help='Add or refresh one part by id, regardless of the scope.')
    parser.add_argument('--fetch-step', metavar='PART_ID',
                        help='Fetch the STEP file for one part.')
    parser.add_argument('--plans', metavar='PART_ID',
                        help='List every inspection plan recorded for one part.')
    parser.add_argument('--adopt-plans', metavar='PART_ID',
                        help='Record plan files already on disk for one part that the '
                             'catalog never captured. The plan producer describes them.')
    parser.add_argument('--describe-cmd', metavar='COMMAND',
                        help='Producer command that describes plan files as JSON '
                             '(default: $PLAN_DESCRIBE_CMD or '
                             f'{DEFAULT_DESCRIBE_COMMAND!r}).')
    parser.add_argument('--record-plan', nargs=2, metavar=('PART_ID', 'PLAN_FILE'),
                        help='Record one plan file in the catalog (needs --stage).')
    parser.add_argument('--stage', help='With --record-plan: the plan stage label.')
    parser.add_argument('--stage-rank', type=int, default=-1,
                        help='With --record-plan: the stage order (higher = later).')
    parser.add_argument('--summary-text', default='',
                        help='With --record-plan: one-line summary for the picker.')
    parser.add_argument('--upload-plan', metavar='PLAN_ID',
                        help='Upload one already-recorded plan.')
    parser.add_argument('--scope', metavar='SCOPE',
                        help="Override the selected backend's scope for this run.")
    parser.add_argument('-v', '--verbose', action='store_true', help='Debug logging.')
    for name in config.backend_names():
        try:
            backend_class = registry.backend_class(name)
        except ImportError:
            backend_class = None
        if backend_class is not None and hasattr(backend_class, 'add_cli_arguments'):
            backend_class.add_cli_arguments(parser)
    return parser


def main(argv=None):
    """Command-line entry point."""
    config = CoreConfig.from_env()
    args = _parser(config).parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(asctime)s %(levelname)-7s %(name)s: %(message)s')

    catalog, errors = build_catalog(config)
    if not catalog.backends and errors:
        return 2
    name = args.backend or (config.backend_names() or [''])[0]
    backend = catalog.backend(name)
    if backend is not None:
        if args.scope:
            backend.scope = args.scope
        if hasattr(backend, 'apply_cli_overrides'):
            backend.apply_cli_overrides(args)

    try:
        return _run(catalog, backend, name, args)
    finally:
        catalog.close()


def _require_remote(backend, name):
    if backend is None:
        logger.error('Backend %r is not configured (PLM_BACKENDS).', name)
        return False
    configured, message = backend.is_configured()
    if not configured:
        logger.error(message)
    return configured


def _run(catalog, backend, name, args):
    # Backend-registered options first; handle_cli returns None when none of
    # its options was given.
    if backend is not None and hasattr(backend, 'handle_cli'):
        code = backend.handle_cli(args)
        if code is not None:
            return code

    if args.backends:
        for info in catalog.backend_info():
            print(f"{info['name']:<8} configured={info['configured']!s:<5} "
                  f"capabilities={','.join(info['capabilities'])}  {info['details']}")
        return 0

    if args.render_thumbnails:
        rendered, failed = catalog.render_missing_thumbnails(force=args.force)
        print(f'Rendered {rendered} thumbnail(s); {failed} could not be rendered.')
        return 0

    if args.sync_item:
        if not _require_remote(backend, name):
            return 2
        success, message = catalog.sync_part(name, args.sync_item)
        print(message)
        if not success:
            return 1

    if args.fetch_step:
        path, message = catalog.fetch_geometry(name, args.fetch_step)
        print(message)
        return 0 if path else 1

    if args.adopt_plans:
        adopted, skipped = catalog.adopt_orphan_plans(
            name, args.adopt_plans,
            describe=lambda paths: describe_with_command(paths, args.describe_cmd))
        print(f'Adopted {adopted} plan(s); skipped {skipped} file(s).')
        return 0

    if args.record_plan:
        part_id, plan_path = args.record_plan
        plan_id, message = catalog.record_plan(
            name, part_id, plan_path, stage=args.stage, stage_rank=args.stage_rank,
            summary_text=args.summary_text)
        print(message)
        return 0 if plan_id else 1

    if args.upload_plan:
        ref, message = catalog.upload_recorded_plan(args.upload_plan)
        print(message)
        return 0 if ref else 1

    if args.plans:
        plans = catalog.list_plans(name, args.plans)
        if not plans:
            print('No plans recorded for this part.')
        for plan in plans:
            flags = ''.join(('*' if plan['is_current'] else ' ',
                             '!' if plan['stale'] else ' ',
                             '?' if not plan['available'] else ' '))
            print(f"{flags} {plan['stage']:<10} {plan['summary_text']:<40.40} "
                  f"{plan['generated_at']}  {plan['upload_status']:<9} {plan['plan_id']}")
        if plans:
            print('\n* current   ! targets an earlier CAD revision   ? file missing')
        return 0

    results = None
    if args.full or args.incremental:
        if not _require_remote(backend, name) and args.backend:
            return 2
        run = catalog.run_full_sync if args.full else catalog.run_incremental_sync
        results = run(args.backend)

    if args.list or (results is None and not args.status):
        parts = catalog.db.list_parts()
        print(f'{len(parts)} part(s) in catalog:')
        for part in parts:
            step = 'cached' if catalog.resolved_geometry_path(part) else (
                'available' if part.get('geometry_available') else 'none')
            print(f"  {part['backend']:<5} {part['part_id']}  {part['title']!r:40} "
                  f"rev {part.get('revision')}  {part.get('maturity')}  "
                  f"[{part.get('container')}]  step={step}")

    if args.status or results is not None:
        status = catalog.status()
        counts = status['counts']
        print(f"Catalog: {counts['total']} parts, {counts['ready']} with a cached STEP")
        print(f"Status breakdown: {counts['by_status']}")
        if status['last_sync']:
            print(f"Last sync: {status['last_sync']['timestamp']} "
                  f"({status['last_sync']['action']}) {status['last_sync']['detail']}")

    if results and not all(result.success for result in results):
        for result in results:
            for detail in result.error_details or [result.message]:
                logger.error('%s: %s', result.backend, detail)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
