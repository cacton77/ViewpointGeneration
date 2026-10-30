"""On-disk conventions for inspection plan files.

An inspection plan is the ViewpointGeneration results JSON. Every stage of the
pipeline -- segmentation, FOV clustering, viewpoint projection, traversal
optimization -- writes one, and each is a complete, loadable plan in its own
right.

**File shape.** The PLM linkage lives in top-level keys *beside* `meshes`,
never wrapping it::

    {
      "schema_version": "2.0",
      "plm_context": {"eng_item_id": ..., "cestamp": ..., "stage": ...},
      "summary":     {"num_regions": 12, "num_clusters": 75, ...},
      "meshes":      [ ... ]
    }

This matters: `task_planning_node`, `visualizer`, `viewpoint_traversal_node`
and `ViewpointGeneration.set_results_file()` all reach straight for
`json.load(...)['meshes']`. A wrapper (the 1.0 `{plm_context, plan: {meshes}}`
envelope) is invisible to all of them and loads as an empty result. Keeping the
context as a sibling means every consumer keeps working untouched, and -- since
the traversal node round-trips the whole dict -- the optimized file inherits
its part linkage for free.

`read_plan()` still accepts the 1.0 envelope so plans uploaded before this
convention existed continue to load.

**PLM context.** The `plm_context` block is authored by the parts catalog and
handed over with each part selection (`PartSelected.plm_context_json`); this
module only adds the producer's own keys -- `stage`, `generated_at` and
`software_version` -- when a plan is written. The catalog never opens plan
files: when it records one, `describe_plan()` is what tells it the stage,
rank, summary, change token and generation time (via the node's
`describe_plans` service, or `python3 -m viewpoint_generation.plan_io describe`).

This module is deliberately dependency-free (no numpy/open3d), so the catalog
package and the ROS nodes can share it without pulling in the heavy geometry
stack.
"""

import functools
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

SCHEMA_VERSION = '2.0'

#: Pipeline stages, in the order they are produced. Later stages supersede
#: earlier ones for the same part.
STAGES = ('segmented', 'clustered', 'projected', 'ordered')

#: Written when a results skeleton exists but segmentation has not run.
STAGE_EMPTY = 'empty'

_TIMESTAMP_FORMAT = '%Y%m%dT%H%M%SZ'


def utc_timestamp():
    """Compact UTC timestamp used in plan filenames."""
    return datetime.now(timezone.utc).strftime(_TIMESTAMP_FORMAT)


def summarize(results):
    """Count what a results dict actually contains.

    Args:
        results: A results dict (the `{'meshes': [...]}` payload).

    Returns:
        dict: num_regions/num_clusters/num_viewpoints plus the mesh identity
        fields recorded alongside them.
    """
    meshes = (results or {}).get('meshes') or [{}]
    mesh = meshes[0] if meshes else {}
    regions = mesh.get('regions') or []
    num_clusters = sum(len(region.get('clusters') or []) for region in regions)
    num_viewpoints = sum(
        1 for region in regions for cluster in (region.get('clusters') or [])
        if 'viewpoint' in cluster)
    return {
        'num_regions': len(regions),
        'num_clusters': num_clusters,
        'num_viewpoints': num_viewpoints,
        'mesh_file': mesh.get('file'),
        'mesh_units': mesh.get('units'),
        'mesh_dimensions': mesh.get('dimensions'),
        'surface_area': mesh.get('surface_area'),
        'source_format': mesh.get('source_format'),
    }


def infer_stage(results):
    """Derive the pipeline stage from a results dict's own content.

    Content is authoritative -- a filename can be renamed, and a plan handed
    over from another cell may follow no convention at all.

    The traversal optimizer is what turns each region's `order` from a plain
    list into an algorithm-keyed dict (`{'greedy': {'order': [...], ...}}`),
    which is the only structural difference between a projected plan and an
    ordered one.

    Returns:
        str: one of STAGES, or STAGE_EMPTY.
    """
    meshes = (results or {}).get('meshes') or [{}]
    mesh = meshes[0] if meshes else {}
    regions = mesh.get('regions') or []
    if not regions:
        return STAGE_EMPTY

    if any(isinstance(region.get('order'), dict) for region in regions):
        return 'ordered'
    if any('viewpoint' in cluster
           for region in regions for cluster in (region.get('clusters') or [])):
        return 'projected'
    if any(region.get('clusters') for region in regions):
        return 'clustered'
    return 'segmented'


def stage_rank(stage):
    """Sort key for a stage; unknown stages sort before everything known."""
    try:
        return STAGES.index(stage)
    except ValueError:
        return -1


def plan_filename(stage, summary, timestamp=None):
    """Build a plan filename that states its stage and size up front.

    Zero counts are omitted, so a segmentation-only plan does not claim
    `0c_0v`::

        segmented_12r_20260814T162427Z.json
        clustered_12r_75c_20260814T162439Z.json
        projected_12r_75c_75v_20260814T162442Z.json
        ordered_12r_75c_75v_20260814T162538Z.json

    Returns:
        str: the bare filename (no directory).
    """
    parts = [stage or STAGE_EMPTY]
    for count, suffix in ((summary.get('num_regions'), 'r'),
                          (summary.get('num_clusters'), 'c'),
                          (summary.get('num_viewpoints'), 'v')):
        if count:
            parts.append(f'{count}{suffix}')
    parts.append(timestamp or utc_timestamp())
    return '_'.join(parts) + '.json'


def unique_path(directory, filename):
    """A path in `directory` that does not exist yet.

    Plan filenames carry a one-second timestamp, and two pipeline stages can
    complete inside the same second (they did in practice), so the name is
    disambiguated rather than silently overwritten.
    """
    candidate = os.path.join(str(directory), filename)
    if not os.path.exists(candidate):
        return candidate
    stem, extension = os.path.splitext(filename)
    for counter in range(2, 1000):
        candidate = os.path.join(str(directory), f'{stem}-{counter}{extension}')
        if not os.path.exists(candidate):
            return candidate
    return candidate


@functools.lru_cache(maxsize=1)
def software_version():
    """`ViewpointGeneration@<short git rev>` of this checkout, for traceability.

    Stamped into every plan so it can be traced back to the code that
    produced it. The revision is 'unknown' outside a git checkout.
    """
    try:
        revision = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL, text=True, timeout=5).strip()
    except (subprocess.SubprocessError, OSError):
        revision = 'unknown'
    return f'ViewpointGeneration@{revision or "unknown"}'


def _utc_iso(timestamp=None):
    """A UTC time (default now) as an ISO-8601 string with a trailing Z."""
    moment = (datetime.fromtimestamp(timestamp, timezone.utc) if timestamp is not None
              else datetime.now(timezone.utc))
    return moment.strftime('%Y-%m-%dT%H:%M:%SZ')


def read_plan(path):
    """Load a plan file into a results dict.

    Accepts all three shapes seen in the wild: the 2.0 sibling form, the 1.0
    `{plm_context, plan}` envelope, and a bare `{'meshes': [...]}` results
    JSON written before either existed.

    Returns:
        tuple: (dict or None, dict, str) the results payload, its plm_context,
        and a message.
    """
    try:
        with open(path) as handle:
            data = json.load(handle)
    except (OSError, ValueError) as e:
        return None, {}, f'Could not read plan file {path}: {e}'

    if not isinstance(data, dict):
        return None, {}, f'Unrecognized plan file structure: {path}'

    # 2.0: context sits beside meshes, so the file *is* the results dict.
    if 'meshes' in data:
        return data, data.get('plm_context') or {}, 'Plan loaded.'

    # 1.0: the results dict is wrapped under 'plan'.
    if 'plan' in data and isinstance(data.get('plan'), dict):
        payload = data['plan']
        context = dict(data.get('plm_context') or {})
        if 'stage' not in context:
            context['stage'] = infer_stage(payload)
        # Carry the context onto the unwrapped payload so a plan read from a
        # 1.0 envelope and re-saved comes back out in 2.0 form.
        payload.setdefault('plm_context', context)
        payload.setdefault('schema_version', SCHEMA_VERSION)
        return payload, context, 'Plan loaded from a 1.0 envelope.'

    return None, {}, f'Unrecognized plan file structure: {path}'


def stamp(results, context=None, stage=None):
    """Fill in a results dict's own schema/context/summary keys in place.

    The context itself comes from the catalog; this adds only the producer's
    keys: the stage, when it was generated (now), and the software version.

    Args:
        results: The results dict, mutated in place.
        context: `plm_context` to stamp in. When None, any context already on
            `results` is kept.
        stage: Override the inferred stage.

    Returns:
        tuple: (str, dict) the resolved stage and summary.
    """
    stage = stage or infer_stage(results)
    summary = summarize(results)

    if context is None:
        context = dict(results.get('plm_context') or {})
    else:
        context = dict(context)
    context['stage'] = stage
    context['generated_at'] = _utc_iso()
    context['software_version'] = software_version()

    results['schema_version'] = SCHEMA_VERSION
    results['plm_context'] = context
    results['summary'] = summary
    return stage, summary


def save_plan_to(path, results, context=None, stage=None):
    """Write a results dict to an exact path, stamping its context first.

    Used to record a plan file in place (adding PLM identity to a file that
    predates it) without renaming it out from under whoever is holding the
    path.

    Returns:
        tuple: (str or None, str) the written path and a message.
    """
    stamp(results, context, stage)
    try:
        parent = os.path.dirname(str(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, 'w') as handle:
            json.dump(results, handle, indent=4)
    except OSError as e:
        return None, f'Could not write plan file: {e}'
    return str(path), f'Plan saved to {path}.'


def write_plan(directory, results, context=None, stage=None, timestamp=None):
    """Write a results dict as a new plan file, named for its stage.

    The stage is taken from `stage` when given, otherwise inferred from the
    content, and is recorded both in the filename and in `plm_context.stage`.

    Args:
        directory: Where to write. Created if absent.
        results: The results dict (mutated in place to carry the context).
        context: `plm_context` to stamp in. When None, any context already on
            `results` is kept.
        stage: Override the inferred stage.
        timestamp: Override the filename timestamp.

    Returns:
        tuple: (str or None, str) the written path and a message.
    """
    stage, summary = stamp(results, context, stage)
    try:
        os.makedirs(str(directory), exist_ok=True)
    except OSError as e:
        return None, f'Could not create plan directory {directory}: {e}'
    path = unique_path(directory, plan_filename(stage, summary, timestamp))
    return save_plan_to(path, results, context=results.get('plm_context'),
                        stage=stage)


def describe(summary):
    """Short human-readable size of a plan, e.g. '12 regions, 75 viewpoints'."""
    bits = []
    if summary.get('num_regions'):
        bits.append(f"{summary['num_regions']} regions")
    if summary.get('num_clusters'):
        bits.append(f"{summary['num_clusters']} clusters")
    if summary.get('num_viewpoints'):
        bits.append(f"{summary['num_viewpoints']} viewpoints")
    return ', '.join(bits) or 'empty'


def legacy_stage_from_name(path):
    """Best-effort stage for a plan file written before this convention.

    Only used to label pre-existing files; content inference wins whenever the
    file can actually be read.
    """
    name = os.path.basename(str(path))
    if '_optimized' in name:
        return 'ordered'
    if re.search(r'\d+_clusters', name):
        return 'clustered'
    if re.search(r'\d+_regions', name):
        return 'segmented'
    return None


def describe_plan(path):
    """Describe a plan file for the catalog: the `DescribePlans` contract.

    This is how the catalog learns about plan files it did not see being
    written (adoption): it lists names, and the producer -- which owns the
    format -- reads them.

    Returns:
        dict: `path`, `valid`, `stage`, `stage_rank`, `summary_json`,
        `summary_text`, `change_token`, `generated_at` and `message`. A file
        that is unreadable or holds no regions is not a plan (`valid` false).
        `change_token` and `generated_at` come from the file's own context,
        falling back to '' (the catalog then uses the part's current token)
        and to the file's modification time.
    """
    description = {'path': str(path), 'valid': False, 'stage': '', 'stage_rank': -1,
                   'summary_json': '{}', 'summary_text': '', 'change_token': '',
                   'generated_at': '', 'message': ''}
    results, context, message = read_plan(path)
    description['message'] = message
    if results is None:
        return description
    summary = summarize(results)
    if not summary['num_regions']:
        description['message'] = 'Not a plan: it holds no regions.'
        return description
    stage = context.get('stage') or infer_stage(results)
    try:
        mtime = _utc_iso(os.stat(path).st_mtime)
    except OSError:
        mtime = ''
    description.update({
        'valid': True,
        'stage': stage,
        'stage_rank': stage_rank(stage),
        'summary_json': json.dumps(summary, sort_keys=True),
        'summary_text': describe(summary),
        'change_token': context.get('change_token') or context.get('cestamp') or '',
        'generated_at': context.get('generated_at') or mtime,
    })
    return description


def main(argv=None):
    """`python3 -m viewpoint_generation.plan_io describe FILE...`

    Prints a JSON list of `describe_plan()` results, which is what the
    catalog CLI's `--adopt-plans` consumes.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] != 'describe':
        print('usage: python3 -m viewpoint_generation.plan_io describe FILE...',
              file=sys.stderr)
        return 2
    print(json.dumps([describe_plan(path) for path in argv[1:]], indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
