"""The contract between the catalog and whatever produces inspection plans.

The catalog never opens a plan file to learn anything about it. The producer
(ViewpointGeneration) owns the plan format; the catalog stores, versions and
uploads the file as an opaque blob, and learns everything it records about a
plan from the producer:

* **On selection** the catalog publishes, with `PartSelected`, the PLM
  context it authored for the part (`plm_context_json`) and the directory new
  plans for that part belong in (`plan_dir`). The producer embeds that object
  as each plan's `plm_context`, adding only `stage`, `generated_at` and
  `software_version`.
* **On recording** (`RecordPlan`) the producer states the stage label, its
  rank (higher = further through the pipeline), a summary (stored verbatim)
  with a one-line form for the picker, the change token the plan was
  generated against, and when it was generated.
* **On adoption** of plan files already on disk, the catalog lists file names
  only and asks the producer to describe them (`DescribePlans`).

The one exception is a plan *downloaded* from the PLM, which arrives with no
producer to ask: there the catalog reads back the `plm_context` block it
authored itself (`read_context()`), and nothing else in the file.
"""

import json
import logging

logger = logging.getLogger(__name__)


def build_context(part, cell_id='', generated_by=''):
    """The PLM context the catalog publishes with a part selection.

    Args:
        part: A catalog `parts` row (or any mapping with its fields).
        cell_id: This cell's identifier.
        generated_by: The PLM account the catalog acts as.

    Returns:
        dict: the context block, without the producer's own keys.
    """
    part = part or {}
    return {
        'eng_item_id': part.get('eng_item_id'),
        'part_number': part.get('part_number'),
        'revision': part.get('revision'),
        'cestamp': part.get('cestamp'),
        'collab_space': part.get('collab_space'),
        'title': part.get('title'),
        'plan_doc_id': None,
        'cell_id': cell_id,
        'generated_by': generated_by,
    }


def context_json(part, cell_id='', generated_by=''):
    """`build_context()` serialized for `PartSelected.plm_context_json`."""
    return json.dumps(build_context(part, cell_id, generated_by), sort_keys=True)


def read_context(path):
    """The `plm_context` block of a plan file downloaded from the PLM.

    Reads the one key the catalog authored and ignores the rest of the file.
    Accepts both the current form (context beside the payload) and the older
    `{plm_context, plan}` envelope, which keep it under the same key.

    Returns:
        dict: the context, empty when the file is unreadable or has none.
    """
    try:
        with open(path) as handle:
            data = json.load(handle)
    except (OSError, ValueError) as e:
        logger.debug('Could not read plan context from %s: %s', path, e)
        return {}
    if not isinstance(data, dict):
        return {}
    context = data.get('plm_context')
    return dict(context) if isinstance(context, dict) else {}
