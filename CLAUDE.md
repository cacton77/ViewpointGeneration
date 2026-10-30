# CLAUDE.md

## README Maintenance

When making changes to this package (adding/removing/modifying modules, ROS interfaces, parameters, dependencies, launch files, or the JSON output format), update `README.md` to reflect those changes. This includes but is not limited to:

- New or removed Python modules in `viewpoint_generation/` or `nodes/`
- Changes to ROS services, topics, parameters, or actions
- New or modified configuration parameters and their defaults
- Changes to launch file arguments
- Dependency additions or removals
- Changes to the results JSON structure

## Config dataclass convention

Algorithm and subsystem configuration lives in dataclasses exposing a
`to_dict()` that returns `{field: {value, type, description, control, range}}`.
The ROS nodes auto-declare parameters from it (`_auto_declare_parameters` in
`viewpoint_generation_node.py`), so **adding a field to the dataclass and its
`to_dict()` is sufficient** -- no node changes are needed. Follow this for any
new config class.

## The PLM catalog lives in plm_bridge

The parts catalog, part picker, 3DEXPERIENCE client and STEP loader moved to
the plm_bridge repository (packages `plm_bridge`, `plm_bridge_ros`,
`plm_interfaces`, `cad_loader`); its `CLAUDE.md` holds their rules (secrets,
the 3DX client's traps, cache paths, STEP invariants). This package's side of
the contract:

- Plans for a selected part go to the selection's `plan_dir`, and their
  `plm_context` is the selection's `plm_context_json` plus only `stage`,
  `generated_at` and `software_version` (`plan_io.stamp`). Do not add other
  PLM keys here: the bridge authors them.
- Everything the bridge records about a plan comes from
  `plan_io.describe_plan()` (sent in `RecordPlan`, and served by
  `describe_plans`). The bridge never opens plan files, so a new plan field
  the catalog should show belongs in `summarize()`/`describe()`.
- `plan_io.read_plan()` must keep reading schema 2.0 (mapping `eng_item_id`
  -> `part_id`, `cestamp` -> `change_token`, `backend: dx3`), the 1.0
  envelope, and bare results JSON.

## Reading model files

Never call `o3d.io.read_triangle_mesh()` on a path that could be a model the
operator selected. It cannot parse STEP and signals that failure by returning
an *empty mesh* rather than raising, so the caller silently displays nothing.
Use `mesh_utils.read_mesh_file()` (re-exported from `cad_loader.mesh_io`),
which dispatches on the extension and returns `(mesh, error)` in the file's
own units. Scale the result by `mesh_utils.mesh_file_units(path,
configured_units)`, not by the configured units: for a STEP file it is always
`'m'`. The exceptions are fixed package assets that are always STL (e.g.
`planning_volume.stl`).

STEP loading itself is `cad_loader.step_loader`; its invariants (the vertex
weld, rebuilding `tri_to_brep` after it, metres, the OCP 7/8 fallbacks, face
identity from OCC's shape map) are in that package's README.
