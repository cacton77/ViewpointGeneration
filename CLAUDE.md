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
`viewpoint_generation_node.py`, `_declare_sync_parameters` in
`catalog/ros_node.py`), so **adding a field to the dataclass and its
`to_dict()` is sufficient** -- no node changes are needed. Follow this for any
new config class.

`DXConfig` is the deliberate exception: it holds a password and must never be
exposed as a ROS parameter or written into a results file. Keep it without a
`to_dict()`, and use `redacted()` when logging it.

## 3DX client: tenant API surface varies

The 3DEXPERIENCE REST surface differs substantially between tenants and
platform releases. Before assuming an endpoint exists, check
`docs/3dx_integration/TENANT_API_FINDINGS.md` in the `inspection-docker` repo,
which records what this tenant actually exposes and what it does not.

Two behaviours to keep in mind when touching `catalog/client.py`:

- **HTTP 200 does not mean the write happened.** Several endpoints accept a
  request with the wrong method or shape, return 200, and do nothing (the
  file check-in via PUT is the known case). Judge success on the returned
  object, not the status code.
- **An expired session is not a 401.** 3DX redirects to the passport login
  page and returns HTML with status 200. `_looks_like_login_redirect()` is what
  detects this; keep new request paths going through `_request()` so they
  inherit the re-login-and-retry behaviour.

Methods that depend on services a tenant may not expose should degrade to an
empty result and log, rather than raising -- a missing derived-output service
must not fail a whole catalog sync.

## Cache paths are environment-relative

`catalog.db` is read from both the host (`./catalog`) and the container
(`/workspaces/shared_ws/catalog`), so absolute paths stored in it are not
portable between the two. Resolve cached STEP/thumbnail paths through
`CatalogSync.resolved_step_path()` / `resolved_thumb_path()` (which prefer the
canonical location for the current environment) rather than trusting the stored
column directly.

## Reading model files

Never call `o3d.io.read_triangle_mesh()` on a path that could be a model the
operator selected. It cannot parse STEP and signals that failure by returning
an *empty mesh* rather than raising, so the caller silently displays nothing.
Use `mesh_utils.read_mesh_file()`, which dispatches on the extension and
returns `(mesh, error)` in the file's own units. The exceptions are fixed
package assets that are always STL (e.g. `planning_volume.stl`).

## STEP loading invariants

If you touch `step_loader.py`, preserve these:

- The vertex weld (`merge_close_vertices`) must stay. Without it, adjacent
  B-rep faces share no vertex indices and `RegionGrowing` cannot grow a region
  across a face boundary.
- `tri_to_brep` and `BRepFace.triangle_indices` must be rebuilt **after** the
  weld, because welding renumbers and can drop triangles.
- B-rep face identity must come from OCC's own shape map, not Python `id()` or
  `hash()`: OCP returns a fresh wrapper object per access, so Python identity
  differs between two handles to the same face.
