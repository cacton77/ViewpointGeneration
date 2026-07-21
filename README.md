# ViewpointGeneration

A ROS 2 package for automated inspection viewpoint generation from CAD models. Given a surface mesh, the library segments the surface into regions, clusters each region to fit within a camera's field of view, and computes collision-free camera viewpoints for full-surface inspection coverage.

**Maintainer:** Colin Acton (actonc@uw.edu)

## Overview

ViewpointGeneration is structured as a standalone Python library (`viewpoint_generation`) with a ROS 2 node wrapper (`ViewpointGenerationNode`) that exposes the library's pipeline through ROS parameters and services.

The pipeline follows these stages:

1. **Load CAD Model** -- Import an STL/OBJ mesh with unit conversion
2. **Surface Segmentation** -- Segment the mesh directly into regions (by triangle face, not a sampled point cloud). Two interchangeable algorithms (selected via `regions.segmentation_algorithm`):
   - `region_growth` -- contiguous regions grown over triangle face-adjacency, using face-normal similarity and a per-face curvature analog (variance of neighboring face normals) as the merge criteria
   - `partfield` -- semantic parts from [PartField](https://github.com/nv-tlabs/PartField), grouped directly from its per-face labels (requires GPU + PartField mounted at `/models/PartField`)
3. **Sample Region Point Clouds** -- Poisson disk sample each region's own submesh (sized from `fov_clustering.point_density`); cached to disk and referenced from the results file
4. **FOV Clustering** -- Subdivide each region's sampled point cloud into clusters that fit within the camera's field of view and depth of field. Two interchangeable algorithms (selected via `regions.fov_clustering.algorithm`):
   - `kmeans` -- K-means with Bayesian optimization for cluster count
   - `greedy_cover` -- greedy set cover: a candidate viewpoint covers a region point only if it passes depth-of-field, field-of-view, and photometric-incidence tests *and* (when `occlusion_check` is enabled) has an unoccluded line of sight to that point against the **full part mesh**, not just the region being clustered. Points the straight-normal predicate leaves occluded get one more chance when `rescue_search` is enabled: a Monte Carlo hemisphere search (see `occlusion_search.py`) looks for any unoccluded viewing angle within `standard_normal_threshold` of that point's normal (angles beyond it are too glancing to image the surface, so they are neither sampled nor accepted), preferring ones close to its true normal. Points still unrescuable are reported per-region as `blind_spots` rather than silently folded into a cluster (see JSON Results Format below). Each emitted cluster carries its winning candidate's occlusion-validated axis forward as `candidate_axis`, for reuse in the next stage. Candidate anchors normally come from blue-noise farthest-point sampling; setting `structured_candidates` instead lays them on a per-region cylindrical grid (elevation = world Z, azimuth = angle around a vertical axis through the part's center, both spaced by `candidate_spacing`) so viewpoints from *different* regions tend to land at shared elevations/azimuths -- e.g. rings around a roughly cylindrical part -- at some cost to the greedy algorithm's freedom to pick the most locally-efficient anchor
5. **Project Viewpoints** -- Compute a camera pose (position + orientation) for each cluster at the configured focal distance. The naive direction (mean of the cluster's own point normals) is refined by the same hemisphere search used above, using `candidate_axis` (when present) as a guaranteed-good starting point so refinement can never do worse than clustering already proved — this closes a gap where the mean direction was never itself re-validated against occlusion. The winning direction is tagged `imaging_mode`: `photometric` if it falls within the photometric incidence cone (`fov_normal_threshold`, supports photometric-stereo normal-map reconstruction), `standard` if only an off-cone angle within `standard_normal_threshold` is occlusion-clear (still gets an image, just not photometric-quality), or `inaccessible` if no angle inside `standard_normal_threshold` is occlusion-clear — directions beyond it are too glancing to image the surface and are neither sampled nor considered valid. FOV clustering and viewpoint projection use **separate** occlusion scenes so the two stages answer different questions. FOV clustering uses a **mesh-only** `RaycastingScene` (self-occlusion only): cluster feasibility is *pose-independent* -- it decides which surface a camera could ever image, regardless of how the part is placed. Viewpoint projection uses a **mesh + ground-plane** scene, with the ground-plane occluder (just below the table/turntable surface, z=0) transformed into the part's current placement (the `object_frame → model_frame` pose), so projection respects the actual environment and no emitted viewpoint's camera position or line-of-sight ends up underground *at the part's current pose*. Because Open3D's `RaycastingScene` is append-only, the ground plane is toggled by rebuilding the scene with or without it (`_make_raycasting_scene(include_ground=...)`)
6. **Optimize Traversal** -- Solve a TSP to order viewpoints for efficient robot motion

Results are saved as a JSON file.

## Package Structure

```
viewpoint_generation/
├── viewpoint_generation/           # Core Python library
│   ├── viewpoint_generation.py     # Main ViewpointGeneration class
│   ├── region_growth.py            # Region growing segmentation
│   ├── partfield_segmentation.py   # PartField part segmentation (GPU, subprocess)
│   ├── fov_clustering.py           # Field-of-view clustering
│   ├── viewpoint_projection.py     # Viewpoint pose computation
│   ├── occlusion_search.py         # Monte Carlo hemisphere occlusion search (blind-spot rescue + direction refinement)
│   ├── mesh_utils.py               # Shared triangle-mesh helpers (submesh extraction)
│   ├── visualizer.py               # Open3D visualization
│   ├── gui_node.py                 # GUI node implementation
│   └── assets/
│       ├── materials.py            # Visualization materials
│       └── planning_volume.stl     # Planning volume mesh
├── nodes/                          # ROS 2 node executables
│   ├── viewpoint_generation_node.py
│   ├── task_planning_node.py
│   ├── viewpoint_traversal_node.py
│   └── gui.py
├── launch/
│   ├── bringup.launch.py           # Full system bringup
│   ├── viewpoint_generation.launch.py
│   └── viewpoint_traversal.launch.py
├── package.xml
├── setup.py
└── requirements.txt
```

A companion package, `viewpoint_generation_interfaces/`, defines the ROS 2 message, service, and action interfaces.

## Dependencies

### Python

Listed in `requirements.txt`:

- numpy (<1.25.0)
- open3d
- pytransform3d
- bayesian-optimization
- matplotlib
- requests

For the optional `partfield` segmentation algorithm: a CUDA GPU and the
[PartField](https://github.com/nv-tlabs/PartField) checkout mounted at
`/models/PartField` (with its `model/model_objaverse.ckpt` checkpoint and
PartField's own dependencies — torch, lightning, torch-scatter, etc.). In this
repo's Docker image these are installed via `docker/requirements.txt` and the
mount is provided by `docker-compose.yaml`.

### ROS 2

- sensor_msgs
- geometry_msgs
- ros2launch
- MoveIt 2 (for viewpoint traversal)

## Library Usage

The core library can be used independently of ROS:

```python
from viewpoint_generation.viewpoint_generation import ViewpointGeneration

vg = ViewpointGeneration()

# Load mesh
vg.set_mesh_file("part.stl", units="mm")

# Configure and run pipeline
# vg.set_segmentation_algorithm("partfield")   # optional: use PartField instead
vg.segment_regions()           # segments the mesh, then samples a point cloud per region
vg.fov_clustering()
vg.project_viewpoints()
```

### Configuration

All algorithm parameters are exposed as dataclass configs with sensible defaults:

**RegionGrowingConfig** -- grows regions over the mesh's triangle face-adjacency
graph (two faces are neighbors only if they share an edge), so growth can
never bridge across a fold, thin wall, or gap the way a spatial-radius search
over a sampled point cloud could. "Curvature" here is a per-face analog: how
much a face's normal deviates from its adjacent faces' normals. Adjacency is
computed against a vertex-welded copy of the mesh (`weld_epsilon`), not the
raw input: STL has no vertex-sharing guarantee, and a mesh exported with each
triangle's vertices stored independently (common for CAD-authored STLs) would
otherwise silently read as disconnected 1-2-triangle islands, one per facet,
with no way to detect the real edges between them.

| Parameter | Default | Description |
|---|---|---|
| `seed_threshold` | 0.1 | Curvature threshold for seed face selection |
| `min_cluster_size` | 10 | Minimum faces per region |
| `max_cluster_size` | 100000 | Maximum faces per region |
| `normal_angle_threshold` | pi/3 | Maximum normal deviation (radians) for region membership |
| `curvature_threshold` | 0.1 | Maximum curvature difference for region membership |
| `weld_epsilon` | 1e-6 m | Distance within which vertices are merged before computing face adjacency |

**Tuning for low-poly / faceted parts.** `seed_threshold`'s default (0.1)
assumes some triangles sit far enough from any edge to read as near-zero
curvature. On a coarse or heavily-faceted mesh (few triangles per flat
facet), *every* triangle can be adjacent to a real edge, so nothing clears
the default threshold and segmentation silently returns zero regions. Raise
`seed_threshold`/`curvature_threshold` to comfortably exceed the mesh's
actual per-face curvature range (print `RegionGrowing.face_curvatures` after
`preprocess_mesh` to check it) and lower `min_cluster_size` to the smallest
facet you still want kept as its own region -- `normal_angle_threshold`
remains what separates genuinely distinct facets, since real part edges are
almost always well above it.

**PartFieldSegmentationConfig** (used when `segmentation_algorithm == 'partfield'`)

| Parameter | Default | Description |
|---|---|---|
| `num_parts` | 12 | Number of parts to segment the mesh into |
| `use_agglo` | True | Agglomerative clustering (spatially connected parts) instead of KMeans |
| `option` | 0 | Face-adjacency graph for agglomerative clustering: 0=naive, 1=face-MST, 2=cc-MST |
| `with_knn` | False | Augment the face-adjacency graph with kNN connections (agglomerative only) |

PartField runs as a subprocess (`partfield_inference.py` then
`run_part_clustering.py`) against an exported copy of the loaded mesh. Its
per-face part labels are grouped directly into regions (a list of triangle
indices per part) with no further projection step. Mesh preprocessing/remeshing
is left disabled so face ordering is preserved. Note: PartField produces no
noise faces (every face is assigned a part).

**FOVClusteringConfig**

| Parameter | Default | Description |
|---|---|---|
| `fov_diameter` | 0.03 m | Camera field of view diameter at focal distance |
| `dof` | 0.02 m | Depth of field |
| `point_density` | 0.5 | Target points per square millimeter, both for sampling each region's point cloud and for K-means cluster evaluation. Only needs to resolve FOV coverage/packing decisions, not final inspection imagery, so this can stay low |
| `lambda_weight` | 1.0 | Weight for out-of-FOV penalty in cost function (K-means only) |
| `beta_weight` | 1.0 | Weight for packing efficiency in cost function (K-means only) |
| `point_weight` | 1.0 | Weight for point positions in K-means |
| `normal_weight` | 1.0 | Weight for point normals in K-means |
| `algorithm` | `kmeans` | Clustering algorithm: `kmeans` (Bayesian-optimised K-means) or `greedy_cover` (greedy set cover) |
| `fov_normal_threshold` | π/4 rad | Max surface-normal incidence angle for photometric-stereo imaging (`greedy_cover` coverage predicate) |
| `standard_normal_threshold` | π/3 rad | Max surface-normal incidence angle for standard imaging; between `fov_normal_threshold` and this angle a viewpoint is standard-tier, and beyond it the view is too glancing to capture surface information and is treated as inaccessible (neither sampled during the hemisphere search nor considered valid) |
| `candidate_spacing` | 0.0 m | Anchor spacing for `greedy_cover` candidate sampling; 0.0 = auto = `fov_diameter/2` |
| `prune_redundant` | `true` | Remove redundant viewpoints after greedy cover (cannot open coverage holes) |
| `rng_seed` | 0 | Random seed for `greedy_cover` candidate sampling (reproducibility) |
| `occlusion_check` | `true` | Require unobstructed line-of-sight to the full part mesh (criterion 4 of the `greedy_cover` coverage predicate). Disabling reproduces the pre-occlusion (criteria 1-3 only) behavior |
| `occlusion_epsilon` | 1e-4 m | Shrink margin subtracted from the occlusion ray's far bound so a point's own triangle is never mistaken for its own occluder |
| `rescue_search` | `true` | Attempt a Monte Carlo hemisphere search for an alternative viewing angle before giving up on an occluded blind-spot point (`greedy_cover` only) |
| `rescue_samples` | 64 | Hemisphere samples per blind-spot rescue attempt |
| `structured_candidates` | `false` | Lay `greedy_cover` candidate anchors on a per-region cylindrical grid (elevation/azimuth around the part center, spaced by `candidate_spacing`) instead of blue-noise farthest-point sampling, so viewpoints across regions tend to share elevations/azimuths |

**ViewpointProjectionConfig**

| Parameter | Default | Description |
|---|---|---|
| `focal_distance` | 0.3 m | Distance from viewpoint to surface |
| `hemisphere_points` | 64 | Hemisphere samples for the occlusion-aware direction-refinement search in `project_viewpoints` (per cluster, not per point) |

## ROS 2 Interface

### ViewpointGenerationNode

Wraps the core library, exposing each pipeline stage as a ROS service and all configuration as ROS parameters.

**Services** (all `std_srvs/srv/Trigger`):

| Service | Description |
|---|---|
| `/viewpoint_generation/segment_regions` | Segment the mesh into regions (region growth or PartField) and sample each region's point cloud |
| `/viewpoint_generation/fov_clustering` | Cluster regions by camera FOV |
| `/viewpoint_generation/viewpoint_projection` | Generate camera viewpoints |
| `/viewpoint_generation/optimize_traversal` | Optimize viewpoint visit order (TSP) |

**Parameters:**

All `RegionGrowingConfig`, `PartFieldSegmentationConfig`, `FOVClusteringConfig`, and `ViewpointProjectionConfig` fields are declared as ROS parameters with type information and valid ranges (prefixed `regions.region_growth.`, `regions.partfield.`, `regions.fov_clustering.`, and `viewpoints.projection.` respectively). Additional parameters:

- `regions.segmentation_algorithm` -- Surface segmentation algorithm: `region_growth` (default) or `partfield`
- `regions.fov_clustering.algorithm` -- FOV clustering algorithm: `kmeans` (default) or `greedy_cover`
- `model.mesh.file` -- Path to the mesh file
- `model.mesh.units` -- Mesh units (`m`, `cm`, `mm`, `in`)
- `model.point_cloud.file` -- Path to a manually-loaded point cloud (optional; independent of the per-region point clouds sampled during segmentation)
- `model.point_cloud.units` -- Point cloud units
  **Part placement is not authored via parameters.** The mesh, regions, and
  viewpoints are always generated and stored in the mesh's own **origin frame**
  (`model_frame`); the mesh file's origin is never moved. Where the part
  physically sits is a live **TF placement**: `tsdf_pose` registers the part and
  broadcasts the `object_frame -> model_frame` transform. That placement is
  applied only to *placement* consumers -- the `/planning_scene` collision
  object pose, the GUI visualization (`Visualizer.apply_model_placement`), and
  the occlusion ground-plane in the raycasting scene
  (`ViewpointGeneration.set_placement`) -- and is never baked into the geometry
  or written to a results file. Downstream viewpoint goals are stamped in
  `model_frame` so MoveIt resolves them through the same TF. (The old
  `model.pose.*` parameters, which baked a pose into the geometry and rigidly
  re-transformed / re-saved results on every `/tsdf_pose/pose` message, have
  been removed.)
- `results.file` -- Path to results JSON
- `settings.data_path` -- Base data directory

> Mesh/region/viewpoint **selection state** lives on the `task_planning` node
> (`navigation.selected_mesh`, `navigation.selected_region`,
> `navigation.selected_viewpoint`), not here. The GUI and visualizer track those
> parameters; `navigation.selected_mesh` is visualization-only while
> `navigation.selected_region`/`navigation.selected_viewpoint` also scope robot
> execution.

**Publishers:**

- `/planning_scene` (`moveit_msgs/PlanningScene`) -- MoveIt planning scene with mesh collision objects

**Subscribers:**

- `/particle_filter/pose` (`geometry_msgs/PoseStamped`) -- Object pose for planning scene updates

### Other Nodes

- **task_planning_node** -- State machine for robot motion control, manages servo/trajectory controller switching. Parameters are namespaced (`controllers.*`, `navigation.*`, `settings.*`) so the GUI renders one tab per namespace. Owns the mesh/region/viewpoint selection parameters (`navigation.selected_mesh`, `navigation.selected_region`, `navigation.selected_viewpoint`, declared with live slider ranges) and `navigation.selected_traversal_algorithm`; `settings.results_file` points it at the results JSON to plan/execute over; the GUI/visualizer track these to highlight the selected geometry
- **viewpoint_traversal_node** -- MoveIt-based motion planning to viewpoints with TSP/VRP optimization and workspace constraints. Exposes `move_to_pose_stamped`, `optimize_traversal`, and `find_nearest_viewpoint` services. The VRP optimizer minimizes an **execution-time surrogate** by default (`vrp_cost_mode='time'`: per-segment TOTG time from the joint velocity/acceleration limits), or weighted joint-space distance in `joint` mode; with `vrp_validate_topk>1` it plans the top-K candidate tours and keeps the one with the lowest real MoveIt time. The `find_nearest_viewpoint` service returns the viewpoint in a given region that is closest to the robot's actual current joint state under the same cost model, enabling dynamic entry-point selection at execution time rather than relying on precomputed IK-based distances
- **gui_node** -- Open3D visualization GUI with interactive mesh, region, cluster, and viewpoint rendering. Visualization is split into two orthogonal axes:

  - **Region surface (exclusive)** — *View → Region Surface* selects one coloring for the region surfaces: `Solid` (uniform region color) or `Cluster` (colored by owning cluster). Each region is its own exact triangle submesh (segmentation is mesh-native, so no subdivision or nearest-point projection is needed to render it), so selecting a region makes the **others semi-transparent** to focus on it. Selecting a region also shows only its path, shows the enabled overlays for its viewpoints, and auto-selects its first viewpoint.
  - **Viewpoint overlays (inclusive)** — *View → Viewpoint Overlays* independently toggles any combination of per-viewpoint geometries: `Viewpoint Marker`, `FOV Cylinder`, `Origin Line` (surface origin → camera), `Frustum`, and `Origin Marker`. **FOV Cylinder** is a white wireframe cylinder (radius `fov_diameter/2`, height `dof`) centered on each cluster's surface target and aligned to its averaged view direction — the literal coverage volume; overlapping FOVs visibly intersect, honestly depicting the *covering* solution. **Origin Marker** is a small white sphere at the cluster's surface origin, always white rather than tinted by selection or cluster color. **Viewpoint Marker** is colored by the viewpoint's `imaging_mode`: green for `photometric`, yellow for `standard`, red for `inaccessible` (blue overrides all when the viewpoint is selected). This set applies to the non-selected viewpoints.
  - **Selected viewpoint overlays** — *View → Selected Viewpoint Overlays* is a parallel, independent set of the same toggles that applies only to the currently-selected viewpoint (drawn with highlight colors). Because the two sets are independent, you can, e.g., show the FOV cylinder for the selected viewpoint only by enabling it here while leaving it off in *Viewpoint Overlays*.

#### VRP Traversal Optimization Parameters (`viewpoint_traversal_node`)

The VRP optimizer treats the inspection as a **clustered vehicle-routing problem**:
the arm (starting/ending at its home "depot") must visit every viewpoint, region
by region, minimising motion in joint space. `vrp_algorithm` selects the solver;
the mathematical formulation and per-algorithm details are documented in
[`VRP_SYSTEM_REFERENCE.md`](VRP_SYSTEM_REFERENCE.md).

| Parameter | Default | Description |
|---|---|---|
| `vrp_algorithm` | `''` | VRP solver: `vrp_greedy`, `vrp_2opt`, `vrp_3opt`, `vrp_ils`, `vrp_lkh`, `vrp_aco`, `vrp_hierarchical`, `vrp_clustered`. Empty → use TSP instead |
| `vrp_cost_mode` | `time` | `time` = TOTG execution-time surrogate (seconds); `joint` = weighted joint-space distance (radians) |
| `vrp_max_velocity` | 0.5 | Per-joint max velocity (rad/s) for the time model |
| `vrp_max_acceleration` | 1.0 | Per-joint max acceleration (rad/s²) for the time model |
| `vrp_joint_weights` | all 1.0 | Length-7 per-joint weights, **`joint` mode only** |
| `vrp_validate_topk` | 1 | >1: plan the top-K candidate tours with MoveIt, keep the lowest **real** execution time |
| `vrp_aco_n_ants` / `_n_iter` | 20 / 100 | ACO colony size and iteration count |
| `vrp_aco_alpha` / `_beta` / `_rho` | 1.0 / 2.0 / 0.1 | ACO pheromone weight / heuristic weight / evaporation rate |
| `vrp_aco_n_jobs` | 1 | >1: run ACO ants across processes (spawn pool) |
| `vrp_clustered_k` | 6 | Candidate entry/exit ports per region for `vrp_clustered` |
| `vrp_n_turntable_samples` | 0 | **>0 enables the multi-config turntable sweep** — multiple feasible IK configs per viewpoint, activating config-aware `vrp_clustered` and weight-independent reachability |
| `vrp_max_configs_per_vp` | 8 | Cap on candidate IK configs kept per viewpoint (sweep) |
| `vrp_config_dedup_tol` | 0.1 | Joint-space L∞ tolerance (rad) for merging near-identical configs (sweep) |

### Custom Interfaces (viewpoint_generation_interfaces)

**Actions:**
- `ViewpointGeneration.action` -- Trigger viewpoint generation (goal/feedback/result as strings)
- `InspectRegion.action` -- Execute a region inspection task

**Services:**
- `ImportCadModel.srv` -- Load a CAD model (file path + units)
- `MoveToPoseStamped.srv` -- Move robot to a target pose
- `OptimizeViewpointTraversal.srv` -- Optimize traversal order for a results file
- `FindNearestViewpoint.srv` -- Given a region index, return the viewpoint in that region nearest to the robot's current joint state (used for dynamic entry selection at execution time)

**Messages:**
- `OrientationControlData.msg` -- Orientation control feedback (pitch/yaw/roll errors, PID gains)
- `FocusValue.msg` -- Focus metric data
- `AutofocusData.msg` -- Autofocus feedback (images, metrics, ROI, poses)

## Launch

### Viewpoint Generation Only

```bash
ros2 launch viewpoint_generation viewpoint_generation.launch.py \
    object:=my_part \
    data_path:=/path/to/data
```

Arguments:
- `object` -- Config/object name
- `data_path` -- Base data directory
- `headless_mode` -- Run without GUI (default: false)
- `compute_threads` -- Cap on CPU threads/worker processes for the segmentation +
  FOV-clustering compute (default: `6`). The node's PartField/torch BLAS threads and
  the sklearn/loky worker pool otherwise fan out across every core, which saturates the
  CPU (and swaps) and starves the real-time `ros2_control` loop when they share a
  machine. Applied via `OMP_/MKL_/OPENBLAS_/NUMEXPR_NUM_THREADS` and `LOKY_MAX_CPU_COUNT`.
  Keep it well below the core count on a shared box; raise it on a dedicated compute host.

### Full System Bringup

```bash
ros2 launch viewpoint_generation bringup.launch.py \
    cell:=alpha \
    object:=my_part \
    data_path:=/path/to/data \
    sim:=true
```

Additional arguments:
- `cell` -- Inspection cell configuration (`alpha` or `beta`)
- `sim` -- Use fake/simulated hardware
- `headless_mode` -- Run without the Open3D GUI (`gui_node`); starts rqt instead
  (default: false). Forwarded to `viewpoint_generation.launch.py`. Use on hosts with
  no X display.
- `compute_threads` -- Forwarded to `viewpoint_generation.launch.py` (see above).
- `launch_control` -- Launch the `ros2_control` layer locally (default: `true`). Set
  `false` to run hard-real-time control on a **dedicated RT host** (bring it up there
  with `inspection_cell_description/launch/inspection_cell_control.launch.py`, which is
  standalone), while this machine runs only MoveIt + perception. The control launch
  exposes `control_lock_memory` / `control_thread_priority` / `control_cpu_affinity`
  for `SCHED_FIFO` priority, `mlockall`, and core pinning (pin to an `isolcpus` core on
  the RT host).
- `admittance_config_file` -- Admittance/orientation control config (default:
  `admittance_control_coupled_pendulum.yaml`). When `cell` is enabled, bringup includes
  `inspection_control/launch/coupled_pendulum.launch.py`, i.e. the Tier 3 coupled-pendulum
  orientation + admittance pair. To use the legacy point-mass pipeline instead, point this
  back at `admittance_control.yaml` and include `admittance_control.launch.py`.

## JSON Results Format

Results are saved with the naming convention `{N_regions}_regions_{N_clusters}_clusters_{timestamp}.json`:

Each mesh holds a `regions` **list**; each region holds a `clusters` **list**.
The `"0"`, `"1"` keys shown below are list indices. A region's `order` is a
list of cluster indices (identity order) until traversal optimization runs, at
which point it becomes a dict keyed by TSP algorithm name (see below).

```json
{
  "meshes": [
    {
      "file": "/path/to/mesh.stl",
      "units": "m",
      "pose": {
        "position": [0.0, 0.0, 0.0],
        "orientation_rpy": [0.0, 0.0, 0.0],
        "rotation_center": "origin"
      },
      "material": "unknown",
      "dimensions": "(LxWxH): 0.14 x 0.09 x 0.10 m",
      "surface_area": "Surface Area: 0.03 m^2",
      "regions": [
        {
          "faces": [8994, 199, 2379],
          "point_cloud": {
            "file": "/path/to/region_0_1500points.ply",
            "units": "m",
            "points": 1500
          },
          "clusters": [
            {
              "points": [1234, 5678],
              "candidate_axis": [0.0, 0.0, 1.0],
              "viewpoint": {
                "origin": [0.01, 0.02, 0.03],
                "position": [0.1, 0.2, 0.3],
                "direction": [0.0, 0.0, 1.0],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "imaging_mode": "photometric"
              }
            }
          ],
          "blind_spots": {
            "points": [17, 42, 88],
            "reason": "occluded"
          },
          "order": {
            "greedy": {
              "order": [0, 2, 1],
              "distance": 0.84,
              "joint_trajectory": {
                "total_time_s": 12.3,
                "total_joint_distance": 4.56,
                "cartesian_waypoints": [[0.1, 0.2, 0.3]],
                "unreachable": [1]
              }
            },
            "vrp_clustered": {
              "order": [0, 2, 1],
              "distance": 3.9,
              "configs": [2, 0, 1],
              "joint_trajectory": {
                "total_time_s": 10.8,
                "total_joint_distance": 4.10,
                "cartesian_waypoints": [[0.1, 0.2, 0.3]],
                "unreachable": []
              }
            },
            "LKH":    {"order": [0, 1, 2], "distance": 0.79}
          }
        }
      ],
      "order": [0, 1, 2],
      "noise_faces": [100, 200],
      "camera_config": {
        "fov_diameter": 0.03,
        "dof": 0.02,
        "focal_distance": 0.3
      }
    }
  ]
}
```

`region.faces` is a list of triangle indices into the mesh (`meshes[i].file`) —
the exact set of faces region growth or PartField assigned to that region.
`region.point_cloud` is that region's own submesh, Poisson-disk-sampled and
cached to disk during `segment_regions()`; cluster `points` are point indices
local to *that region's own point cloud*, not a shared/global one. `noise_faces`
is a list of triangle indices no region claimed (always empty for PartField,
since every face gets a part).

**`cluster.candidate_axis`** (only produced by `fov_clustering.algorithm ==
'greedy_cover'`; absent for `kmeans`, which never validates occlusion during
clustering) -- the owning candidate's occlusion-validated view axis (unit
vector), i.e. a direction already proven, during `fov_clustering()`, to have
unoccluded line of sight to every point in this cluster. `project_viewpoints()`
reuses it as a guaranteed-good starting point for direction refinement (see
`viewpoint.imaging_mode` below) rather than trusting an unvalidated
mean-of-normals direction.

**`region.blind_spots`** (only produced by `fov_clustering.algorithm ==
'greedy_cover'`; always `{"points": []}` for `kmeans`, which has no coverage
predicate to fail) -- region-local point indices, in the same indexing space
as cluster `points`, that no candidate viewpoint could cover, even after a
Monte Carlo hemisphere search (`rescue_search`) tried every incidence angle
in the outward hemisphere looking for *any* unoccluded viewing direction, not
just the straight surface normal. These points are never assigned to any
cluster, so `blind_spots.points` and the union of a region's
`clusters[*].points` are disjoint. `reason` (present only when `points` is
non-empty) is `"occluded"` when every affected point had at least one
geometrically-valid candidate that occlusion alone ruled out, `"geometric"`
when none did (e.g. a degenerate surface normal), or `"mixed"` when the
region has both. A region can have `blind_spots` with zero `clusters` (the
whole region is occluded). This is a genuine inspection coverage gap, not a
clustering bug, and is also logged as a warning (`region N: <count> points
are unreachable (...)`) when `fov_clustering()` runs. The GUI's *View → Show
Blind Spots* toggle renders these points as markers (see below).

**Viewpoint fields:**
- `origin` -- Cluster centroid on the surface
- `position` -- Camera position (centroid projected along `direction` by `focal_distance`)
- `direction` -- Unit vector, the emitted camera axis. Not simply the mean of
  the cluster's point normals: when a raycasting scene and `fov_normal_threshold`
  are available, `project_viewpoints()` refines it via the same hemisphere
  search used for blind-spot rescue (starting from `candidate_axis` when
  present, so it can never be worse than what clustering already proved),
  since the raw mean direction is never itself guaranteed occlusion-free
- `orientation` -- Camera orientation as quaternion (xyzw)
- `imaging_mode` -- `"photometric"` if `direction` falls within
  `fov_normal_threshold` of the cluster's true mean surface normal (supports
  photometric-stereo normal-map reconstruction), `"standard"` if only an
  off-cone angle within `standard_normal_threshold` has unoccluded line of
  sight (still a usable image, just not photometric-quality), or
  `"inaccessible"` if no direction inside `standard_normal_threshold` is
  occlusion-clear (any remaining angle is too glancing to capture the surface).
  The GUI's `Viewpoint Marker` overlay colors accordingly (green/yellow/red;
  see below)

**`camera_config`** records the FOV geometry the results were generated with
(`fov_diameter`, `dof`, `focal_distance`). Besides documenting the capture
settings, the visualizer reads `fov_diameter`/`dof` from here to draw the **FOV
Cylinder** viewpoint overlay (see above). Results files written before this field
existed simply skip that overlay.

All `regions`/`point_cloud`/`viewpoint` coordinates in this file are expressed
in the mesh's **origin frame** (`model_frame`); the part's physical placement is
never baked in (it is a live TF -- see the `model.pose.*` note under Parameters).
Older results files may carry a now-ignored `pose` field recording a
baked-in `model.pose.*` transform; it is no longer read.

**Traversal order:**
- Mesh-level `order` -- Region visit order (a list of region indices).
- Region-level `order` -- Cluster visit order *within* that region. Out of FOV
  clustering this is a plain list (identity order). After `optimize_traversal`
  runs it becomes a **dict keyed by TSP/VRP algorithm name**, where each value is
  an object `{"order": [...], "distance": ...}` holding that algorithm's optimized
  list of cluster indices plus its path metrics. The key is the selected algorithm's
  name: TSP keys are `greedy`, `2opt`, `3opt`, `ILS`, `LKH`; VRP keys are
  `vrp_greedy`, `vrp_2opt`, `vrp_3opt`, `vrp_ils`, `vrp_lkh`, `vrp_aco`,
  `vrp_hierarchical`, `vrp_clustered` (i.e. the `vrp_algorithm` parameter value).
  `distance` units follow `vrp_cost_mode` — seconds in `time` mode (execution-time
  surrogate) or radians in `joint` mode. Multiple algorithms accumulate in the same
  dict across runs, so the different optimization results — and their metrics — are
  stored together in the results file.
- Region-level `order.<algorithm>.configs` -- **Present for any VRP algorithm when
  the turntable sweep is enabled** (`vrp_n_turntable_samples > 0`), which gives each
  viewpoint multiple candidate IK configurations. A list parallel to `order`:
  `configs[k]` is the index of the chosen IK configuration for cluster `order[k]`,
  selected (via a config-selection DP over the whole tour) so joint-space travel is
  minimised and the arm never has to swing between far IK branches. The trajectory
  planner plans each segment to this exact configuration (joint-space goal) rather
  than re-resolving IK from the Cartesian pose. `vrp_clustered` additionally
  optimises the config *jointly* with region order and entry/exit; the other
  algorithms optimise the order first, then select configs for that fixed tour.
- Region-level `order.<algorithm>.joint_trajectory` -- moveit_py-computed arm
  motion cost for that algorithm's cluster order, added by
  `viewpoint_traversal_node`'s `optimize_traversal` service whenever the
  planning component is available:
  - `total_time_s` / `total_joint_distance` -- Summed trajectory duration
    (seconds) and joint-space path length (radians) across all *successfully
    planned* segments in the region.
  - `cartesian_waypoints` -- Flattened list of `eoat_camera_link` positions
    (metres) along every successfully planned segment, used to draw the arm's
    actual path in the GUI (the "Cartesian Path" toggle).
  - `unreachable` -- Cluster indices (within this region) that a segment could
    not be planned to. The GUI highlights these viewpoints in the 3D view, and
    `optimize_traversal` logs them (`region X:Y: viewpoint N unreachable ...`)
    and reports per-region and whole-inspection totals in its summary output.
- Selected algorithm -- Which algorithm's path the visualizer and motion
  consumers follow when a region's `order` is a dict is **not** stored in the
  results file. It is the `selected_traversal_algorithm` parameter on the
  `task_planning` node (set via the GUI, e.g. by clicking an algorithm under a
  region's *Paths* in the tree view), so the same choice drives both the
  visualized path and the execution order. An empty value falls back to the
  first available algorithm.
