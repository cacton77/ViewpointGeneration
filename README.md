# ViewpointGeneration

A ROS 2 package for automated inspection viewpoint generation from CAD models. Given a surface mesh, the library segments the surface into regions, clusters each region to fit within a camera's field of view, and computes collision-free camera viewpoints for full-surface inspection coverage.

**Maintainer:** Colin Acton (actonc@uw.edu)

## Overview

ViewpointGeneration is structured as a standalone Python library (`viewpoint_generation`) with a ROS 2 node wrapper (`ViewpointGenerationNode`) that exposes the library's pipeline through ROS parameters and services.

The pipeline follows these stages:

1. **Load CAD Model** -- Import an STL/OBJ mesh with unit conversion
2. **Sample Point Cloud** -- Poisson disk sampling of the mesh surface
3. **Estimate Curvature** -- Per-point curvature via KNN covariance eigenvalues (region-growth algorithm only)
4. **Surface Segmentation** -- Segment the surface into regions. Two interchangeable algorithms (selected via `regions.segmentation_algorithm`):
   - `region_growth` -- contiguous regions based on normal similarity and curvature
   - `partfield` -- semantic parts from [PartField](https://github.com/nv-tlabs/PartField), mapped from per-face labels onto the sampled point cloud (requires GPU + PartField mounted at `/models/PartField`)
5. **FOV Clustering** -- Subdivide each region into clusters that fit within the camera's field of view and depth of field, using K-means with Bayesian optimization for cluster count
6. **Project Viewpoints** -- Compute a camera pose (position + orientation) for each cluster at the configured focal distance, with ray-cast occlusion checking
7. **Optimize Traversal** -- Solve a TSP to order viewpoints for efficient robot motion

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
vg.set_sampling_number_of_points(10000)
vg.sample_point_cloud()
vg.estimate_curvature()        # skipped automatically when using PartField
# vg.set_segmentation_algorithm("partfield")   # optional: use PartField instead
vg.region_growth()             # runs the selected segmentation algorithm
vg.fov_clustering()
vg.project_viewpoints()

# Save results
vg.save_results("/path/to/results.json")
```

### Configuration

All algorithm parameters are exposed as dataclass configs with sensible defaults:

**RegionGrowingConfig**

| Parameter | Default | Description |
|---|---|---|
| `seed_threshold` | 0.1 | Curvature threshold for seed point selection |
| `region_threshold` | 0.2 | Curvature threshold for region membership |
| `min_cluster_size` | 10 | Minimum points per region |
| `normal_angle_threshold` | pi/3 | Maximum normal deviation (radians) for region membership |
| `curvature_threshold` | 0.1 | Maximum curvature difference for region membership |
| `knn_neighbors` | 30 | K for nearest-neighbor queries |

**PartFieldSegmentationConfig** (used when `segmentation_algorithm == 'partfield'`)

| Parameter | Default | Description |
|---|---|---|
| `num_parts` | 12 | Number of parts to segment the mesh into |
| `use_agglo` | True | Agglomerative clustering (spatially connected parts) instead of KMeans |
| `option` | 0 | Face-adjacency graph for agglomerative clustering: 0=naive, 1=face-MST, 2=cc-MST |
| `with_knn` | False | Augment the face-adjacency graph with kNN connections (agglomerative only) |

PartField runs as a subprocess (`partfield_inference.py` then
`run_part_clustering.py`) against an exported copy of the loaded mesh. Per-face
part labels are mapped to the sampled point cloud by nearest triangle. Mesh
preprocessing/remeshing is left disabled so face ordering is preserved. Note:
PartField produces no noise points (every face is assigned a part).

**FOVClusteringConfig**

| Parameter | Default | Description |
|---|---|---|
| `fov_diameter` | 0.03 m | Camera field of view diameter at focal distance |
| `dof` | 0.02 m | Depth of field |
| `point_density` | 10.0 | Target points per square millimeter (K-means only) |
| `lambda_weight` | 1.0 | Weight for out-of-FOV penalty in cost function (K-means only) |
| `beta_weight` | 1.0 | Weight for packing efficiency in cost function (K-means only) |
| `point_weight` | 1.0 | Weight for point positions in K-means |
| `normal_weight` | 1.0 | Weight for point normals in K-means |
| `algorithm` | `kmeans` | Clustering algorithm: `kmeans` (Bayesian-optimised K-means) or `greedy_cover` (greedy set cover) |
| `fov_normal_threshold` | π/4 rad | Max surface-normal incidence angle for the `greedy_cover` coverage predicate |
| `candidate_spacing` | 0.0 m | Anchor spacing for `greedy_cover` candidate sampling; 0.0 = auto = `fov_diameter/2` |
| `prune_redundant` | `true` | Remove redundant viewpoints after greedy cover (cannot open coverage holes) |
| `rng_seed` | 0 | Random seed for `greedy_cover` candidate sampling (reproducibility) |

**ViewpointProjectionConfig**

| Parameter | Default | Description |
|---|---|---|
| `focal_distance` | 0.3 m | Distance from viewpoint to surface |
| `hemisphere_points` | 10000 | Points used for hemisphere sampling |

## ROS 2 Interface

### ViewpointGenerationNode

Wraps the core library, exposing each pipeline stage as a ROS service and all configuration as ROS parameters.

**Services** (all `std_srvs/srv/Trigger`):

| Service | Description |
|---|---|
| `/viewpoint_generation/sample_point_cloud` | Sample point cloud from loaded mesh |
| `/viewpoint_generation/estimate_curvature` | Compute surface curvature |
| `/viewpoint_generation/region_growth` | Run region growing segmentation |
| `/viewpoint_generation/fov_clustering` | Cluster regions by camera FOV |
| `/viewpoint_generation/viewpoint_projection` | Generate camera viewpoints |
| `/viewpoint_generation/optimize_traversal` | Optimize viewpoint visit order (TSP) |

**Parameters:**

All `RegionGrowingConfig`, `PartFieldSegmentationConfig`, `FOVClusteringConfig`, and `ViewpointProjectionConfig` fields are declared as ROS parameters with type information and valid ranges (prefixed `regions.region_growth.`, `regions.partfield.`, `regions.fov_clustering.`, and `viewpoints.projection.` respectively). Additional parameters:

- `regions.segmentation_algorithm` -- Surface segmentation algorithm: `region_growth` (default) or `partfield`
- `regions.fov_clustering.algorithm` -- FOV clustering algorithm: `kmeans` (default) or `greedy_cover`
- `model.mesh.file` -- Path to the mesh file
- `model.mesh.units` -- Mesh units (`m`, `cm`, `mm`, `in`)
- `model.point_cloud.file` -- Path to a pre-sampled point cloud
- `model.point_cloud.units` -- Point cloud units
- `model.point_cloud.sampling.number_of_points` -- Sampling density
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
- **viewpoint_traversal_node** -- MoveIt-based motion planning to viewpoints with TSP optimization and workspace constraints
- **gui_node** -- Open3D visualization GUI with interactive mesh, region, cluster, and viewpoint rendering. Visualization is split into two orthogonal axes:

  - **Region surface (exclusive)** — *View → Region Surface* selects one coloring for the region surfaces: `Solid` (uniform region color) or `Cluster` (colored by owning cluster). Each region is its own triangle submesh (or a sub-cloud when no mesh is available), so selecting a region makes the **others semi-transparent** to focus on it. Selecting a region also shows only its path, shows the enabled overlays for its viewpoints, and auto-selects its first viewpoint.
  - **Viewpoint overlays (inclusive)** — *View → Viewpoint Overlays* independently toggles any combination of per-viewpoint geometries: `Viewpoint Marker`, `FOV Cylinder`, `Origin Line` (surface origin → camera), `Frustum`, and `Origin Sphere`. **FOV Cylinder** is a white wireframe cylinder (radius `fov_diameter/2`, height `dof`) centered on each cluster's surface target and aligned to its averaged view direction — the literal coverage volume; overlapping FOVs visibly intersect, honestly depicting the *covering* solution. Selecting a viewpoint highlights its overlay geometries (and restores the previously selected one).

### Custom Interfaces (viewpoint_generation_interfaces)

**Actions:**
- `ViewpointGeneration.action` -- Trigger viewpoint generation (goal/feedback/result as strings)
- `InspectRegion.action` -- Execute a region inspection task

**Services:**
- `ImportCadModel.srv` -- Load a CAD model (file path + units)
- `MoveToPoseStamped.srv` -- Move robot to a target pose
- `OptimizeViewpointTraversal.srv` -- Optimize traversal order for a results file

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
      "material": "unknown",
      "dimensions": "(LxWxH): 0.14 x 0.09 x 0.10 m",
      "surface_area": "Surface Area: 0.03 m^2",
      "point_cloud": {
        "file": "/path/to/points.ply",
        "units": "m",
        "num_points": 10000
      },
      "regions": [
        {
          "points": [8994, 199, 2379],
          "clusters": [
            {
              "points": [1234, 5678],
              "viewpoint": {
                "origin": [0.01, 0.02, 0.03],
                "position": [0.1, 0.2, 0.3],
                "direction": [0.0, 0.0, 1.0],
                "orientation": [0.0, 0.0, 0.0, 1.0]
              }
            }
          ],
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
            "LKH":    {"order": [0, 1, 2], "distance": 0.79}
          }
        }
      ],
      "order": [0, 1, 2],
      "noise_points": [100, 200],
      "camera_config": {
        "fov_diameter": 0.03,
        "dof": 0.02,
        "focal_distance": 0.3
      }
    }
  ]
}
```

**Viewpoint fields:**
- `origin` -- Cluster centroid on the surface
- `position` -- Camera position (centroid projected along normal by `focal_distance`)
- `direction` -- Surface normal (unit vector, points outward from surface)
- `orientation` -- Camera orientation as quaternion (xyzw)

**`camera_config`** records the FOV geometry the results were generated with
(`fov_diameter`, `dof`, `focal_distance`). Besides documenting the capture
settings, the visualizer reads `fov_diameter`/`dof` from here to draw the **FOV
Cylinder** viewpoint overlay (see above). Results files written before this field
existed simply skip that overlay.

**Traversal order:**
- Mesh-level `order` -- Region visit order (a list of region indices).
- Region-level `order` -- Cluster visit order *within* that region. Out of FOV
  clustering this is a plain list (identity order). After `optimize_traversal`
  runs it becomes a **dict keyed by TSP algorithm name**, where each value is an
  object `{"order": [...], "distance": ...}` holding that algorithm's optimized
  list of cluster indices plus its path metrics (e.g. `distance` in metres).
  Multiple algorithms accumulate in the same dict across runs, so the different
  optimization results — and their metrics — are stored together in the results
  file rather than in node parameters or a separate file.
- Region-level `order.<algorithm>.joint_trajectory` -- moveit_py-computed arm
  motion cost for that algorithm's cluster order, added by
  `viewpoint_traversal_node`'s `optimize_traversal` service whenever the
  planning component is available:
  - `total_time_s` / `total_joint_distance` -- Summed trajectory duration
    (seconds) and joint-space path length (radians) across all *successfully
    planned* segments in the region.
  - `cartesian_waypoints` -- Flattened list of `eoat_camera_link` positions
    (metres) along every successfully planned segment, used to draw the arm's
    actual path in the GUI (as opposed to the straight-line viewpoint-to-viewpoint
    path).
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
