# ViewpointGeneration

A ROS 2 package for automated inspection viewpoint generation from CAD models. Given a surface mesh, the library segments the surface into regions, clusters each region to fit within a camera's field of view, and computes collision-free camera viewpoints for full-surface inspection coverage.

**Maintainer:** Colin Acton (actonc@uw.edu)

## Overview

ViewpointGeneration is structured as a standalone Python library (`viewpoint_generation`) with a ROS 2 node wrapper (`ViewpointGenerationNode`) that exposes the library's pipeline through ROS parameters and services.

The pipeline follows these stages:

1. **Load CAD Model** -- Import an STL/OBJ mesh with unit conversion
2. **Sample Point Cloud** -- Poisson disk sampling of the mesh surface
3. **Estimate Curvature** -- Per-point curvature via KNN covariance eigenvalues
4. **Region Growing** -- Segment the point cloud into contiguous surface regions based on normal similarity and curvature
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
vg.estimate_curvature()
vg.region_growth()
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

**FOVClusteringConfig**

| Parameter | Default | Description |
|---|---|---|
| `fov_diameter` | 0.03 m | Camera field of view diameter at focal distance |
| `dof` | 0.02 m | Depth of field |
| `point_density` | 10.0 | Target points per square millimeter |
| `lambda_weight` | 1.0 | Weight for out-of-FOV penalty in cost function |
| `beta_weight` | 1.0 | Weight for packing efficiency in cost function |
| `point_weight` | 1.0 | Weight for point positions in K-means |
| `normal_weight` | 1.0 | Weight for point normals in K-means |

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

All `RegionGrowingConfig`, `FOVClusteringConfig`, and `ViewpointProjectionConfig` fields are declared as ROS parameters with type information and valid ranges. Additional parameters:

- `model.mesh.file` -- Path to the mesh file
- `model.mesh.units` -- Mesh units (`m`, `cm`, `mm`, `in`)
- `model.point_cloud.file` -- Path to a pre-sampled point cloud
- `model.point_cloud.units` -- Point cloud units
- `model.point_cloud.sampling.number_of_points` -- Sampling density
- `results.file` -- Path to results JSON
- `results.selected_mesh`, `results.selected_region`, `results.selected_cluster` -- Selection state
- `settings.data_path` -- Base data directory

**Publishers:**

- `/planning_scene` (`moveit_msgs/PlanningScene`) -- MoveIt planning scene with mesh collision objects

**Subscribers:**

- `/particle_filter/pose` (`geometry_msgs/PoseStamped`) -- Object pose for planning scene updates

### Other Nodes

- **task_planning_node** -- State machine for robot motion control, manages servo/trajectory controller switching
- **viewpoint_traversal_node** -- MoveIt-based motion planning to viewpoints with TSP optimization and workspace constraints
- **gui_node** -- Open3D visualization GUI with interactive mesh, region, cluster, and viewpoint rendering

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

## JSON Results Format

Results are saved with the naming convention `{N_regions}_regions_{N_clusters}_clusters_{timestamp}.json`:

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
      "regions": {
        "0": {
          "points": [8994, 199, 2379],
          "clusters": {
            "0": {
              "points": [1234, 5678],
              "viewpoint": {
                "origin": [0.01, 0.02, 0.03],
                "position": [0.1, 0.2, 0.3],
                "direction": [0.0, 0.0, 1.0],
                "orientation": [0.0, 0.0, 0.0, 1.0]
              }
            }
          },
          "order": [0, 1, 2]
        }
      },
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
- `order` -- Optimized traversal order for regions and clusters
