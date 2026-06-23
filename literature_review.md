# Literature Review: Surface Partitioning, Camera-Based Clustering, and Viewpoint Generation for Robotic Inspection

**Colin — March 2026**

---

## 1. Introduction

This review covers three interconnected problems in automated surface inspection: (1) how to segment a triangle mesh or point cloud into geometrically coherent regions, (2) how to cluster those regions into groups imageable from a single camera pose (FOV clusters), and (3) how to generate and select the minimum number of viewpoints for complete surface coverage. The literature spans computational geometry, computer vision, and robotics path planning. The emphasis is on methods applicable to known-geometry inspection (CAD model or pre-scanned mesh available a priori), with a camera-on-manipulator configuration.

---

## 2. Region-Based Point Cloud / Mesh Partitioning

The goal at this stage is to decompose the object's surface into regions that are "logically coherent" — meaning each region has approximately uniform geometric character (similar normals, curvature class, or primitive type) and can eventually be assigned to a single camera viewpoint with minimal projection distortion.

### 2.1 Classical Geometric Approaches

**Planar segmentation (RANSAC + region growing):** The simplest and most commonly deployed method. RANSAC extracts dominant planes; remaining points are clustered by proximity and normal similarity via region growing. This is what you're currently using. It works well on prismatic/machined parts but struggles with curved, blended, or organic geometry where no clean plane fit exists. Chen et al. (2024) use exactly this as the first stage of their inspection CPP pipeline: RANSAC for initial clustering followed by K-means refinement, which they call "hybrid unsupervised region segmentation." They report significant speedups over purely curvature-based methods on free-form 3C components.

**Curvature-based region growing:** A large family of methods labels each vertex/point by its discrete curvature class (peak, pit, saddle, flat, etc.) using Gaussian and/or mean curvature, then grows connected regions of homogeneous curvature. Besl & Jain (1988) is the foundational work; Srinark & Kambhamettu (2008) define four segment types (peak, pit, minimal-surface, flat) using Gaussian curvature at each vertex. Yamauchi et al. (2005) drive segmentation directly by integrated Gaussian curvature, measuring the developability of each chart — highly relevant to your problem because a region with low integrated Gaussian curvature is approximately developable and thus projects well onto a planar camera sensor. Zhang et al. (2008) use mean-shifted curvature as the clustering feature, then region-grow to collect connected subgraphs; this is more robust to noise than raw curvature thresholding.

**Limitation common to all curvature methods:** Sensitivity to mesh quality and noise. Discrete curvature estimates on noisy scan data are unreliable, and threshold selection is fragile.

**Watershed segmentation:** Mangan & Whitaker (1999) apply the watershed transform using Gaussian curvature as the height field. Produces fine-grained over-segmentation that must be merged. Useful as a preprocessing step but rarely used standalone for inspection.

### 2.2 Variational Shape Approximation (VSA)

Cohen-Steiner, Alliez & Desbrun (2004) introduced Variational Shape Approximation, arguably the most principled framework for the type of partitioning you need. VSA casts mesh partitioning as an optimization problem: partition the triangles into k regions, each approximated by a "shape proxy" (a best-fit plane), and minimize the total approximation error via Lloyd iteration.

Two error metrics are defined:

- **L² metric:** Measures point-to-proxy-plane distance. Good for geometric fidelity but insensitive to orientation.
- **L²,₁ metric (normal metric):** Measures deviation of face normals from the proxy normal. This is the one most relevant to inspection — it directly captures how much the surface deviates from being viewable by a single planar sensor from one direction. The L²,₁ metric captures anisotropy: elongated regions on near-parabolic surfaces, compact regions on near-spherical surfaces.

The Lloyd iteration alternates between (a) assigning each face to its nearest proxy and (b) recomputing proxies as best-fit planes of their assigned faces. Convergence is guaranteed since each step monotonically decreases the energy.

**Extensions:**

- Yan, Liu & Wang (2006) extend VSA to extract general quadric surfaces (spheres, cylinders, cones, etc.) rather than just planes, using an approximate L² metric for efficiency. This is directly useful for machined parts with cylindrical/spherical features.
- Wu & Kobbelt (2005) propose "structure recovery via hybrid variational surface approximation," automatically determining both the number and type of shape proxies.
- Mu et al. (2023) present a **multi-stage clustering** approach for engineering models: VSA produces initial patches, border types between adjacent patches are classified (concave, convex, flat, hole-rim, curvature-similar), and patches are hierarchically aggregated into part-level then surface-level sub-meshes. This is the state of the art for mechanical/engineering models and handles non-zero genus topology (holes, slots) well.

**Why VSA is highly relevant to your problem:** The L²,₁-optimized VSA partition directly produces regions of approximately uniform normal direction, which is exactly the criterion for a single camera viewpoint. The number of proxies k controls the granularity and can be increased until each region fits within a camera FOV.

### 2.3 Developability-Based Partitioning

A surface region is "developable" if it can be unrolled onto a plane without distortion — i.e., it has zero Gaussian curvature everywhere. For camera-based inspection, near-developability of each region is a natural objective: a developable patch projects onto the camera sensor with no geometric distortion.

- Yamauchi et al. (2005) use integrated Gaussian curvature as a measure of chart developability and segment the mesh by growing charts until their integrated curvature exceeds a threshold.
- Julius, Kraevoy & Sheffer (2005) define "D-charts" — quasi-developable mesh segments — by partitioning the mesh into patches with bounded normal deviation, specifically targeting applications like UV unwrapping and papercraft.
- The geodesics-based approach of Issac et al. (2024) directly addresses inspection: they partition curved panels into "near-developable patches" and then tile each patch with camera footprints along geodesic grids. Their stopping criterion is when geodesic lines on the surface diverge or converge beyond a threshold, indicating the surface is no longer near-developable.
- **Evolutionary Piecewise Developable Approximation** (recent ACM TOG work): Uses a genetic algorithm to optimize a combinatorial fitness function over patch count, boundary length, approximation error, and small-patch penalties. The key insight is that mapping distortion between an input surface and its developable approximation is positively correlated with approximation error, so the fitness evaluation can use a cheap distortion proxy instead of fully computing developable patches.

### 2.4 Learning-Based Segmentation

These methods are primarily designed for scene-level semantic segmentation and may require adaptation for your single-object engineering inspection use case, but they represent the cutting edge.

**Point Transformer family:**

- **Point Transformer V3** (Wu et al., 2024, CVPR): Currently state-of-the-art for point cloud segmentation. Uses space-filling curves to serialize point clouds, patch-based attention within non-overlapping patches, and sparse convolution for positional encoding. Dramatically faster and more scalable than earlier versions.
- **Superpoint Transformer** (Robert et al., 2023, ICCV): Partitions the point cloud into a hierarchical superpoint structure and applies self-attention among superpoints. Achieves competitive accuracy with significantly fewer parameters.

**Point-SAM** (ICLR 2025): Extends the Segment Anything Model paradigm to 3D point clouds. A promptable segmentation model with a point cloud encoder, prompt encoder, and mask decoder. Takes point prompts as input and outputs segmentation masks. Relevant because it could allow interactive specification of inspection regions — e.g., "segment this fillet" or "segment all flat faces."

**MSSPTNet** (Miao et al., 2024): Uses dynamic region growing to extract "super-patches" with consistent geometric features, then applies a multi-scale super-patch transformer for segmentation. The initial region-growing step is particularly relevant — it produces clusters of geometrically similar points that could serve as FOV cluster seeds.

**PartUV** (2025): A part-based UV unwrapping pipeline for AI-generated meshes that produces part-aligned charts with low distortion. Although targeting UV mapping, the partitioning step (producing few, semantically meaningful charts) is closely related to your problem.

**Surface variation + HDBSCAN** (Bhargava et al., 2025): A hybrid four-step method for point clouds of mechanical parts: compute surface variation to separate edge/non-edge points, expand edge regions, then cluster remaining points with HDBSCAN. Validated on synthetic and real mechanical parts with noise and density variation. Provides an interactive GUI for real-time parameter adjustment — a practical advantage for an inspection workflow.

### 2.5 Feature-Based Mesh Segmentation for Engineering Parts

Ma et al. (2025) propose a method specifically targeting machined components: they combine feature boundary detection with region segmentation using point cloud feature recognition. This addresses the specific challenge of manufactured parts where feature boundaries (edges between machined surfaces) are semantically meaningful and should coincide with region boundaries.

### 2.6 Recommendations for Your Pipeline

Your current planar segmentation is a reasonable baseline for prismatic parts. For complex geometry, the most promising upgrade paths in increasing order of complexity are:

1. **VSA with L²,₁ metric** — drop-in replacement that produces normal-coherent regions, handles curved surfaces, and gives you direct control over region count. Open-source implementations exist (CGAL).
2. **Curvature-class region growing** — as a preprocessing step to seed VSA, giving it boundary hints at sharp features.
3. **Multi-stage clustering à la Mu et al. (2023)** — for parts with holes, slots, fillets, and chamfers, adds border-type classification and hierarchical merging.
4. **Developability-driven partitioning** — directly optimizes what you care about (regions that project well onto a planar sensor).

---

## 3. Camera-Based FOV Clustering

Once the surface is partitioned into geometric regions, the next step is to group those regions (or individual surface elements) into sets that can be imaged from a single camera viewpoint. This is the "FOV clustering" step.

### 3.1 The Set Cover Formulation

The problem is fundamentally a **Set Covering Problem (SCP)**: given a universe U of surface elements (triangles, patches, or points) and a collection S of subsets (each subset being the set of surface elements visible and adequately resolved from a candidate viewpoint), find the minimum-cardinality sub-collection of S that covers all of U.

SCP is NP-hard. The **greedy algorithm** (repeatedly select the viewpoint that covers the most uncovered elements) provides an O(ln n)-approximation, which is the best achievable in polynomial time unless P=NP (Chvátal 1979, Wolsey 1982). This greedy approach is the workhorse of most practical inspection planning systems.

**Submodular formulation:** Surface coverage is a monotone submodular function — adding a viewpoint to a set never decreases coverage, and the marginal gain of adding a viewpoint diminishes as coverage increases. This means the greedy algorithm's approximation bound of (1 - 1/e) for maximum coverage also applies (Nemhauser, Wolsey & Fisher, 1978), and more sophisticated methods like lazy evaluation (CELF) can accelerate the greedy selection.

### 3.2 Visibility and Imaging Quality Constraints

A surface element is "adequately imageable" from a viewpoint only if multiple constraints are satisfied simultaneously:

- **Visibility:** No occlusion by other parts of the object or the environment.
- **Resolution (GSD):** The ground sampling distance at the surface must be below the required defect detection threshold. This constrains the camera-to-surface distance.
- **Incidence angle:** The angle between the camera optical axis and the surface normal must be below a threshold (typically 30–45°). Grazing-angle views produce foreshortening that degrades defect visibility.
- **Focus / depth of field:** All points within the FOV must be within the depth of field at the working distance.
- **Field of view:** The surface element must project inside the camera's sensor bounds at the candidate pose.

Tarabanis, Allen & Tsai (1995) formalized this as the "MVP" (Machine Vision Planning) problem and developed constraint-based viewpoint generation that satisfies all requirements simultaneously.

Schönberger & Frahm (2016) of the "Feature-Driven Viewpoint Placement" line define functionals on surface patches to evaluate viewpoint quality: integral-based criteria (measuring total normal deviation from the viewing direction), maximum-deviation criteria, and camera-ray-deviation criteria. They use recursive binary subdivision of the surface — splitting the worst-scoring patch along its longest axis — until all patches satisfy the imaging quality threshold. This recursive splitting is one of the cleanest approaches to FOV clustering in the literature.

### 3.3 Normal-Sensitive Clustering

Mu et al. (2024, IEEE TIP) cluster voxelized 3D scenes into K regions using **k-means in a 6D normal-sensitive space** (3D position + 3D surface normal). This joint position-normal clustering directly produces regions with coherent viewing direction, making each cluster imageable from approximately one direction. The clustering is independent of any downstream training and takes ~42–57 seconds per scene. They then treat virtual view generation as an instance of the SCP.

This is conceptually very close to what you need: cluster surface elements by (position, normal) → each cluster maps to a viewing direction → generate a viewpoint per cluster.

### 3.4 Spectral Clustering for Viewpoint Grouping

Jing & Shimada (2017) use **spectral clustering** to group surface primitives into viewpoint-compatible sets, followed by local potential field methods and hyper-heuristic algorithms for viewpoint optimization. Spectral clustering captures non-convex region shapes better than k-means and can handle the complex boundaries that arise on real parts.

### 3.5 Region-of-Interest (ROI) Based Methods

Chen et al. (2024) define an **adaptive ROI method** for line-scanner inspection: after hybrid region segmentation (RANSAC + K-means), each region is assigned an ROI that defines the local scanning path. The ROI dimensions are adapted to the region's geometry to ensure the scanner stays within its effective measurement range. For area cameras (your case), the analogous concept is to define each FOV cluster as the maximal connected region whose normal cone fits within the camera's acceptable incidence angle range.

### 3.6 Graph-Based and Hierarchical Approaches

Marshall & Fisher (1999) describe the problem as partitioning the object's face adjacency graph: faces are grouped into sets viewable from a common viewpoint by constructing a "generalized cone" for each face (bounding the directions of unobstructed view) and then finding overlapping cone intersections. The partition is found by computing all possible groupings and applying a heuristic — in practice, greedy merging of faces with compatible visibility cones.

For complex parts, hierarchical approaches are more tractable: start with a fine-grained partitioning (e.g., VSA with many proxies), then merge adjacent regions whose combined proxy still falls within a single camera FOV.

### 3.7 Recommendations for Your Pipeline

The most practical approach for your system (UR5e + macro camera with known intrinsics):

1. **Partition the mesh using VSA (L²,₁ metric)** with a conservatively large k.
2. **Compute the normal cone for each region** — the bounding cone of all face normals.
3. **Greedily merge adjacent regions** whose combined normal cone apex angle is less than the camera's acceptable incidence angle (function of DOF, working distance, required GSD).
4. **Validate each merged region against FOV constraints** — project it onto the camera sensor at the candidate pose; if it exceeds the sensor bounds, split.
5. **The resulting merged regions are your FOV clusters.**

Alternatively, directly cluster face centroids in (position, normal) space using k-means or spectral clustering, then validate against camera constraints. The k in k-means can be initialized from a lower bound (total surface area / single-view coverage area) and increased until all clusters satisfy the FOV and quality constraints.

---

## 4. Viewpoint Generation

Given a set of FOV clusters, the final step is to compute a camera pose (position + orientation) for each cluster that satisfies all imaging constraints.

### 4.1 Geometric Viewpoint Generation

**Normal-based placement:** The simplest approach: for each cluster, compute the centroid and average normal, then place the camera along the average normal at the required working distance. This is the most common method in practice and works well for approximately planar regions.

**Geodesic grid methods:** Issac et al. (2024) generate viewpoints by propagating geodesic lines on the surface mesh to create a grid of camera footprints. Starting from a seed point, they extend geodesic lines in the length and width directions of the camera sensor projection. When the surface deviates too much from developable (geodesics diverge/converge beyond threshold), a new seed is selected and coverage continues. Advantages: naturally handles curved surfaces; geodesics approximate straight lines on near-planar regions, so the grid is near-rectangular there.

**Feature-driven placement:** Schönberger et al. (2020, Machine Vision and Applications) construct B-spline surface approximations of each surface region, then recursively subdivide regions until viewpoint quality criteria (normal deviation, resolution) are satisfied. Each resulting sub-region generates one viewpoint. Quality criteria include integral-based functionals and maximum-angle functionals. They observe that integral-based criteria show monotone decrease during subdivision (stable behavior), while angle-based criteria can exhibit jumps requiring finer subdivision on wiggly surfaces.

### 4.2 Sampling-Based Methods

**Random and quasi-random sampling on a viewing sphere/shell:** Generate candidate viewpoints uniformly on a sphere (or shell, respecting min/max working distance) around the object, evaluate each for coverage and quality, then select the minimum covering set via greedy SCP. Almadhoun et al. (2016) survey this family extensively.

**Targeted viewpoint sampling (Glorieux, Franciosa & Ceglarek, 2020, Robotics & CIM):** A key paper for your application. They formulate CPP for robotic dimensional inspection of free-form sheet metal surfaces. The critical insight is that **non-random, targeted viewpoint sampling significantly outperforms random sampling** — by optimizing viewpoint positions during the sampling phase (not just during the selection phase), they achieve up to 23.8% reduction in inspection cycle time compared to random sampling with the same selection algorithm. Their optimization considers the number of surface primitives visible from each candidate viewpoint and the measurement quality at each primitive.

**Viewpoint resampling (Bircher et al., 2015, 2016):** The Structural Inspection Planner (SIP) uses an alternating two-step optimization: (1) optimize viewpoint positions to reduce path cost while maintaining coverage, (2) optimize the tour (TSP) connecting them. This iterates until convergence. Open-source implementation available. Originally for UAV inspection but applicable to manipulator-based systems.

### 4.3 Optimization-Based Methods

**Genetic algorithms:** The two-stage approach of the UAS inspection paper (2024) uses a genetic algorithm to determine viewpoint positions, then a greedy algorithm for camera orientations. The sensitivity analysis shows this outperforms direct 5-DOF optimization by at least 30% in path length.

**PSO-based methods:** Chen et al. (2024) use Particle Swarm Optimization for global path optimization after local scanning paths are defined per region. The PSO minimizes total inspection time by optimizing the sequence of viewpoints.

**Monte Carlo Tree Search (MCTS):** Wang et al. (2025, Journal of Field Robotics) propose MCTS-based viewpoint selection with an integrated viewpoint evaluation framework. MCTS effectively explores the viewpoint search space and avoids local optima that trap greedy methods. Applied to UAV surface inspection but the algorithm is platform-agnostic.

### 4.4 Learning-Based Viewpoint Planning

**Deep reinforcement learning (DRL):**

- Xiao et al. (2024) apply DRL to optimize next-best-view planning for turbine blade reconstruction, improving efficiency in industrial inspection.
- Wang et al. (2024) use RL-based NBV selection for unknown object reconstruction, learning a policy that predicts the most informative next viewpoint.
- The digital-twin photogrammetry work uses deep RL to optimize camera positions in a reconfigurable manufacturing environment, learning to adapt viewpoint layouts to dynamic production line configurations.

**Neural Visibility Fields (NeVF):** Xue et al. (2024) model uncertainty through a neural visibility field that estimates which parts of a scene remain uncertain. Viewpoints are selected to maximize expected information gain. This is an uncertainty-driven approach and is most relevant to iterative/online inspection where the object model is being refined during inspection.

**Gaussian Splatting for active view selection:** FisherRF (Jiang, Lei & Daniilidis, 2024) uses Fisher information computed from a 3DGS representation to quantify observational information at each candidate viewpoint, selecting views with maximal information gain. ActiveSplat (2025, RAL) extends this with a full active mapping framework: hybrid map representation, decoupled viewpoint selection for translation and rotation, and topology-based hierarchical path planning. While targeting scene reconstruction rather than inspection, the active view selection methodology transfers directly.

### 4.5 Coverage Path Planning (CPP) Integration

Viewpoint generation is one sub-problem of the full CPP pipeline. The two main paradigms are:

**Two-stage (AGP + TSP):** First solve an Art Gallery Problem variant to find the minimum viewpoint set, then solve a Traveling Salesman Problem variant to find the shortest path visiting all viewpoints. Most existing work follows this paradigm. Limitation: the two stages can produce globally suboptimal solutions because viewpoint selection doesn't consider path cost.

**Joint optimization:** Bircher et al.'s SIP and related methods jointly optimize viewpoint selection and path planning via alternating optimization. Wang et al.'s MCTS approach also inherently considers both coverage and path cost.

### 4.6 The NBV Taxonomy (Alsadik et al., 2025)

Alsadik et al. (2025, ISPRS Open J. Photogrammetry and Remote Sensing) provide the most comprehensive recent review and taxonomy of next-best-view strategies, categorizing methods as:

- **Rule-based:** Fixed angular steps, visual-region-based placement, heuristic criteria. Fast but often suboptimal.
- **Uncertainty-based:** Maximize expected information gain, minimize reconstruction uncertainty. Best for online/iterative settings.
- **Sampling-based:** Probabilistic viewpoint candidate generation with coverage evaluation. Flexible but computationally expensive.
- **Learning-based:** DRL, neural fields, learned policies. Most cutting-edge but require training data or simulation environments.

For your offline inspection planning with a known mesh, the uncertainty-based and learning-based methods are less relevant than the geometric and sampling-based approaches, since you have complete geometry a priori.

### 4.7 Recommendations for Your Pipeline

For a UR5e with a macro camera and known part geometry:

1. **Generate candidate viewpoints geometrically:** For each FOV cluster, compute a candidate pose along the cluster's average normal at the working distance. Validate against all imaging constraints.
2. **Refine via targeted sampling:** If the initial candidate doesn't satisfy all constraints (resolution, incidence angle, DOF coverage), sample additional candidates in a neighborhood around the initial pose and select the best.
3. **Solve the covering problem greedily:** If viewpoints have overlapping coverage (intentional for inspection reliability), use a greedy set cover to find the minimum set providing complete coverage.
4. **Optimize the path:** Once viewpoints are fixed, solve a TSP variant (or use MoveIt2 motion planning directly) to find the shortest collision-free robot path visiting all viewpoints.

---

## 5. Cross-Cutting Themes and Open Problems

### 5.1 The Partitioning-Viewpoint Feedback Loop

Most pipelines treat segmentation and viewpoint generation as sequential stages. But the optimal partitioning depends on the camera model (FOV, DOF, resolution), and the optimal viewpoints depend on the partitioning. **Joint optimization** of partitioning and viewpoint placement remains an open problem. The closest existing work is the recursive subdivision approach of the feature-driven viewpoint placement paper, where subdivision is driven by camera-quality functionals.

### 5.2 Macro-Scale Inspection Challenges

Your macro imaging setup (photometric stereo with NeoPixel ring, focus stacking) introduces additional constraints not typically addressed in the inspection CPP literature: working distance is very short (~30–50mm typical for macro), DOF is extremely shallow, and the illumination geometry (LED positions relative to surface) matters. These constraints significantly reduce the effective FOV per viewpoint and increase the minimum number of viewpoints needed.

### 5.3 Multi-Channel Imaging

Your 7-channel imaging pipeline (RGB + normals + albedo from photometric stereo) is unique. No existing viewpoint planning framework accounts for photometric stereo illumination requirements (multiple illumination directions per viewpoint, surface orientation relative to light sources). This is an opportunity for novel contribution.

### 5.4 Computational Complexity

For a mesh with n triangles and k candidate viewpoints, the greedy SCP runs in O(nk) per iteration with at most k iterations, giving O(nk²) total. For practical inspection (n ~ 10⁴–10⁶, k ~ 10²–10³), this is tractable. The bottleneck is usually the visibility computation (occlusion checking via ray casting), which can be accelerated with BVH or GPU-based methods.

---

## 6. Key References

### Region Partitioning
- Cohen-Steiner, Alliez & Desbrun. *Variational Shape Approximation.* ACM TOG (SIGGRAPH), 2004.
- Yan, Liu & Wang. *Quadric Surface Extraction by Variational Shape Approximation.* GMP, 2006.
- Mu, Liu, Duan, Tan. *Part-to-Surface Mesh Segmentation for Mechanical Models Based on Multi-Stage Clustering.* Computer-Aided Design, 2023.
- Yamauchi, Gumhold, Zayer & Seidel. *Mesh Segmentation Driven by Gaussian Curvature.* The Visual Computer, 2005.
- Wu & Kobbelt. *Structure Recovery via Hybrid Variational Surface Approximation.* Eurographics, 2005.
- Bhargava et al. *A Comprehensive and Hybrid Approach to Automatic and Interactive Point Cloud Segmentation Using Surface Variation Analysis and HDBSCAN Clustering.* 2025.
- Ma et al. *A Novel Point Cloud Segmentation Method for Accurate Surface Partitioning Based on Feature Boundaries.* ScienceDirect, 2025.
- Robert, Raguet & Landrieu. *Efficient 3D Semantic Segmentation with Superpoint Transformer.* ICCV, 2023.
- Wu et al. *Point Transformer V3: Simpler Faster Stronger.* CVPR, 2024.
- Point-SAM. *Promptable 3D Segmentation.* ICLR, 2025.
- ACM TOG. *Evolutionary Piecewise Developable Approximations.* 2023.
- Julius, Kraevoy & Sheffer. *D-Charts: Quasi-Developable Mesh Segmentation.* CGF, 2005.

### FOV Clustering & Camera-Based Grouping
- Tarabanis, Allen & Tsai. *A Survey of Sensor Planning in Computer Vision.* IEEE T-RA, 1995.
- Marshall & Fisher. *Viewpoint Planning for Complete 3D Object Surface Coverage.* University of Edinburgh.
- Mu et al. *Learning Virtual View Selection for 3D Scene Semantic Segmentation.* IEEE TIP, 2024.
- Jing & Shimada. *Spectral Clustering-based Viewpoint Planning.* J. Comput. Des. Eng., 2017.
- Wolsey. *An Analysis of the Greedy Algorithm for the Submodular Set Covering Problem.* Combinatorica, 1982.
- Nemhauser, Wolsey & Fisher. *An Analysis of Approximations for Maximizing Submodular Set Functions.* Math. Programming, 1978.

### Viewpoint Generation & Coverage Path Planning
- Glorieux, Franciosa & Ceglarek. *Coverage Path Planning with Targeted Viewpoint Sampling for Robotic Free-Form Surface Inspection.* Robotics & CIM, 2020.
- Bircher, Alexis et al. *Three-Dimensional Coverage Path Planning via Viewpoint Resampling and Tour Optimization for Aerial Robots.* Autonomous Robots, 2016.
- Bircher, Alexis et al. *Structural Inspection Path Planning via Iterative Viewpoint Resampling.* ICRA, 2015.
- Chen et al. *PSO-Based Optimal Coverage Path Planning for Surface Defect Inspection of 3C Components with a Robotic Line Scanner.* 2024.
- Wang et al. *A Path Planning Algorithm for UAV 3D Surface Inspection Based on Normal Vector Filtering and Integrated Viewpoint Evaluation.* J. Field Robotics, 2025.
- Issac et al. *Viewpoint Generation Using Geodesics and Associated Semi-Automated Coverage Path Planning of Panels for Inspection.* Applied Sciences, 2024.
- Schönberger et al. *Feature-Driven Viewpoint Placement for Model-Based Surface Inspection.* Machine Vision and Applications, 2020.
- Alsadik et al. *A Structured Review and Taxonomy of Next-Best-View Strategies for 3D Reconstruction.* ISPRS Open J. Photogrammetry, 2025.
- Optimized Structural Inspection Path Planning for Automated UAS. *Automation in Construction*, 2024.
- Liu et al. *Coverage Path Planning for Robotic Quality Inspection with Control on Measurement Uncertainty.* 2024.
- Jiang, Lei & Daniilidis. *FisherRF: Active View Selection and Mapping with Radiance Fields.* 2024.
- ActiveSplat. *Gaussian Splatting-Based Active Mapping.* IEEE RAL, 2025.
- Xue et al. *Neural Visibility Field for Uncertainty-Based Active Mapping.* 2024.
- Liu et al. *Multi-Robot Coverage Path Planning for Dimensional Inspection of Large Free-Form Surfaces Based on Hierarchical Optimization.* IJAMT, 2023.
