# Implementation Brief: Greedy Set-Cover FOV Clustering

**Target repo:** `cacton77/ViewpointGeneration`
**Primary file to change:** `viewpoint_generation/viewpoint_generation/fov_clustering.py`
**Related files to read (do not assume signatures — verify in code):**
`viewpoint_generation/viewpoint_generation.py` (pipeline + config dataclasses),
`nodes/viewpoint_generation_node.py` (ROS parameter declarations),
`README.md` (the `FOVClusteringConfig` table and the `JSON Results Format` section).

---

## 0. Read before writing

This brief is self-contained on *intent and algorithm*, but the exact class
names, method signatures, and data structures must be taken from the live code.
Before implementing:

1. Read `fov_clustering.py` end to end. Identify the existing entry point that
   subdivides one region into clusters (currently K-means whose cluster count
   `k` is chosen by Bayesian optimization). Note its inputs (point cloud array,
   normals array, region point indices, config) and **exactly** what it returns
   or writes (the per-region list of clusters, where each cluster is a set of
   point indices into the global cloud).
2. Read how `FOVClusteringConfig` is defined and how its fields are declared as
   ROS parameters (README shows the prefix `regions.fov_clustering.`).
3. Read the `JSON Results Format` section of the README so you understand the
   downstream contract: each region has a `clusters` list; each cluster has a
   `points` index list and (later, from `project_viewpoints`) a `viewpoint`.

**The output structure must not change.** Downstream consumers
(`viewpoint_projection.py`, the visualizer, traversal/TSP, and the results JSON)
must keep working untouched. Your new code produces the same per-region list of
disjoint clusters that the K-means path produces; only *how* clusters are
chosen changes.

---

## 1. Why we are changing this

The current FOV clustering solves a **partitioning** problem (K-means assigns
every point to exactly one cluster) and wraps it in Bayesian optimization to
discover the cluster count `k`. But inspection-viewpoint generation is a
**covering** problem: a surface point may fall inside several camera FOVs, and
it only needs to fall inside *at least one acceptable* FOV. Using a partition
algorithm here forces a structure the problem doesn't require, and turns `k`
(which is really a derived quantity of part geometry + FOV size) into a
hyperparameter that must be searched. The BO-over-`k` loop runs K-means to
convergence many times per region — expensive and ad hoc.

Replace it with **greedy forward set cover**: directly minimize the number of
viewpoints subject to a hard full-coverage constraint. This is simpler, cheaper
(no nested iterative optimization), and standard.

**Critical design correction:** do **not** minimize FOV overlap. Minimal
overlap and full coverage are in tension — driving overlap to zero on a curved
surface opens coverage holes between footprints. The objective is *minimize
viewpoint count subject to full coverage*; overlap is tolerated, then reduced as
a final prune step (Section 5.4), never used as the optimization target.

---

## 2. Integration strategy (low risk)

Mirror the existing `regions.segmentation_algorithm` selector pattern
(`region_growth` | `partfield`). Add a clustering-algorithm selector so both
methods coexist and nothing breaks:

- Add ROS parameter / config field `regions.fov_clustering.algorithm` with
  values `kmeans` (existing behavior) and `greedy_cover` (this new method).
- Keep the existing K-means path intact and reachable.
- Default may stay `kmeans` for safety, or switch to `greedy_cover` — confirm
  the maintainer's preference; if unspecified, leave default `kmeans` and make
  `greedy_cover` opt-in.

The K-means-specific weights (`lambda_weight`, `beta_weight`, `point_weight`,
`normal_weight`) are unused by `greedy_cover`. Leave them declared (so existing
launch configs don't break); they simply have no effect when
`algorithm == greedy_cover`.

---

## 3. New configuration fields

Add these to `FOVClusteringConfig` and declare them as ROS parameters under
`regions.fov_clustering.` with sensible ranges, following the existing
declaration style:

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `algorithm` | str enum | `kmeans` | `kmeans` or `greedy_cover` |
| `fov_normal_threshold` | float (rad) | `pi/4` | Max incidence angle: a point is only "covered" if its surface normal is within this cone of the candidate view axis. **This is the photometric-stereo acceptability constraint** and replaces the role `normal_weight` played in K-means. The default is a starting point — it must be tuned to the actual PS imaging geometry. |
| `candidate_spacing` | float (m) | `None` → auto = `fov_diameter / 2` | Surface spacing of candidate viewpoint anchors (farthest-point / Poisson-disk sampling). Tighter spacing = more candidates = better coverage but slower. |
| `prune_redundant` | bool | `True` | Enable the redundant-viewpoint prune tail step. |
| `rng_seed` | int | `0` | Seed for candidate sampling so a generated inspection plan is reproducible and re-executable. |

Existing fields used by this algorithm: `fov_diameter` (→ `fov_radius =
fov_diameter / 2`) and `dof`. Note: `focal_distance` lives in
`ViewpointProjectionConfig`, **not** here — and this algorithm does **not** need
it (see Section 4, the DoF test is expressed in the anchor's local frame).

---

## 4. The coverage predicate (must be normal-aware)

For a candidate anchored at surface point `a` with outward unit normal `n`
(camera sits at `a + focal_distance·n` looking back along `−n`; the focal plane
passes through `a`), a region point `p` with outward unit normal `np` is
**covered** iff all three hold. Work in the anchor's local frame; let
`d = p − a`:

1. **Depth of field (axial):** `|d · n| ≤ dof / 2`
   (signed distance of `p` from the focal plane, along the view axis).
2. **Field of view (lateral):** `‖ d − (d · n) n ‖ ≤ fov_radius`
   (perpendicular distance from the view axis).
3. **Photometric incidence:** `angle(np, n) ≤ fov_normal_threshold`,
   i.e. `np · n ≥ cos(fov_normal_threshold)`. (Outward normals: a well-imaged
   point's normal faces the camera, so it should be near-parallel to `n`.)

Because the DoF and FOV tests are relative to the anchor and its normal,
`focal_distance` is not required here.

**Occlusion / line-of-sight is intentionally excluded** from this predicate.
It stays where it already lives — the ray-cast in `viewpoint_projection.py`.
Folding ray-casting into candidate scoring would destroy the performance win.
Do not add it here.

---

## 5. Algorithm (per region)

Run independently **per region**, never globally, so a footprint cannot straddle
two semantic parts (preserves the part-awareness the rest of the system relies
on). A little extra redundancy at region boundaries is an acceptable trade.

### 5.1 Candidate generation
Sample anchor points across the region surface using farthest-point or
Poisson-disk sampling at `candidate_spacing` (auto = `fov_radius`), seeded by
`rng_seed`. Each candidate = `(center = anchor_pt, axis = anchor_normal)`.
Spaced sampling pre-tiles the surface, so far fewer candidates are needed than
uniform random, and runs are reproducible.

### 5.2 Precompute coverage sets
For each candidate, find its covered region points using a KD-tree radius query
(reuse the existing cloud KD-tree if the codebase already builds one for
curvature / region growth — do not rebuild per region) followed by the
predicate of Section 4. Store `coverage[c]` = set of region-local point ids.

### 5.3 Coverability guarantee
Compute the union of all coverage sets. Any region point not in the union is
intrinsically uncovered (typically a high-curvature spot no sampled candidate
satisfies). For each such point, append a **self-anchored** candidate at that
point — a footprint centered on a point trivially covers it (zero lateral
offset, zero axial offset, matching normal). Recompute that candidate's coverage
set. This prevents the greedy loop from stalling.

### 5.4 Greedy forward set cover
Initialize `uncovered` = all region points except designated noise points
(region-growth produces noise; PartField produces none — pull the noise set from
whatever the region carries). Repeatedly pick the candidate covering the most
still-uncovered points, append it to `chosen`, and subtract its coverage from
`uncovered`. Stop when `uncovered` is empty. Stopping rule is *coverage
complete* — yielding a (1 + ln n) approximation to minimum viewpoint count.

### 5.5 Redundancy prune (this is where "downsample overlap" belongs — AFTER coverage)
If `prune_redundant`: iterate `chosen` (try dropping later picks first); drop any
selected candidate whose coverage set is a subset of the union of the other
selected candidates' coverage sets. This reduces overlap *after* coverage is
guaranteed, so it cannot open holes.

### 5.6 Resolve overlap into DISJOINT clusters (preserve output contract)
Selection allows overlap, but emitted clusters must be disjoint to match the
existing data model. Assign each covered (non-noise) region point to exactly one
chosen candidate: the one with the smallest normal angle to the point, tie-
broken by smallest `|d · n|` (closeness to the focal plane). The result is a
partition of the region's covered points across the chosen viewpoints —
structurally identical to the K-means output.

### 5.7 Emit
Emit clusters as lists of **global** cloud indices (map region-local ids back
through the region's point-index array), in the same container type the K-means
path returns. Each chosen candidate with at least one assigned point becomes one
cluster.

### 5.8 Orientation note (verify downstream)
Each cluster's view orientation should be derived from the **average normal of
its assigned points**, not a single anchor normal (averaging recovers the
smoothing K-means did implicitly). `project_viewpoints` likely already computes
direction from the cluster centroid/mean normal — verify it does, and if it uses
a single point's normal, switch it to the mean over the cluster's points.

---

## 6. Pseudo-code

```python
def fov_cluster_region(region_point_idx, cloud_pts, cloud_normals, cfg,
                       kdtree, noise_global_ids):
    """Greedy set-cover replacement for K-means+BO on ONE region.
    Returns the same container the kmeans path returns: a list of clusters,
    each a list of GLOBAL cloud indices, disjoint."""
    region_set   = set(region_point_idx)
    g2l          = {g: i for i, g in enumerate(region_point_idx)}   # global->local
    pts          = cloud_pts[region_point_idx]        # (m,3)
    normals      = cloud_normals[region_point_idx]    # (m,3)
    noise_local  = {g2l[g] for g in noise_global_ids if g in g2l}
    fov_radius   = cfg.fov_diameter / 2.0
    spacing      = cfg.candidate_spacing or fov_radius
    cos_thr      = cos(cfg.fov_normal_threshold)
    rng          = seeded_rng(cfg.rng_seed)

    def covers(cand):
        """Local ids covered by cand. cand = (center, axis(unit))."""
        out = set()
        for g in kdtree.query_radius(cand.center, r=fov_radius):
            if g not in region_set:
                continue
            p = g2l[g]
            d = pts[p] - cand.center
            axial = dot(d, cand.axis)
            if abs(axial) > cfg.dof / 2.0:
                continue
            lateral = norm(d - axial * cand.axis)
            if lateral > fov_radius:
                continue
            if dot(normals[p], cand.axis) < cos_thr:     # incidence cone
                continue
            out.add(p)
        return out

    # 5.1 candidate anchors via spaced surface sampling
    anchors   = farthest_point_sample(pts, spacing=spacing, rng=rng)   # local ids
    candidates = [Candidate(center=pts[a], axis=unit(normals[a])) for a in anchors]
    coverage   = [covers(c) for c in candidates]

    # 5.3 coverability guarantee: self-anchor any uncovered point
    covered_union = set().union(*coverage) if coverage else set()
    for p in range(len(pts)):
        if p not in covered_union and p not in noise_local:
            c = Candidate(center=pts[p], axis=unit(normals[p]))
            candidates.append(c); coverage.append(covers(c))

    # 5.4 greedy forward set cover
    uncovered = set(range(len(pts))) - noise_local
    chosen = []
    while uncovered:
        best = max(range(len(candidates)),
                   key=lambda c: len(coverage[c] & uncovered))
        gain = coverage[best] & uncovered
        if not gain:
            break                      # safety: remaining points uncoverable
        chosen.append(best)
        uncovered -= gain

    # 5.5 redundancy prune (overlap reduction AFTER coverage)
    if cfg.prune_redundant:
        for c in list(reversed(chosen)):
            others = set().union(*(coverage[o] for o in chosen if o != c)) \
                     if len(chosen) > 1 else set()
            if coverage[c] <= others:
                chosen.remove(c)

    # 5.6 disjoint assignment for the emitted clusters
    assignment = {c: [] for c in chosen}
    assignable = (set(range(len(pts))) - noise_local) - uncovered
    for p in assignable:
        owner = min((c for c in chosen if p in coverage[c]),
                    key=lambda c: (acos_clamp(dot(normals[p], candidates[c].axis)),
                                   abs(dot(pts[p] - candidates[c].center,
                                           candidates[c].axis))))
        assignment[owner].append(p)

    # 5.7 emit as GLOBAL indices, dropping empty clusters
    clusters = [[region_point_idx[p] for p in members]
                for members in assignment.values() if members]
    return clusters
```

`Candidate` can be a tiny dataclass/namedtuple `(center: vec3, axis: vec3)`.
`unit`, `norm`, `dot`, `acos_clamp` (arccos with input clamped to [-1, 1]) are
trivial numpy helpers — prefer vectorizing `covers()` over the radius-query
result rather than the Python loop shown, which is written for clarity.

---

## 7. Acceptance criteria

- **Full coverage:** after running, every non-noise region point satisfies the
  Section 4 predicate for at least one emitted cluster's anchor. Add an
  assertion / test that checks this on a sample part.
- **Disjoint partition:** emitted clusters share no points; their union equals
  the region's covered non-noise points.
- **Determinism:** two runs with the same `rng_seed` and inputs produce
  identical clusters.
- **No downstream changes required:** `project_viewpoints`, the visualizer,
  traversal/TSP, and the results JSON run unchanged on the new clusters.
- **K-means path preserved:** with `algorithm == kmeans`, behavior is unchanged.
- **Performance:** on a representative part, `greedy_cover` should be faster than
  the BO+K-means path (no nested optimization). A rough timing comparison in the
  PR description is sufficient.

---

## 8. Explicitly out of scope

- **Occlusion / line-of-sight** — stays in `viewpoint_projection.py`. Do not add
  ray-casting to the coverage predicate.
- **Results JSON format / JSON Schema** — no changes. (A separate effort may add
  a schema; it is unrelated to this task.)
- **Removing the K-means weights** — leave `lambda_weight`, `beta_weight`,
  `point_weight`, `normal_weight` declared to avoid breaking existing configs.
