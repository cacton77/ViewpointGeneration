# Implementation Brief: Greedy Set-Cover FOV Clustering

**Target repo:** `cacton77/ViewpointGeneration`
**Primary file to change:** `viewpoint_generation/viewpoint_generation/fov_clustering.py`
**Related files to read (do not assume signatures — verify in code):**
`viewpoint_generation/viewpoint_generation.py` (pipeline + config dataclasses,
mesh-load sequencing),
`viewpoint_generation/viewpoint_generation/viewpoint_projection.py` (existing
`RaycastingScene` construction and `set_mesh` pattern to mirror, and the
`check_occlusion` ray-cast style already used for point-to-point visibility),
`nodes/viewpoint_generation_node.py` (ROS parameter declarations),
`README.md` (the `FOVClusteringConfig` table and the `JSON Results Format` section).

> **Revision note:** an earlier version of this brief excluded occlusion /
> line-of-sight from the coverage predicate entirely (see the old Section 4
> and Section 8) on performance grounds. That decision is **reversed** below.
> The pure geometric predicate (DoF + FOV + incidence) has no knowledge of the
> rest of the mesh, so on non-convex parts it over-claims coverage for points
> that are actually self-occluded from a candidate's implied camera pose
> (pocket walls, adjacent ribs/bosses, the far side of a bend). That is a
> correctness gap, not a style choice: the greedy set-cover can converge on a
> viewpoint that geometrically "fits" but cannot actually see everything
> assigned to it, and the error only surfaces downstream when
> `project_viewpoints` produces a pose that has to look through the part. The
> performance concern is addressed below by applying occlusion as a cheap,
> batched, any-hit ray test *after* the geometric filters have already
> shrunk the candidate set — see Section 4 and Section 5.2.

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

**New for this revision — occlusion wiring.** `FOVClustering` currently has no
access to a mesh or `RaycastingScene`; only `ViewpointProjection` does
(populated via `ViewpointProjection.set_mesh`, called from
`ViewpointGeneration.set_mesh_file` once a mesh is loaded). Mirror that exact
pattern on `FOVClustering`: add `FOVClustering.set_mesh(mesh)` that builds its
own `o3d.t.geometry.RaycastingScene` from the **full part mesh** (not the
region point cloud — occluders may belong to any region), and call it from
`ViewpointGeneration.set_mesh_file` alongside the existing
`self.vp.set_mesh(self.mesh)`. Building a `RaycastingScene` is a one-time BVH
build at mesh-load time, not per-region, so building it twice (once for `vp`,
once for `fc`) is a negligible, low-risk cost — do not attempt to share a
single scene instance between the two objects in this pass, that's an
optimization for later if profiling shows it matters.

Occlusion filtering is gated by a new `occlusion_check` bool (default `True`,
see Section 3) so it can be disabled for debugging or for a like-for-like
performance comparison against the pre-occlusion behavior — mirrors the
existing `prune_redundant` toggle pattern.

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
| `occlusion_check` | bool | `True` | Enable the line-of-sight visibility term (Section 4, criterion 4) in the coverage predicate. Disable only for debugging or for an apples-to-apples perf comparison against the pre-occlusion predicate. |
| `occlusion_epsilon` | float (m) | `1e-4` | Shrink margin subtracted from the ray's `tfar` (distance to the target point) in the occlusion test, so the target point's own triangle is never mistaken for an occluder. Tune down for very small parts / tight FOVs, up if near-miss self-occlusion false positives appear. |

Existing fields used by this algorithm: `fov_diameter` (→ `fov_radius =
fov_diameter / 2`) and `dof`.

**`focal_distance` is now needed, but stays out of `FOVClusteringConfig`.**
Occlusion requires a concrete camera position per candidate
(`cam = center + focal_distance · axis`, Section 4), and `focal_distance`
already exists as the single source of truth in `ViewpointProjectionConfig`.
Do **not** add a duplicate `focal_distance` field here — that would let the
two configs drift out of sync (e.g. a ROS param update to one and not the
other). Instead, thread it through as a plain argument: the orchestrator
(`ViewpointGeneration.fov_clustering()`, which already holds `self.vp_config`)
passes `self.vp_config.focal_distance` into
`self.fc.fov_clustering(region_point_cloud, focal_distance=...)`, which forwards
it to `greedy_cover_clustering`. The DoF and FOV lateral tests remain expressed
in the anchor's local frame as before and don't need it; only the new
occlusion ray's origin does.

---

## 4. The coverage predicate (must be normal-aware AND occlusion-aware)

For a candidate anchored at surface point `a` with outward unit normal `n`
(camera sits at `cam = a + focal_distance·n` looking back along `−n`; the
focal plane passes through `a`), a region point `p` with outward unit normal
`np` is **covered** iff all four hold. Work in the anchor's local frame; let
`d = p − a`:

1. **Depth of field (axial):** `|d · n| ≤ dof / 2`
   (signed distance of `p` from the focal plane, along the view axis).
2. **Field of view (lateral):** `‖ d − (d · n) n ‖ ≤ fov_radius`
   (perpendicular distance from the view axis).
3. **Photometric incidence:** `angle(np, n) ≤ fov_normal_threshold`,
   i.e. `np · n ≥ cos(fov_normal_threshold)`. (Outward normals: a well-imaged
   point's normal faces the camera, so it should be near-parallel to `n`.)
4. **Line-of-sight (occlusion):** the segment from `cam` to `p` has no
   intersection with the **full part mesh** at a distance shorter than
   `|p − cam|` (minus `occlusion_epsilon`, so `p`'s own triangle doesn't
   register as its own occluder). Skipped entirely when
   `cfg.occlusion_check` is `False`.

Criteria 1–3 need only the anchor's local frame; criterion 4 is the one place
`focal_distance` enters, since it needs a concrete 3D camera position to cast
a ray from (see Section 3).

**Why occlusion belongs in the predicate, not just downstream in
`viewpoint_projection.py`:** criteria 1–3 alone are a pure frustum/cylinder
test against the region's own points — they have no notion of the rest of the
mesh, so on non-convex geometry they over-claim coverage for points that are
actually hidden from `cam` by another part of the surface (a pocket wall, an
adjacent rib/boss, the far side of a bend). `viewpoint_projection.py`'s
ray-cast runs *after* clustering and can only report that a chosen viewpoint
is bad — it can't tell the greedy search to pick a different one. Catching it
here means the set-cover itself never credits a candidate with coverage it
can't deliver.

**Keeping this cheap (the original performance concern):** criterion 4 is
applied **last**, and only over the points that already survived criteria
1–3 for that candidate (Section 5.2) — not the full broad-radius KD-tree
neighborhood. The DoF/FOV/incidence filters are pure vector math and
typically shrink a candidate's neighborhood to a small fraction of the
points within `broad_radius`; only that survivor set gets a ray. Use the
any-hit `RaycastingScene.test_occlusions` query (boolean per ray), not
`cast_rays`/`list_intersections` — occlusion here only needs a yes/no, and
any-hit lets Embree early-exit instead of resolving the closest hit and its
UVs/normals. Rays for all survivors of one candidate are batched into a
single `test_occlusions` call (Section 6), never issued one ray at a time in
a Python loop.

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
predicate of Section 4. Apply criteria 1–3 (DoF/FOV/incidence) first to shrink
the neighborhood down to survivors, then apply criterion 4 (occlusion) as one
batched `test_occlusions` call over just those survivors — see the
performance note at the end of Section 4. Store `coverage[c]` = set of
region-local point ids that pass all four criteria.

### 5.3 Coverability guarantee
Compute the union of all coverage sets. Any region point not in the union is
intrinsically uncovered (typically a high-curvature spot no sampled candidate
satisfies, or — now that occlusion is enforced — a point whose only
geometrically valid views are all blocked by other geometry). For each such
point, append a **self-anchored** candidate at that point — a footprint
centered on a point trivially satisfies criteria 1–3 (zero lateral offset,
zero axial offset, matching normal). Recompute that candidate's coverage set,
occlusion included: self-anchoring resolves the pre-occlusion stall case, but
a point on a genuine undercut/overhang can still fail criterion 4 even from
directly above its own normal — that residual case is handled by 5.4's safety
break, not by adding more candidates here.

### 5.4 Greedy forward set cover
Initialize `uncovered` = all region points except designated noise points
(region-growth produces noise; PartField produces none — pull the noise set from
whatever the region carries). Repeatedly pick the candidate covering the most
still-uncovered points, append it to `chosen`, and subtract its coverage from
`uncovered`. Stop when `uncovered` is empty. Stopping rule is *coverage
complete* — yielding a (1 + ln n) approximation to minimum viewpoint count.

**Occlusion changes what the existing safety break means.** The loop already
breaks if the best remaining candidate has zero gain (Section 6's `if not
gain: break`). Pre-occlusion this branch was effectively unreachable in
practice, since a self-anchored candidate always geometrically covers itself.
With criterion 4 enforced, it becomes reachable for real parts: a point deep
in a hole or under an overhang can be intrinsically un-viewable by *any*
camera pose respecting `fov_normal_threshold`, not just poorly clustered.
When this triggers, do not treat it as a bug — log a warning naming the region
and the count of leftover `uncovered` points, so an operator can see it's a
genuine inspection blind spot rather than a silent coverage gap. Do **not**
add a new field to the results JSON for this in this pass (Section 8: results
format is out of scope) — a warning log is sufficient here; surfacing blind
spots in the results contract is a separate, future change. See Section 7 for
how this changes the "full coverage" acceptance criterion.

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
                       kdtree, noise_global_ids, raycasting_scene,
                       focal_distance):
    """Greedy set-cover replacement for K-means+BO on ONE region.
    raycasting_scene: o3d.t.geometry.RaycastingScene built from the FULL part
    mesh (FOVClustering.set_mesh) — occluders may belong to any region, not
    just this one.
    focal_distance: from ViewpointProjectionConfig, passed in by the caller.
    Not a FOVClusteringConfig field — see Section 3.
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

    def occluded_mask(cam, targets_local):
        """Batched any-hit line-of-sight test from cam to each target point.
        targets_local: array of local ids that already passed criteria 1-3.
        Returns a bool array, True where the ray is blocked before reaching
        the target (i.e. NOT visible)."""
        if not cfg.occlusion_check or len(targets_local) == 0:
            return zeros(len(targets_local), dtype=bool)
        target_pts = pts[targets_local]                    # (k,3)
        d          = target_pts - cam                       # (k,3)
        dist       = norm(d, axis=1)                         # (k,)
        dirs       = d / dist[:, None]
        rays       = concat([tile(cam, (len(targets_local), 1)), dirs], axis=1)  # (k,6)
        tfar       = dist - cfg.occlusion_epsilon
        # Any-hit query: cheaper than cast_rays/list_intersections since we
        # only need blocked/clear, not hit distance, UVs, or normals.
        return raycasting_scene.test_occlusions(rays, tfar=tfar).numpy()

    def covers(cand):
        """Local ids covered by cand. cand = (center, axis(unit))."""
        survivors = []   # local ids passing criteria 1-3, pre-occlusion
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
            survivors.append(p)
        # Criterion 4: occlusion, batched over just the criteria-1-3
        # survivors (Section 4's performance note / Section 5.2).
        cam = cand.center + focal_distance * cand.axis
        blocked = occluded_mask(cam, survivors)
        return {p for p, b in zip(survivors, blocked) if not b}

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
            break                      # remaining points unreachable (Section 5.4) —
                                       # can now trigger for real on occluded points
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
`unit`, `norm`, `dot`, `acos_clamp` (arccos with input clamped to [-1, 1]),
`zeros`, `tile`, `concat` are trivial numpy helpers — prefer vectorizing
`covers()` over the radius-query result rather than the Python loop shown,
which is written for clarity. `occluded_mask` is already written batched
(one `test_occlusions` call per candidate) since that's the part where a
naive per-point Python loop would actually show up in profiling.

---

## 7. Acceptance criteria

- **Full coverage (occlusion-qualified):** after running, every non-noise
  region point either (a) satisfies the full Section 4 predicate — DoF, FOV,
  incidence, *and* line-of-sight — for at least one emitted cluster's anchor,
  or (b) is in the logged unreachable set from the 5.4 safety break. Add an
  assertion / test that checks every region point is one or the other on a
  sample part — a point silently missing from both is the bug this brief is
  guarding against. On a convex/simple part, case (b) should be empty; use
  a part with at least one pocket, hole, or rib to exercise it.
- **Occlusion respected:** for every emitted cluster, every assigned point has
  an unoccluded line of sight (per criterion 4) from that cluster's implied
  camera position (`anchor + focal_distance · axis`) against the **full part
  mesh**, not just the region's own points. Test on a part where an occluding
  feature belongs to a *different* region than the point it occludes, to
  confirm the RaycastingScene is built from the full mesh and not
  reconstructed per-region.
- **Disjoint partition:** emitted clusters share no points; their union equals
  the region's covered non-noise points (excluding the logged-unreachable set,
  if any).
- **Determinism:** two runs with the same `rng_seed` and inputs produce
  identical clusters.
- **`occlusion_check` toggle behaves as documented:** with `occlusion_check =
  False`, output is identical to the pre-occlusion predicate (criteria 1–3
  only) — this is what makes the performance comparison below meaningful and
  gives an escape hatch if occlusion causes unexpected regressions.
- **No downstream changes required:** `project_viewpoints`, the visualizer,
  traversal/TSP, and the results JSON run unchanged on the new clusters.
- **K-means path preserved:** with `algorithm == kmeans`, behavior is unchanged.
- **Performance:** on a representative part, `greedy_cover` (with
  `occlusion_check = True`) should still be faster than the BO+K-means path
  (no nested optimization). Also report `occlusion_check = True` vs. `False`
  timing on the same part, to confirm the batched-any-hit-after-geometric-
  filter design (Section 4/5.2) keeps the added ray-casting cost small rather
  than dominating. A rough timing comparison in the PR description is
  sufficient for both.

---

## 8. Explicitly out of scope

- **Solid-angle-based candidate scoring** — a continuous "how exposed/open is
  this candidate's implied camera position" measure (hemisphere-sampled
  ambient-occlusion-style score) could be used as a tie-breaker in the 5.4
  greedy selection (`max(..., key=...)`) when two candidates cover the same
  number of points, to prefer anchors that are less brittle to registration
  error or camera placement. That is a real follow-up idea but is **not**
  part of this change — this brief only adds the binary line-of-sight term to
  the coverage predicate (criterion 4). Do not add hemisphere sampling or a
  secondary sort key here.
- **Results JSON format / JSON Schema** — no changes. Unreachable-point
  reporting (Section 5.4) is a log line only, not a new results field. (A
  separate effort may add a schema, and could be where blind-spot reporting
  eventually belongs; it is unrelated to this task.)
- **Removing the K-means weights** — leave `lambda_weight`, `beta_weight`,
  `point_weight`, `normal_weight` declared to avoid breaking existing configs.
- **Sharing one `RaycastingScene` between `FOVClustering` and
  `ViewpointProjection`** — Section 2 calls for each object to build its own
  from the same mesh. Consolidating into a single shared scene is a valid
  later optimization, not required here.
