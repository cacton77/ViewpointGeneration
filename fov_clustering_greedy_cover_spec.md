# Spec: Occlusion-Aware Greedy Set-Cover FOV Clustering

**Status:** Implemented and live (default algorithm).
**Target repo:** `cacton77/ViewpointGeneration`
**Core files:**
- `viewpoint_generation/viewpoint_generation/fov_clustering.py` — candidate
  generation, coverage predicate, greedy set-cover, blind-spot rescue.
- `viewpoint_generation/viewpoint_generation/occlusion_search.py` — shared
  Monte Carlo hemisphere occlusion search, used by both blind-spot rescue and
  viewpoint direction refinement.
- `viewpoint_generation/viewpoint_generation/viewpoint_projection.py` —
  final camera-pose computation, including direction refinement.
- `viewpoint_generation/viewpoint_generation/viewpoint_generation.py` —
  orchestrator: owns the shared `RaycastingScene` (part mesh + ground-plane
  occluder), threads `focal_distance`/`fov_normal_threshold`/`part_center_xy`
  between the two configs' owners.
- `README.md` — `FOVClusteringConfig`/`ViewpointProjectionConfig` tables and
  the `JSON Results Format` section (both current).

> This document describes the algorithm as implemented, not a proposal.
> Earlier revisions of this brief explicitly deferred occlusion checking,
> blind-spot recovery, direction refinement, and a shared raycasting scene as
> "out of scope" or "a later optimization." All of those have since been
> built; see Section 11 for what remains genuinely deferred.

---

## 1. Why greedy set-cover

The original FOV clustering solved a **partitioning** problem (K-means
assigns every point to exactly one cluster) and wrapped it in Bayesian
optimization to discover the cluster count `k`. But inspection-viewpoint
generation is a **covering** problem: a surface point may fall inside several
camera FOVs, and it only needs to fall inside *at least one acceptable* FOV.
Using a partition algorithm here forces a structure the problem doesn't
require, and turns `k` (really a derived quantity of part geometry + FOV
size) into a hyperparameter that must be searched. The BO-over-`k` loop runs
K-means to convergence many times per region — expensive and ad hoc.

`greedy_cover` replaces this with **greedy forward set cover**: directly
minimize the number of viewpoints subject to a hard full-coverage
constraint. Simpler, cheaper (no nested iterative optimization), and
standard.

**Design principle carried through every extension below:** do **not**
minimize FOV overlap. Minimal overlap and full coverage are in tension —
driving overlap to zero on a curved surface opens coverage holes between
footprints. The objective is *minimize viewpoint count subject to full
coverage*; overlap is tolerated, then reduced as a final prune step
(Section 5.5), never used as the optimization target. Every later addition
(occlusion, rescue, refinement, structured candidates) preserves this: none
of them are allowed to silently drop coverage in exchange for a
nicer-looking result.

---

## 2. Architecture

`FOVClusteringConfig.algorithm` selects between `kmeans` (legacy,
Bayesian-optimized K-means, still fully supported) and `greedy_cover`
(default). Both paths return the same container shape from
`FOVClustering.fov_clustering()` (Section 5.7), so `project_viewpoints()`,
the visualizer, traversal/TSP, and the results JSON never need to know which
algorithm produced a given cluster.

**Shared `RaycastingScene`.** `ViewpointGeneration.set_mesh_file()` builds
one `o3d.t.geometry.RaycastingScene` from the full part mesh and hands the
same instance to both `FOVClustering` (`set_scene`, used by the coverage
predicate and blind-spot rescue) and `ViewpointProjection` (`set_scene`,
used by direction refinement). One BVH build per mesh load, not two — an
earlier revision of this brief allowed each object to build its own scene as
a low-risk stopgap; that has since been consolidated.

**Ground-plane occluder.** The same scene also includes a large flat quad
(`ViewpointGeneration._build_ground_plane_mesh`, `ground_plane_half_size =
5.0` m half-width) so that no ray anywhere in the pipeline — coverage
predicate, blind-spot rescue, or direction refinement — can credit a
candidate whose camera position would sit underground (below the
table/turntable the part sits on). The plane sits at `ground_plane_z =
-0.02`, **not** at `z = 0`: placing it exactly at the nominal table level
made it coincide with any mesh point that itself rests at `z = 0` (e.g. the
base of a part flush with the table), and the occlusion ray's
self-intersection-avoidance shrink (`occlusion_epsilon`, meant to stop a ray
just short of the *target's own* triangle) then also stopped it just short
of the *coincident ground plane* — so the ray never geometrically reached
z=0 and the ground silently failed to register as blocking. `-0.02` clears
`occlusion_epsilon`'s entire declared range (`[0, 0.01]`) with 2x margin
while remaining negligible next to any genuinely-underground camera
position. Verified: reproduced the coincidence bug numerically against a
minimal scene, confirmed the offset fixes it, and confirmed it doesn't
introduce false occlusion for ordinary above-ground views.

**Candidate generation** is pluggable per region: default blue-noise
farthest-point sampling, or an opt-in structured cylindrical grid
(`structured_candidates`, Section 7).

---

## 3. Configuration reference

**`FOVClusteringConfig`** (all declared as ROS parameters under
`regions.fov_clustering.`):

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `algorithm` | str enum | `greedy_cover` | `kmeans` or `greedy_cover` |
| `fov_diameter` | float (m) | `0.02` | FOV diameter at the focal plane; `fov_radius = fov_diameter/2` |
| `dof` | float (m) | `0.02` | Depth of field |
| `fov_normal_threshold` | float (rad) | `pi/4` | Max incidence angle: a point is only "covered" (and a direction only classified `photometric`) if its surface normal is within this cone of the view axis. The photometric-stereo acceptability constraint |
| `candidate_spacing` | float (m) | `0.0` → auto = `fov_radius` | Spacing of candidate anchors — both farthest-point spacing and the structured grid's elevation/azimuth spacing |
| `prune_redundant` | bool | `True` | Enable the redundant-viewpoint prune tail step (Section 5.5) |
| `rng_seed` | int | `0` | Seed for candidate sampling and blind-spot rescue, for reproducibility |
| `occlusion_check` | bool | `True` | Enable the line-of-sight term (criterion 4) in the coverage predicate. Disabling reproduces the pre-occlusion (criteria 1-3 only) behavior exactly, including disabling rescue (see below) |
| `occlusion_epsilon` | float (m) | `1e-4` | Shrink margin subtracted from an occlusion ray's far bound so a point's own triangle is never mistaken for its own occluder |
| `rescue_search` | bool | `True` | Attempt a Monte Carlo hemisphere search (Section 5.4.5) for an alternative viewing angle before giving up on an occlusion-blind point. Only runs when `occlusion_check` is also enabled |
| `rescue_samples` | int | `64` | Hemisphere samples per blind-spot rescue attempt |
| `structured_candidates` | bool | `False` | Lay candidate anchors on a per-region cylindrical grid instead of farthest-point sampling (Section 7) |

Unused by `greedy_cover`, kept declared so existing configs don't break:
`point_density`, `lambda_weight`, `beta_weight`, `max_point_out_percentage`,
`point_weight`, `normal_weight`, `number_of_runs`, `maximum_iterations`
(all K-means-only).

**`ViewpointProjectionConfig`**:

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `focal_distance` | float (m) | `0.3` | Distance from surface to camera along the view axis |
| `hemisphere_points` | int | `64` | Hemisphere samples for the direction-refinement search in `generate_viewpoint` (Section 6), per cluster, not per point |

**Cross-config threading, not duplication.** `focal_distance`,
`fov_normal_threshold`, `occlusion_epsilon`, and `rng_seed` are each owned by
exactly one config and threaded across the boundary as plain call arguments
by the orchestrator (`ViewpointGeneration.fov_clustering()` /
`project_viewpoints()`, which hold both configs) rather than duplicated —
e.g. `self.fc.fov_clustering(region_point_cloud,
focal_distance=self.vp_config.focal_distance, ...)` and
`self.vp.generate_viewpoint(fov_points, fov_normals,
fov_normal_threshold=self.fc_config.fov_normal_threshold, ...)`. A duplicate
field would let the two configs drift out of sync (a ROS param update to
one and not the other).

---

## 4. The coverage predicate

For a candidate anchored at `center` with unit axis `axis` (camera at
`cam = center + focal_distance · axis`, focal plane through `center`), a
region point `p` with outward unit normal `np` is **covered** iff all four
hold (`d = p − center`):

1. **Depth of field (axial):** `|d · axis| ≤ dof / 2`
2. **Field of view (lateral):** `‖ d − (d · axis) axis ‖ ≤ fov_radius`
3. **Photometric incidence:** `np · axis ≥ cos(fov_normal_threshold)`
4. **Line-of-sight (occlusion):** the segment from `cam` to `p` has no
   intersection with the shared scene (full part mesh **and** ground-plane
   occluder, Section 2) closer than `|p − cam|` minus `occlusion_epsilon`.
   Skipped entirely when `occlusion_check` is `False`.

One relaxation: during blind-spot rescue (Section 5.4.5), the candidate's own
force-included point may satisfy criterion 3 by construction rather than by
the raw dot-product test — see `_coverage_set(..., force_include=)` below.
Criteria 1/2/4 are never relaxed for any point, rescued or not.

**Performance.** Criterion 4 is applied **last**, only over the points that
already survived criteria 1–3 for that candidate — not the full
broad-radius neighborhood. Uses the any-hit `RaycastingScene.test_occlusions`
query (boolean per ray) rather than `cast_rays`/`list_intersections`, batched
into one call per candidate (`_occluded_mask`) rather than issued one ray at
a time.

`test_occlusions` accepts only a single scalar `tfar` for an entire ray
batch, not one per ray — so the per-ray "stop just short of this ray's own
target" bound is expressed by leaving each ray's direction vector
**un-normalized**, scaled to length `(dist − occlusion_epsilon)`. Open3D
measures hit distance in units of the direction vector, so a single scalar
`tfar = 1.0` then bounds every ray in the batch to just short of its own
target in one call. (Verified empirically against the installed Open3D
version; this is not documented Open3D behavior, just the observed
contract.)

---

## 5. Algorithm — per region

Run independently **per region**, never globally, so a footprint cannot
straddle two semantic parts. A little extra redundancy at region boundaries
is an acceptable trade.

### 5.1 Candidate generation
Two interchangeable strategies, selected by `structured_candidates`:

- **Farthest-point (default).** Greedy maximin sampling at `candidate_spacing`
  (auto = `fov_radius`), seeded by `rng_seed`. `_farthest_point_sample`.
- **Structured cylindrical grid (opt-in).** `_structured_grid_sample` bins
  this region's points by elevation (world Z) and azimuth (angle around a
  vertical axis through the *whole part's* center, not each region's own —
  `part_center_xy`, computed once from the mesh's AABB center and threaded
  in by the orchestrator, so different regions' bins share the same
  absolute reference frame and land at genuinely shared elevations/azimuths,
  not just the same coarse bucket). Elevation bin width and azimuthal arc
  length both use `candidate_spacing`; the azimuthal *angular* step is
  derived per-region from that region's mean radius from the axis
  (`angular_step = spacing / mean_radius`). Regions whose mean radius is
  smaller than `spacing` are too close to the axis for azimuth to be
  meaningful (the "pole" case) and fall back to elevation-only bands. Each
  occupied bin keeps whichever point is closest to the bin's exact center
  (in elevation/arc-length units) — not the first point encountered — so an
  anchor's actual position tracks the bin, not wherever farthest-point-style
  ordering happened to put it. Fully deterministic (no RNG).

Either way, each candidate = `(center = anchor_pt, axis = anchor_normal)`.

### 5.2 Precompute coverage sets
For each candidate, KD-tree radius query over the region's points, then the
Section 4 predicate. `coverage[c]` = set of region-local point ids passing
all four criteria (or three, for a force-included rescue point).

### 5.3 Coverability guarantee
Any region point not in the union of all coverage sets gets a **self-anchored**
candidate at that point — trivially satisfies criteria 1–3 (zero
offset, matching normal). Recomputed with occlusion included: self-anchoring
resolves the pre-occlusion stall case, but a point on a genuine
undercut/overhang can still fail criterion 4 even looking straight along its
own normal. That residual case is what 5.4's rescue step (below) exists for.

### 5.4 Greedy forward set cover
`uncovered` = all region points. Repeatedly pick the candidate covering the
most still-uncovered points, append to `chosen`, subtract its coverage from
`uncovered`. Breaks early if the best remaining candidate has zero gain.

Pre-occlusion, that break was effectively unreachable (a self-anchored
candidate always geometrically covers itself). With criterion 4 enforced, a
point deep in a hole or under an overhang can be intrinsically un-viewable
by any camera pose respecting `fov_normal_threshold` from its own straight
normal — that's what 5.4.5 attempts to rescue before the point is finally
accepted as a genuine blind spot.

### 5.4.5 Blind-spot rescue
Gated on `occlusion_check and rescue_search and uncovered`. For each still-
uncovered point `p` (stable sorted order, skipping any already incidentally
covered by an earlier rescue this pass):

1. `search_hemisphere_direction(scene, pts[p], normals[p],
   target_points=[pts[p]], fov_normal_threshold, focal_distance,
   occlusion_epsilon, rescue_samples, rng)` — searches the **full outward
   hemisphere**, not just the incidence cone (Section 6). If we can't do
   photometric imaging of a point, we can still potentially capture a
   standard (non-photometric-stereo) image of it from a wider angle, as
   long as *something* clears occlusion.
2. If a working axis is found (`visible_fraction > 0`), build a new
   candidate at `(pts[p], axis)` via `_coverage_set(pts[p], axis,
   force_include=p)` — `force_include` skips the criterion-3 incidence gate
   for exactly this point, since the axis was deliberately chosen to rescue
   it and may trade photometric quality for actual visibility; criteria
   1/2/4 still apply normally, to this point and to any neighbors the same
   candidate also happens to cover.
3. Add to `chosen`, subtract its coverage from `uncovered`.

Rescue does **not** track or report imaging tier (photometric vs. standard)
— it only needs to *achieve* coverage. The authoritative tier is determined
once, correctly, at projection time (Section 6), using the full cluster and
a proven-good axis, regardless of which candidate — rescued or not —
produced the cluster.

Points still uncovered after rescue are genuine blind spots: `blind_spots =
{'points': [...], 'reason': ...}` (Section 9). `reason` classification
(`occluded` / `geometric` / `mixed`) is unchanged from before rescue existed,
computed against `covered_union_geo` (points that passed criteria 1–3 for
*any* candidate, regardless of occlusion outcome) — it just now runs after a
rescue attempt has already been made, so `occluded` is a strictly stronger
claim than before: not just "the straight-normal view was blocked" but "no
direction in the hemisphere works either."

### 5.5 Redundancy prune
If `prune_redundant`: iterate `chosen` (drop later picks first); drop any
selected candidate whose coverage set is a subset of the union of the other
selected candidates' coverage sets. Runs after coverage is guaranteed
(including after rescue), so it can't reopen a hole — this holds for rescue
candidates too, since `others_union` is recomputed live against the current
`chosen` set on every prune iteration.

### 5.6 Resolve overlap into disjoint clusters
Assign each covered point to exactly one chosen candidate: smallest normal
angle to the point, tie-broken by smallest axial offset. `owners` is never
empty for a point that reached this step, by construction of 5.4/5.4.5 —
the old fallback that force-assigned an "uncovered" point to the nearest
chosen candidate regardless of predicate was removed once this was proven.

### 5.7 Emit
Each chosen candidate with at least one assigned point becomes one cluster,
emitted as `{'points': [global indices], 'axis': [unit vector]}` — `axis` is
that candidate's occlusion-validated view axis, carried forward so
`project_viewpoints()` can reuse it as a guaranteed-good starting point
instead of re-deriving an unvalidated one from scratch (Section 6). The
`kmeans` path emits the same shape with `'axis': None` (kmeans never
validates occlusion during clustering, so there's nothing proven to
persist).

---

## 6. Viewpoint direction refinement (`project_viewpoints`)

**The gap this closes.** `ViewpointProjection.generate_viewpoint()` computes
each cluster's direction as the mean of its assigned points' normals — a
different direction than whatever single candidate axis was actually
occlusion-tested during clustering (Section 5). That mean was never itself
re-validated against occlusion, so the "occlusion respected" guarantee
silently didn't cover the direction that actually got used. Averaging
several points' normals can drift toward a direction with worse (or zero)
visibility of the same points than the original per-candidate axis had.

**The fix.** `generate_viewpoint(surface_points, surface_normals,
fov_normal_threshold=None, occlusion_epsilon=1e-4, rng_seed=0,
candidate_axis=None)` always runs the mean-normal direction through
`search_hemisphere_direction` (when a raycasting scene and
`fov_normal_threshold` are available), scored against the cluster's own
`surface_points` — the persisted `candidate_axis` (5.7) is passed in as an
**additional guaranteed candidate** alongside the mean normal itself, so
refinement is provably at least as good as clustering already established,
deterministically, regardless of RNG state. This matters concretely: for a
cluster that exists *only* because 5.4.5 rescued a single point, the mean
normal *is* that same point's own normal — the exact direction already
proven occluded, which is why it needed rescuing in the first place. Without
`candidate_axis` as a guaranteed floor, refinement would have to
*independently rediscover* a working direction via its own random draw, with
no guarantee of success. Verified directly: forced this scenario end-to-end
through the real orchestrator with a mismatched RNG seed between clustering
and projection, confirming `candidate_axis` alone guarantees an unoccluded
final direction.

**`imaging_mode` tiers.** The winning direction is tagged:
- `photometric` — within `fov_normal_threshold` of the true mean normal.
  Supports photometric-stereo normal-map reconstruction.
- `standard` — only an off-cone angle is occlusion-clear. Still a usable
  image, just not photometric-quality. Also the fallback tag if the search
  totally fails (returns `None` — only reachable for `kmeans`-sourced
  clusters, which have no `candidate_axis` floor); an unvalidated direction
  must never claim the better tier.

`search_hemisphere_direction`'s ranking (shared with rescue, Section 5.4.5):
score every candidate direction (guaranteed ones + `num_samples`
cosine-weighted random hemisphere samples) by `visible_fraction` against the
target points. Let `best_count = round(max(visible_fraction) · n)`.
`allowed_loss = 1` for `n > 1`, else `0` (no partial-credit concept for a
single-point rescue target). A direction is "eligible" if its covered-point
count is within `allowed_loss` of `best_count`. Among eligible directions,
prefer `photometric` tier, then highest incidence (closest to the true
normal). Coverage correctness gates the choice first; tier is a
tie-break among near-ceiling options, not a competing objective — an
earlier draft of this ranking sorted by tier first and caught a real failure
mode in review: it could flip a 23/24-covered photometric direction to a
24/24-covered standard one over a single boundary point, sensitive to
Monte-Carlo sampling noise.

Both `FOVClustering`'s rescue and `ViewpointProjection`'s refinement share
the exact same `occlusion_search.py` primitives (`sample_hemisphere_directions`,
`batched_occlusion_grid`, `search_hemisphere_direction`) — one implementation
of "search a hemisphere for an unoccluded direction," two call sites.

`viewpoint.imaging_mode` is stored in the results JSON (Section 9) and drives
the visualizer: the `Viewpoint Marker` overlay is green for `photometric`,
yellow for `standard` (blue still overrides both when the viewpoint is
selected).

---

## 7. Structured candidate sampling

See 5.1 for the mechanism. The motivation: default farthest-point sampling
produces a blue-noise pattern — candidates land wherever maximin spacing
happens to put them, with no relationship between one region's anchor
positions and a neighboring region's. `structured_candidates` trades some of
the greedy algorithm's freedom to pick the most locally-efficient anchor for
a physically-motivated, human-interpretable layout: since the system drives
an actual turntable, a **cylindrical** (elevation + azimuth around a shared
vertical axis) parametrization is the natural fit — not spherical, which
would need a different physical capture mechanism.

Verified on a hexagonal-prism test part (six side-wall regions,
independently segmented and clustered): raw candidate-anchor elevations
across all six regions had a mean deviation of **0.19mm** from the nearest
shared 10mm elevation line under `structured_candidates`, versus **2.06mm**
under default farthest-point sampling — anchors from independently-clustered
regions genuinely land on the same rings, not merely the same coarse bucket.
(An initial implementation that kept the *first* point encountered per bin,
rather than the one closest to the bin's center, only achieved the latter —
caught by direct measurement, not by inspection.)

---

## 8. Pseudo-code

```python
def greedy_cover_clustering(pts, normals, cfg, kdtree, raycasting_scene,
                             focal_distance, part_center_xy=None):
    """pts, normals: this region's own point cloud (already region-local).
    raycasting_scene: shared scene (full part mesh + ground-plane occluder).
    Returns (clusters, blind_spots):
      clusters: [{'points': [local ids], 'axis': unit_vec3}, ...]
      blind_spots: {'points': [...]} + {'reason': ...} when non-empty."""
    fov_radius = cfg.fov_diameter / 2.0
    half_dof   = cfg.dof / 2.0
    spacing    = cfg.candidate_spacing or fov_radius
    cos_thr    = cos(cfg.fov_normal_threshold)
    rng        = seeded_rng(cfg.rng_seed)
    occlusion_active = cfg.occlusion_check and raycasting_scene is not None

    covered_union_geo = set()  # criteria-1-3 survivors, any candidate, any occlusion outcome

    def coverage_set(center, axis, force_include=None):
        idx = kdtree.query_radius(center, r=broad_radius(fov_radius, half_dof))
        d = pts[idx] - center
        axial   = d @ axis
        lateral = norm(d - outer(axial, axis))
        incidence_ok = (normals[idx] @ axis) >= cos_thr
        if force_include is not None:
            incidence_ok |= (idx == force_include)
        survivors = idx[(abs(axial) <= half_dof) & (lateral <= fov_radius) & incidence_ok]
        covered_union_geo.update(survivors)
        if len(survivors) == 0 or not occlusion_active:
            return set(survivors)
        cam = center + focal_distance * axis
        blocked = occluded_mask(raycasting_scene, cam, pts[survivors], cfg.occlusion_epsilon)
        return {p for p, b in zip(survivors, blocked) if not b}

    # 5.1 candidate anchors
    if cfg.structured_candidates and part_center_xy is not None:
        anchor_ids = structured_grid_sample(pts, spacing, part_center_xy)
    else:
        anchor_ids = farthest_point_sample(pts, spacing, rng)
    cand_centers = [pts[a] for a in anchor_ids]
    cand_axes    = [normals[a] for a in anchor_ids]
    coverages    = [coverage_set(c, a) for c, a in zip(cand_centers, cand_axes)]

    # 5.3 self-anchor guarantee
    covered_union = union(coverages)
    for p in range(len(pts)):
        if p not in covered_union:
            cand_centers.append(pts[p]); cand_axes.append(normals[p])
            coverages.append(coverage_set(pts[p], normals[p]))

    # 5.4 greedy forward set cover
    uncovered, chosen = set(range(len(pts))), []
    while uncovered:
        best = argmax_c(len(coverages[c] & uncovered))
        gain = coverages[best] & uncovered
        if not gain:
            break
        chosen.append(best); uncovered -= gain

    # 5.4.5 blind-spot rescue
    if occlusion_active and cfg.rescue_search and uncovered:
        for p in sorted(uncovered):
            if p not in uncovered:
                continue
            axis, tier, vis = search_hemisphere_direction(
                raycasting_scene, pts[p], normals[p], [pts[p]],  # target_points = just this point
                cfg.fov_normal_threshold, focal_distance, cfg.occlusion_epsilon,
                cfg.rescue_samples, rng)
            if axis is None or vis <= 0.0:
                continue
            cov = coverage_set(pts[p], axis, force_include=p)
            if p not in cov:
                continue
            cand_centers.append(pts[p]); cand_axes.append(axis); coverages.append(cov)
            chosen.append(len(coverages) - 1)
            uncovered -= cov

    blind_spots = classify_blind_spots(uncovered, covered_union_geo)  # occluded/geometric/mixed

    # 5.5 redundancy prune
    if cfg.prune_redundant:
        for c in reversed(list(chosen)):
            if len(chosen) == 1: break
            others = union(coverages[o] for o in chosen if o != c)
            if coverages[c] <= others:
                chosen.remove(c)

    # 5.6 disjoint assignment
    assignment = {c: [] for c in chosen}
    for p in set(range(len(pts))) - uncovered:
        owners = [c for c in chosen if p in coverages[c]]
        owner = min(owners, key=lambda c: (angle(normals[p], cand_axes[c]),
                                            abs(dot(pts[p] - cand_centers[c], cand_axes[c]))))
        assignment[owner].append(p)

    # 5.7 emit
    clusters = [{'points': members, 'axis': cand_axes[owner]}
                for owner, members in assignment.items() if members]
    return clusters, blind_spots


def search_hemisphere_direction(scene, anchor, mean_normal, target_points,
                                 fov_normal_threshold, focal_distance,
                                 occlusion_epsilon, num_samples, rng,
                                 candidate_axis=None):
    """Shared by rescue (target_points = one point) and refinement
    (target_points = a whole cluster). mean_normal (and candidate_axis, if
    given) are always evaluated as guaranteed baseline candidates."""
    directions = [mean_normal] + ([candidate_axis] if candidate_axis is not None else [])
    directions += sample_hemisphere_directions(mean_normal, num_samples, rng)  # cosine-weighted

    cams = [anchor + focal_distance * d for d in directions]
    occluded_grid = batched_occlusion_grid(scene, cams, target_points, occlusion_epsilon)  # ONE call
    visible_frac = 1.0 - occluded_grid.mean(axis=1)
    incidence    = [d @ mean_normal for d in directions]
    is_photometric = incidence >= cos(fov_normal_threshold)

    if max(visible_frac) <= 0.0:
        return None, None, 0.0
    best_count    = round(max(visible_frac) * len(target_points))
    allowed_loss  = 1 if len(target_points) > 1 else 0
    eligible      = [i for i in range(len(directions))
                      if round(visible_frac[i] * len(target_points)) >= best_count - allowed_loss]
    best = max(eligible, key=lambda i: (is_photometric[i], incidence[i]))  # tier, then closeness to normal
    tier = 'photometric' if is_photometric[best] else 'standard'
    return directions[best], tier, visible_frac[best]
```

---

## 9. Results JSON schema

`region.clusters[i]`:
```json
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
```
`candidate_axis` is present only for `greedy_cover`-sourced clusters (omitted
— not `null` — for `kmeans`). `viewpoint` and `imaging_mode` only exist after
`project_viewpoints()` runs.

`region.blind_spots`:
```json
{"points": [17, 42, 88], "reason": "occluded"}
```
`{"points": []}` when empty (always, for `kmeans`). `reason` present only
when `points` is non-empty: `occluded` (every affected point had a
geometrically-valid candidate that occlusion alone ruled out, even after
rescue), `geometric` (none did — e.g. a degenerate surface normal), `mixed`
(both).

This is a change from the original brief, which explicitly scoped blind-spot
reporting and `RaycastingScene` sharing as future work with "no results JSON
changes." Both are now part of the live schema; see README's `JSON Results
Format` section for the authoritative, up-to-date reference.

---

## 10. Acceptance criteria (current, verified this session)

- **Full coverage after rescue:** every non-noise region point either (a)
  satisfies the full Section 4 predicate for at least one emitted cluster's
  anchor, or (b) is in `blind_spots` — and only after a hemisphere rescue
  attempt has already failed for it, when `rescue_search` is enabled.
- **Occlusion respected against the full scene:** every assigned point has
  unoccluded line of sight from its cluster's *actually emitted* camera
  position (not just the original candidate axis) — closed by Section 6's
  refinement, tested against a scenario where clustering and projection use
  different RNG seeds.
- **No viewpoint underground:** verified end-to-end on a part with a face
  resting exactly at the table level (z=0) — the scenario that originally
  exposed the ground-plane/epsilon coincidence bug (Section 2).
- **Disjoint partition:** emitted clusters share no points.
- **Determinism:** identical `rng_seed` and inputs (including the cached
  per-region point cloud) produce identical clusters, rescues, and refined
  directions.
- **Toggles behave as documented:** `occlusion_check=False` reproduces the
  pre-occlusion predicate exactly (rescue never triggers, since `uncovered`
  is always empty in that mode — no separate guard needed).
  `rescue_search=False` reproduces pre-rescue blind-spot behavior exactly.
- **`kmeans` path preserved:** unchanged behavior, now returning the same
  `{'points', 'axis': None}` container shape as `greedy_cover` for a
  consistent downstream contract; `imaging_mode` still gets set via
  `generate_viewpoint`'s fallback path (tagged `standard`, since kmeans never
  validates occlusion).
- **`structured_candidates` measurably tightens cross-region alignment:**
  quantified on a multi-region test part (Section 7), not just visually
  inspected.

---

## 11. Explicitly out of scope / deferred

- **General per-candidate axis optimization for every candidate**, not just
  rescued/blind ones — i.e. running the hemisphere search up front to
  improve *all* anchors' robustness, not only to rescue coverage failures.
  Considered and explicitly declined in favor of the targeted rescue-only
  scope implemented here; would multiply ray-casting cost across the whole
  candidate pool for a robustness benefit, not a correctness one.
- **Global (whole-part) structured grid.** `structured_candidates` is
  per-region by design — a single global cylindrical/spherical grid was
  considered and declined because it only produces sensible results on
  roughly star-convex/cylindrical parts; per-region scoping stays robust to
  arbitrary topology while a shared world-frame reference (`part_center_xy`,
  elevation from world Z) still gets the cross-region alignment benefit.
- **`ground_plane_half_size` / `ground_plane_z` are not exposed as ROS
  parameters** — hardcoded `ViewpointGeneration` class constants. Would be
  a small follow-up if a workspace ever needs a non-default ground
  reference.
- **Removing the K-means-only weights** — `lambda_weight`, `beta_weight`,
  `point_weight`, `normal_weight` remain declared and unused by
  `greedy_cover`, to avoid breaking existing configs.
- **Per-point rescue diagnostics** — blind-spot rescue logs a warning naming
  the region and leftover count, and `blind_spots.reason` gives a
  region-level classification, but individual rescue attempts (which
  directions were tried, why they failed) are not persisted anywhere.
- **Light-source visibility for the `photometric` tier (not yet implemented —
  design only).** `imaging_mode == 'photometric'` currently means only that
  the camera axis is within `fov_normal_threshold` of the true surface
  normal (Section 6). That's necessary but not sufficient: photometric
  stereo also needs each of several light sources to actually illuminate the
  point, and a light whose path to the point is obstructed (self-shadowing
  from a nearby rib, pocket wall, or undercut — the same kind of geometry
  that already motivates the camera occlusion check) contributes no usable
  signal for it, even when the camera's own view is perfectly clear and
  on-axis. The physical rig arranges light sources in concentric rings
  around the camera's principal axis, so each light's 3D position is
  derivable from the candidate/cluster's camera pose (position + axis) plus
  the rig's known ring radii/counts — once that parametrization is decided,
  checking a light is a batched any-hit ray-cast from the light's position
  to the target point(s) against the same shared `RaycastingScene`
  (part mesh + ground plane) already used everywhere else, mechanically the
  same shape as `occlusion_search.batched_occlusion_grid` (K origins × N
  shared targets → one call) — this should slot into the existing
  infrastructure rather than requiring new ray-casting machinery. The
  natural hook point is tier classification itself: `photometric` would
  require both the existing incidence condition *and* light visibility;
  falling short would downgrade to `standard` (or, if only some lights are
  blocked, possibly a new intermediate tier — open question, not decided).
  Also open: how many of the ring's lights must be unobstructed (all of
  them, since photometric-stereo normal reconstruction typically needs
  several non-degenerate illumination directions to be usable, or some
  minimum subset), the exact ring geometry (radii, light count per ring,
  offset from the focal plane), and whether this check runs for every
  candidate during clustering (expensive — multiplies ray count by light
  count on top of the existing per-candidate occlusion check) or only at
  final projection time on the already-narrowed candidate/cluster set.
