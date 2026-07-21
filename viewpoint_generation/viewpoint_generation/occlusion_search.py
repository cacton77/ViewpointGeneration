"""Monte Carlo hemisphere occlusion search, shared by fov_clustering.py's
blind-spot rescue and viewpoint_projection.py's direction refinement.

Both callers face the same underlying problem: given an anchor position and
a "preferred" normal direction, find a camera axis within the accessible
imaging cone (standard_normal_threshold) that has unoccluded line of sight to
a set of target points, preferring directions within the narrower photometric
incidence cone (fov_normal_threshold) and, among those, directions closest to
the true normal. Directions outside standard_normal_threshold are glancing
enough that they can't capture surface information, so they are neither sampled
nor considered valid.
"""

import math

import numpy as np
import open3d as o3d


def sample_hemisphere_directions(normal: np.ndarray, num_samples: int,
                                  rng: np.random.Generator,
                                  max_angle: float = math.pi / 2) -> np.ndarray:
    """Cosine-weighted random unit directions within a cone of half-angle
    `max_angle` around `normal` (the full outward hemisphere when
    max_angle == pi/2). Restricting the cone avoids wasting samples on
    glancing directions that can't image the surface. Returns an
    (num_samples, 3) array."""
    normal = normal / np.linalg.norm(normal)
    ref = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, ref)
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)

    # Cosine-weighted disk sampling maps r1 -> polar angle via
    # sin(theta) = sqrt(r1); capping r1 at sin^2(max_angle) truncates the
    # distribution to the cone while preserving its cosine weighting.
    r1 = rng.random(num_samples) * (math.sin(max_angle) ** 2)
    r2 = rng.random(num_samples)
    r = np.sqrt(r1)
    theta = 2.0 * np.pi * r2
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.sqrt(np.clip(1.0 - r1, 0.0, None))

    dirs = x[:, None] * u + y[:, None] * v + z[:, None] * normal
    return dirs / np.linalg.norm(dirs, axis=1, keepdims=True)


def batched_occlusion_grid(raycasting_scene: 'o3d.t.geometry.RaycastingScene',
                            cams: np.ndarray, targets: np.ndarray,
                            epsilon: float) -> np.ndarray:
    """For K candidate camera positions and N shared target points, test
    line-of-sight from every camera to every target in a single batched
    any-hit ray-cast call. Returns a (K, N) bool array, True = occluded.

    Uses the same trick as fov_clustering.py's `_occluded_mask`:
    RaycastingScene.test_occlusions only accepts one scalar tfar for the
    whole ray batch, not one per ray, so each ray's direction is left
    UN-normalized and scaled to length (dist - epsilon) — Open3D measures
    hit distance in units of the direction vector — letting a single
    scalar tfar=1.0 bound every ray to just short of its own target.
    """
    k = len(cams)
    n = len(targets)
    cams_rep = np.repeat(cams, n, axis=0)      # (K*N, 3)
    targets_rep = np.tile(targets, (k, 1))     # (K*N, 3)
    d = targets_rep - cams_rep
    dist = np.linalg.norm(d, axis=1)
    shrink = np.clip(1.0 - epsilon / np.maximum(dist, 1e-12), 0.0, 1.0)
    dirs = d * shrink[:, None]
    rays_np = np.concatenate([cams_rep, dirs], axis=1).astype(np.float32)
    rays = o3d.core.Tensor(rays_np, dtype=o3d.core.Dtype.Float32)
    occluded = raycasting_scene.test_occlusions(rays, tfar=1.0).numpy()
    return occluded.reshape(k, n)


def search_hemisphere_direction(raycasting_scene, anchor: np.ndarray, mean_normal: np.ndarray,
                                 target_points: np.ndarray, fov_normal_threshold: float,
                                 standard_normal_threshold: float,
                                 focal_distance: float, occlusion_epsilon: float,
                                 num_samples: int, rng: np.random.Generator,
                                 candidate_axis: np.ndarray = None):
    """Search the accessible imaging cone at `anchor` for a camera axis with
    unoccluded line of sight to `target_points`, preferring directions
    within `fov_normal_threshold` of `mean_normal` ("photometric" tier)
    over ones between there and `standard_normal_threshold` ("standard"
    tier), and among ties preferring directions closest to `mean_normal`.

    Directions whose incidence angle exceeds `standard_normal_threshold` are
    too glancing to capture surface information: they are neither sampled nor
    selectable (treated as inaccessible), even if they happen to have
    unoccluded line of sight.

    `mean_normal` is always evaluated as a guaranteed baseline candidate,
    and `candidate_axis` (e.g. a previously occlusion-validated axis from
    clustering) is evaluated too if given — the search can never do worse
    than either (subject to accessibility: a candidate_axis outside the
    standard cone is still rejected as inaccessible).

    Returns (axis, tier, visible_fraction):
        axis: unit vector (3,), or None if no accessible direction sees any
            target point at all.
        tier: 'photometric' | 'standard' | None
        visible_fraction: fraction of target_points visible from axis (0..1)
    """
    mean_normal = mean_normal / np.linalg.norm(mean_normal)
    target_points = np.asarray(target_points, dtype=float).reshape(-1, 3)
    n = len(target_points)
    if raycasting_scene is None or n == 0:
        return None, None, 0.0

    guaranteed = [mean_normal]
    if candidate_axis is not None:
        candidate_axis = np.asarray(candidate_axis, dtype=float)
        guaranteed.append(candidate_axis / np.linalg.norm(candidate_axis))
    guaranteed = np.array(guaranteed)

    sampled = sample_hemisphere_directions(mean_normal, max(1, num_samples), rng,
                                            max_angle=standard_normal_threshold)
    directions = np.vstack([guaranteed, sampled])

    cams = anchor + focal_distance * directions
    occluded_grid = batched_occlusion_grid(raycasting_scene, cams, target_points, occlusion_epsilon)
    visible_frac = 1.0 - occluded_grid.mean(axis=1)
    incidence = directions @ mean_normal
    is_photometric = incidence >= math.cos(fov_normal_threshold)
    # Directions outside the standard cone are inaccessible: too glancing to
    # image the surface. Sampled directions already respect this cone, but the
    # guaranteed candidates (candidate_axis in particular) may not, so enforce
    # it here — zeroing their visibility keeps them out of selection.
    is_accessible = incidence >= math.cos(standard_normal_threshold)
    visible_frac = np.where(is_accessible, visible_frac, 0.0)

    best_visible = float(visible_frac.max())
    if best_visible <= 0.0:
        return None, None, 0.0

    best_count = int(round(best_visible * n))
    counts = np.round(visible_frac * n).astype(int)
    allowed_loss = 1 if n > 1 else 0
    eligible = np.where((counts >= max(0, best_count - allowed_loss)) & is_accessible)[0]

    # Among eligible (near-ceiling coverage) directions, prefer photometric
    # tier, then highest incidence (closest to the true normal).
    order = eligible[np.lexsort((incidence[eligible], is_photometric[eligible]))]
    best = order[-1]

    tier = 'photometric' if is_photometric[best] else 'standard'
    return directions[best], tier, float(visible_frac[best])
