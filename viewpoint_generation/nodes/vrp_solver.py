#!/usr/bin/env python3
import random
import numpy as np
from dataclasses import dataclass, field

from moveit.core.robot_state import RobotState
from geometry_msgs.msg import Pose

try:
    import cupy as cp
    _CUPY = True
except ImportError:
    _CUPY = False

# Home group state from inspection_cell.srdf
# [turntable_disc_joint, shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]
DEPOT_Q = np.array([0.0, 3.14, -1.8398, -1.8224, -1.0, 1.57, 0.0])


@dataclass
class IKResult:
    q_all: np.ndarray             # (N, 7) joint configs; NaN rows where IK failed entirely
    manip_all: np.ndarray         # (N,) Yoshikawa manipulability index
    cartesian_fallback: np.ndarray  # (N,) bool — True where Cartesian proxy was used
    positions: np.ndarray         # (N, 3) viewpoint positions in object_frame (for fallback cost)
    status: dict = field(default_factory=dict)  # global_idx → {'region', 'cluster', 'tier'}


@dataclass
class VRPSolution:
    region_order: list    # optimized inter-region visit sequence (region indices)
    region_paths: list    # region_paths[r] = local cluster visit order for region r
    cost: float
    intra_cost: float     # sum of within-region edge costs
    inter_cost: float     # sum of region-to-region transition costs
    depot_cost: float     # depot→first_vp + last_vp→depot
    algorithm: str = ''


class SingularityHandler:
    # Turntable offsets (rad) tried as alternative IK seeds to escape near-singular configs
    _SEED_OFFSETS = [np.pi/12, -np.pi/12, np.pi/6, -np.pi/6,
                     np.pi/4, -np.pi/4, np.pi/2, -np.pi/2]

    def __init__(self, robot_model, planning_group='disc_to_ur5e', threshold=0.05):
        self._model = robot_model
        self._group = planning_group
        self.threshold = threshold  # Yoshikawa index below this is near-singular

    def manipulability(self, robot_state) -> float:
        """Yoshikawa index: sqrt(det(J @ J.T)) where J is the (6, 7) geometric Jacobian."""
        J = robot_state.get_jacobian(self._group, np.zeros((3, 1)))
        return float(np.sqrt(max(0.0, np.linalg.det(J @ J.T))))

    def solve_ik_from_seed(self, pose: Pose, seed_q: np.ndarray,
                            tip_link: str = 'eoat_camera_link') -> tuple:
        """IK seeded from seed_q (tier 0). Falls through to standard 3-tier on failure."""
        state = RobotState(self._model)
        state.set_joint_group_positions(self._group, seed_q)
        if state.set_from_ik(self._group, pose, tip_link):
            state.update()
            manip = self.manipulability(state)
            if manip >= self.threshold:
                return np.array(state.get_joint_group_positions(self._group)), manip, 0
        return self.solve_ik(pose, tip_link)

    def solve_ik(self, pose: Pose, tip_link: str = 'eoat_camera_link') -> tuple:
        """3-tier IK with singularity avoidance. Returns (q, manip, tier) or (None, 0.0, 3)."""
        state = RobotState(self._model)

        # Tier 1: direct IK seeded from DEPOT_Q
        state.set_joint_group_positions(self._group, DEPOT_Q)
        if state.set_from_ik(self._group, pose, tip_link):
            state.update()
            manip = self.manipulability(state)
            if manip >= self.threshold:
                return np.array(state.get_joint_group_positions(self._group)), manip, 1

        # Tier 2: perturb turntable seed, keep highest-manipulability result
        best_q, best_manip = None, 0.0
        for offset in self._SEED_OFFSETS:
            seed = DEPOT_Q.copy()
            seed[0] += offset
            state.set_joint_group_positions(self._group, seed)
            if state.set_from_ik(self._group, pose, tip_link):
                state.update()
                manip = self.manipulability(state)
                if manip > best_manip:
                    best_manip = manip
                    best_q = np.array(state.get_joint_group_positions(self._group))

        if best_q is not None:
            return best_q, best_manip, 2

        return None, 0.0, 3  # Cartesian proxy handled in precompute_ik


class VRPSolver:

    # Turntable weight lower than arm joints: it is a continuous rotation joint and
    # cheaper to reposition without moving the arm links.
    # Configurable via the vrp_joint_weights parameter on viewpoint_traversal_node.
    DEFAULT_WEIGHTS = [0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    def __init__(
        self,
        joint_weights=None,
        use_gpu=False,
        singularity_threshold=0.05,  # Yoshikawa index below this → near-singular
        singularity_action='avoid',  # 'avoid' | 'warn'
        logger=None,
    ):
        self.joint_weights = np.array(joint_weights or self.DEFAULT_WEIGHTS, dtype=float)
        # Pre-scale by sqrt(w): ||sqrt(w)⊙qa - sqrt(w)⊙qb||_2 == weighted L2 c(qa, qb)
        self._w = np.sqrt(self.joint_weights)
        self.use_gpu = use_gpu and _CUPY
        self.singularity_threshold = singularity_threshold
        self.singularity_action = singularity_action
        self.logger = logger

    # ── Distance primitives ───────────────────────────────────────────────────

    def joint_dist_matrix(self, q_all: np.ndarray) -> np.ndarray:
        """Weighted joint-space (N×N) distance matrix; GPU-accelerated when CuPy is available."""
        q_scaled = q_all * self._w
        if self.use_gpu:
            q_gpu = cp.asarray(q_scaled)
            diff = q_gpu[:, None, :] - q_gpu[None, :, :]
            return cp.asnumpy(cp.sqrt(cp.sum(diff ** 2, axis=-1)))
        diff = q_scaled[:, None, :] - q_scaled[None, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=-1))

    def _depot_dists(self, q_all: np.ndarray) -> np.ndarray:
        """Weighted joint-space distance from DEPOT_Q to each row of q_all."""
        return np.linalg.norm((q_all - DEPOT_Q) * self._w, axis=1)

    def build_cost_matrix(self, ik_result: IKResult) -> np.ndarray:
        """Joint-space distance matrix; Cartesian proxy fills rows/cols for IK-failed viewpoints."""
        dm = self.joint_dist_matrix(ik_result.q_all)
        fallback = ik_result.cartesian_fallback
        if not np.any(fallback):
            return dm

        cart = np.linalg.norm(
            ik_result.positions[:, None, :] - ik_result.positions[None, :, :], axis=-1)
        finite = dm[np.isfinite(dm) & (dm > 0)]
        pos_cart = cart[cart > 0]
        scale = (float(np.mean(finite) / np.mean(pos_cart))
                 if finite.size > 0 and pos_cart.size > 0 else 2.0)

        for i in np.where(fallback)[0]:
            dm[i, :] = cart[i] * scale
            dm[:, i] = cart[:, i] * scale
            dm[i, i] = 0.0

        return dm

    def depot_dists(self, ik_result: IKResult) -> np.ndarray:
        """Depot distances; IK-failed viewpoints get 2× the maximum finite distance."""
        dd = self._depot_dists(ik_result.q_all)
        if np.any(np.isnan(dd)):
            finite_max = float(np.nanmax(dd)) if not np.all(np.isnan(dd)) else 10.0
            dd = np.where(np.isnan(dd), finite_max * 2.0, dd)
        return dd

    # ── IK precomputation ─────────────────────────────────────────────────────

    def precompute_ik(self, mesh_entry: dict, robot_model,
                      planning_group: str = 'disc_to_ur5e') -> tuple:
        """IK for every viewpoint across all regions. Returns (IKResult, region_offsets).

        region_offsets[r] = first global index of region r in IKResult.q_all.
        Rows where IK fails entirely are NaN; cartesian_fallback[i] marks them.
        """
        sing = SingularityHandler(robot_model, planning_group, self.singularity_threshold)
        q_rows, manip_vals, fallback_flags, pos_rows, status = [], [], [], [], {}
        region_offsets = []
        g = 0

        for region in mesh_entry.get('regions', []):
            region_offsets.append(g)
            for cluster in region.get('clusters', []):
                vp = cluster.get('viewpoint')
                if vp is None:
                    q_rows.append(np.full(7, np.nan))
                    manip_vals.append(0.0)
                    fallback_flags.append(True)
                    pos_rows.append(np.zeros(3))
                    status[g] = {'tier': 3}
                    g += 1
                    continue

                pos = vp['position']
                ori = vp['orientation']  # [x, y, z, w]
                pose = Pose()
                pose.position.x, pose.position.y, pose.position.z = pos[0], pos[1], pos[2]
                pose.orientation.x, pose.orientation.y = ori[0], ori[1]
                pose.orientation.z, pose.orientation.w = ori[2], ori[3]

                q, manip, tier = sing.solve_ik(pose)
                q_rows.append(q if q is not None else np.full(7, np.nan))
                manip_vals.append(manip)
                fallback_flags.append(q is None)
                pos_rows.append(np.array(pos, dtype=float))
                status[g] = {'tier': tier}
                g += 1

        return IKResult(
            q_all=np.array(q_rows),
            manip_all=np.array(manip_vals),
            cartesian_fallback=np.array(fallback_flags, dtype=bool),
            positions=np.array(pos_rows),
            status=status,
        ), region_offsets

    def recompute_ik_chained(self, initial_ik: IKResult, region_offsets: list,
                              region_order: list, region_paths: dict,
                              mesh_entry: dict, robot_model,
                              planning_group: str = 'disc_to_ur5e') -> IKResult:
        """Re-solve IK along the VRP path using sequential seeding.

        Seeds the first viewpoint of the first region from DEPOT_Q.
        Each subsequent viewpoint is seeded from the previous viewpoint's IK solution,
        so configs form a continuous chain through joint space rather than clustering
        at the home position. This makes dm values along the path direction much
        smaller, giving the optimizer a strong and accurate traversal-cost signal.

        The q_all array preserves the same index layout as initial_ik (region_offsets
        remain valid). Only viewpoints included in region_paths are recomputed;
        the rest keep their initial_ik values.
        """
        sing = SingularityHandler(robot_model, planning_group, self.singularity_threshold)
        regions = mesh_entry.get('regions', [])

        q_new = initial_ik.q_all.copy()
        manip_new = initial_ik.manip_all.copy()
        fallback_new = initial_ik.cartesian_fallback.copy()

        prev_q = DEPOT_Q.copy()

        for r_idx in region_order:
            path = region_paths.get(r_idx, [])
            if not path:
                continue
            off = region_offsets[r_idx]
            clusters = regions[r_idx].get('clusters', []) if r_idx < len(regions) else []

            for local_i in path:
                vp = clusters[local_i].get('viewpoint') if local_i < len(clusters) else None
                if vp is None:
                    continue

                pos = vp['position']
                ori = vp['orientation']
                pose = Pose()
                pose.position.x, pose.position.y, pose.position.z = pos[0], pos[1], pos[2]
                pose.orientation.x, pose.orientation.y = ori[0], ori[1]
                pose.orientation.z, pose.orientation.w = ori[2], ori[3]

                q, manip, tier = sing.solve_ik_from_seed(pose, prev_q)
                g = off + local_i
                if q is not None:
                    q_new[g] = q
                    manip_new[g] = manip
                    fallback_new[g] = False
                    prev_q = q
                # On IK failure: keep initial solution, do not update prev_q seed

        return IKResult(
            q_all=q_new,
            manip_all=manip_new,
            cartesian_fallback=fallback_new,
            positions=initial_ik.positions,
            status=initial_ik.status,
        )

    def tour_cost_breakdown(self, solution: 'VRPSolution',
                             ik_result: 'IKResult',
                             region_offsets: list) -> tuple:
        """Separate arm and turntable cost along the optimized tour.

        Returns (arm_cost, tt_cost) in weighted-radian units:
          arm_cost = Σ_edges  ||_w[1:] ⊙ dq[1:]||   (6 UR5e joints)
          tt_cost  = Σ_edges  _w[0] * |dq[0]|        (turntable)

        Note: arm_cost² + tt_cost² ≠ total_tour_cost² in general because
        the per-edge values don't add linearly; these are the *summed*
        component contributions, useful for tuning joint_weights.
        """
        q = ik_result.q_all
        w = self._w

        def _edge(i, j):
            qi, qj = q[i], q[j]
            if np.any(np.isnan(qi)) or np.any(np.isnan(qj)):
                return 0.0, 0.0
            dq = qi - qj
            return float(np.linalg.norm(dq[1:] * w[1:])), float(abs(dq[0]) * w[0])

        arm = tt = 0.0
        prev_g = None
        for r_idx in solution.region_order:
            path = solution.region_paths[r_idx]
            off = region_offsets[r_idx]
            first_g = off + path[0]
            if prev_g is not None:
                a, t = _edge(prev_g, first_g)
                arm += a; tt += t
            for k in range(len(path) - 1):
                a, t = _edge(off + path[k], off + path[k + 1])
                arm += a; tt += t
            prev_g = off + path[-1]
        return round(arm, 4), round(tt, 4)

    # ── Algorithm helpers ─────────────────────────────────────────────────────

    def _region_sizes(self, region_offsets: list, N: int) -> list:
        """Number of viewpoints in each region."""
        return [b - a for a, b in zip(region_offsets, list(region_offsets[1:]) + [N])]

    def _greedy_path_from(self, r: int, start: int, region_offsets: list,
                          sizes: list, dm: np.ndarray) -> tuple:
        """Open nearest-neighbor path within region r from local index start."""
        n, off = sizes[r], region_offsets[r]
        if n == 1:
            return [0], 0.0
        visited = [False] * n
        visited[start] = True
        path, cost = [start], 0.0
        for _ in range(n - 1):
            cur_g = off + path[-1]
            best_d, best_j = float('inf'), -1
            for j in range(n):
                d = dm[cur_g, off + j]
                if not visited[j] and d < best_d:
                    best_d, best_j = d, j
            cost += best_d
            visited[best_j] = True
            path.append(best_j)
        return path, cost

    def _greedy_vrp(self, dm: np.ndarray, dd: np.ndarray,
                    region_offsets: list, sizes: list) -> tuple:
        """Greedy region ordering with jointly optimized entry points.

        For each unvisited region, picks the (region, entry viewpoint) pair that
        minimises transition_cost + intra_region_cost at each step.
        """
        R = len(region_offsets)
        cands = {r: {s: self._greedy_path_from(r, s, region_offsets, sizes, dm)
                     for s in range(sizes[r])} for r in range(R)}

        visited = [False] * R
        region_order, region_paths = [], {}
        prev_g = None  # None = at depot

        for _ in range(R):
            best_r = best_s = -1
            best_total = float('inf')
            for r in range(R):
                if visited[r]:
                    continue
                off = region_offsets[r]
                for s, (_, intra) in cands[r].items():
                    trans = dd[off + s] if prev_g is None else dm[prev_g, off + s]
                    if trans + intra < best_total:
                        best_total, best_r, best_s = trans + intra, r, s

            region_order.append(best_r)
            visited[best_r] = True
            region_paths[best_r] = cands[best_r][best_s][0]
            prev_g = region_offsets[best_r] + region_paths[best_r][-1]

        return region_order, region_paths

    # ── Cost function ─────────────────────────────────────────────────────────

    def full_tour_cost(
        self,
        solution: VRPSolution,
        dm: np.ndarray,
        depot_dists: np.ndarray,
        region_offsets: list,
    ) -> tuple:
        """Return (total, intra, inter, depot) cost for a closed VRP tour.

        region_offsets[r] = starting row index of region r in dm / q_all.
        """
        intra = inter = depot = 0.0
        prev_g = None  # None means currently at depot

        for r_idx in solution.region_order:
            path = solution.region_paths[r_idx]
            off = region_offsets[r_idx]
            first_g = off + path[0]

            if prev_g is None:
                depot += depot_dists[first_g]
            else:
                inter += dm[prev_g, first_g]

            for k in range(len(path) - 1):
                intra += dm[off + path[k], off + path[k + 1]]

            prev_g = off + path[-1]

        if prev_g is not None:
            depot += depot_dists[prev_g]

        total = intra + inter + depot
        return total, intra, inter, depot

    def _compute_tour_cost(self, region_order: list, region_paths: dict,
                           dm: np.ndarray, dd: np.ndarray, region_offsets: list) -> float:
        """Lightweight total tour cost for algorithm internals (no VRPSolution overhead)."""
        cost, prev_g = 0.0, None
        for r_idx in region_order:
            path, off = region_paths[r_idx], region_offsets[r_idx]
            first_g = off + path[0]
            cost += dd[first_g] if prev_g is None else dm[prev_g, first_g]
            for k in range(len(path) - 1):
                cost += dm[off + path[k], off + path[k + 1]]
            prev_g = off + path[-1]
        if prev_g is not None:
            cost += dd[prev_g]
        return cost

    def _intra_two_opt(self, path: list, off: int, dm: np.ndarray) -> list:
        """First-improvement 2-opt on an open intra-region path."""
        n = len(path)
        improved = True
        while improved:
            improved = False
            for i in range(n - 1):
                a, b = off + path[i], off + path[i + 1]
                for j in range(i + 2, n):
                    c = off + path[j]
                    delta = dm[a, c] - dm[a, b]
                    if j + 1 < n:
                        d = off + path[j + 1]
                        delta += dm[b, d] - dm[c, d]
                    if delta < -1e-10:
                        path[i + 1:j + 1] = path[i + 1:j + 1][::-1]
                        improved = True
                        break
                if improved:
                    break
        return path

    def _intra_or_opt_1(self, path: list, off: int, dm: np.ndarray) -> list:
        """Relocate individual viewpoints to improve the intra-region open path."""
        n = len(path)
        improved = True
        while improved:
            improved = False
            for i in range(n):
                curr = off + path[i]
                prev_g = off + path[i - 1] if i > 0 else None
                next_g = off + path[i + 1] if i < n - 1 else None
                # Net saving from removing path[i] (removed edges minus bridge added)
                remove_save = (0.0 if prev_g is None else dm[prev_g, curr]) + \
                              (0.0 if next_g is None else dm[curr, next_g]) - \
                              (0.0 if prev_g is None or next_g is None else dm[prev_g, next_g])
                remaining = path[:i] + path[i + 1:]
                nr = len(remaining)
                for j in range(nr + 1):
                    ip = off + remaining[j - 1] if j > 0 else None
                    ix = off + remaining[j] if j < nr else None
                    insert_cost = (dm[ip, curr] + dm[curr, ix] - dm[ip, ix]
                                   if ip is not None and ix is not None
                                   else (dm[ip, curr] if ip is not None else
                                         (dm[curr, ix] if ix is not None else 0.0)))
                    if insert_cost < remove_save - 1e-10:
                        path = remaining[:j] + [path[i]] + remaining[j:]
                        improved = True
                        break
                if improved:
                    break
        return path

    def _inter_region_improve(self, region_order: list, region_paths: dict,
                               dm: np.ndarray, dd: np.ndarray,
                               region_offsets: list) -> tuple:
        """Or-opt at region level: relocate each region to its best position with optional reversal."""
        R = len(region_order)
        best = self._compute_tour_cost(region_order, region_paths, dm, dd, region_offsets)
        improved = True
        while improved:
            improved = False
            for i in range(R):
                r = region_order[i]
                remaining = region_order[:i] + region_order[i + 1:]
                fwd = region_paths[r]
                rev = list(reversed(fwd))
                for j in range(R):
                    for p in (fwd, rev):
                        test_order = remaining[:j] + [r] + remaining[j:]
                        test_paths = dict(region_paths)
                        test_paths[r] = p
                        c = self._compute_tour_cost(test_order, test_paths, dm, dd, region_offsets)
                        if c < best - 1e-10:
                            best = c
                            region_order, region_paths = test_order, test_paths
                            fwd, rev = p, list(reversed(p))
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return region_order, region_paths

    def _intra_local_search(self, path: list, off: int, dm: np.ndarray) -> list:
        """Alternate 2-opt and Or-opt(1) until neither finds an improvement."""
        while True:
            prev = path[:]
            path = self._intra_two_opt(path, off, dm)
            path = self._intra_or_opt_1(path, off, dm)
            if path == prev:
                break
        return path

    def _fixed_start_intra_path(self, r: int, start: int, region_offsets: list,
                                 sizes: list, dm: np.ndarray) -> tuple:
        """Greedy NN from fixed start + local search → (near-optimal open path, intra cost)."""
        path, _ = self._greedy_path_from(r, start, region_offsets, sizes, dm)
        path = self._intra_local_search(path, region_offsets[r], dm)
        off = region_offsets[r]
        cost = sum(dm[off + path[k], off + path[k + 1]] for k in range(len(path) - 1))
        return path, cost

    def _double_bridge_regions(self, region_order: list) -> list:
        """4-opt double-bridge perturbation: A+C+B+D reconnection of the region sequence."""
        R = len(region_order)
        if R < 4:
            return region_order[:]
        a, b, c = sorted(random.sample(range(1, R), 3))
        A, B, C, D = region_order[:a], region_order[a:b], region_order[b:c], region_order[c:]
        return A + C + B + D

    def _vrp_local_search(self, region_order: list, region_paths: dict,
                           dm: np.ndarray, dd: np.ndarray, region_offsets: list) -> tuple:
        """Full local search pass: intra-region LS for every region, then inter-region relocation."""
        for r_idx in region_order:
            region_paths[r_idx] = self._intra_local_search(
                region_paths[r_idx], region_offsets[r_idx], dm)
        return self._inter_region_improve(region_order, region_paths, dm, dd, region_offsets)

    def _aco_construct_ant(self, dm: np.ndarray, dd: np.ndarray, region_offsets: list,
                           sizes: list, tau_inter: np.ndarray, tau_intra: list,
                           alpha: float, beta: float, k_entries: int = 3) -> tuple:
        """Build one ant's VRP solution using pheromone + heuristic probabilities.

        Entry viewpoint for each region is chosen from the K nearest candidates
        (by joint-space distance from the current arm position) using combined
        pheromone (mean outgoing τ_intra) and distance heuristic — so ACO learns
        which entry produces good downstream paths, not just the geometrically
        nearest one.
        """
        R = len(region_offsets)
        visited = [False] * R
        region_order, region_paths = [], {}
        prev_r, prev_g = R, None  # R = depot row index in tau_inter

        for _ in range(R):
            tau_row = tau_inter[prev_r]
            eta = np.zeros(R)
            for r in range(R):
                if not visited[r] and sizes[r] > 0:
                    off = region_offsets[r]
                    min_c = min(dd[off + s] if prev_g is None else dm[prev_g, off + s]
                                for s in range(sizes[r]))
                    eta[r] = 1.0 / (min_c + 1e-10)
            mask = np.array([0.0 if visited[r] or sizes[r] == 0 else 1.0 for r in range(R)])
            attract = (tau_row ** alpha) * (eta ** beta) * mask
            total = attract.sum()
            probs = attract / total if total > 1e-10 else mask / mask.sum()
            next_r = int(np.random.choice(R, p=probs))

            off, n_r = region_offsets[next_r], sizes[next_r]
            tau_r = tau_intra[next_r]

            # K-nearest entry candidates; select among them with pheromone + heuristic.
            # tau_r[s].mean() = mean outgoing pheromone from s, reflecting how often
            # s appeared as a productive starting point in previously good solutions.
            dists_e = dd[off:off + n_r] if prev_g is None else dm[prev_g, off:off + n_r]
            k = min(k_entries, n_r)
            k_cand = np.argsort(dists_e)[:k]
            eta_e = 1.0 / (dists_e[k_cand] + 1e-10)
            tau_e = np.array([tau_r[s].mean() for s in k_cand])
            attract_e = (tau_e ** alpha) * (eta_e ** beta)
            s_e = attract_e.sum()
            probs_e = attract_e / s_e if s_e > 1e-10 else np.ones(k) / k
            entry = int(k_cand[np.random.choice(k, p=probs_e)])

            vp_vis = [False] * n_r
            vp_vis[entry] = True
            path = [entry]

            for _ in range(n_r - 1):
                cur = path[-1]
                eta_vp = np.array([
                    0.0 if vp_vis[j] else 1.0 / (dm[off + cur, off + j] + 1e-10)
                    for j in range(n_r)])
                attract_vp = (tau_r[cur] ** alpha) * (eta_vp ** beta)
                total_vp = attract_vp.sum()
                mask_vp = np.array([0.0 if vp_vis[j] else 1.0 for j in range(n_r)])
                probs_vp = (attract_vp / total_vp if total_vp > 1e-10
                            else mask_vp / mask_vp.sum())
                nxt = int(np.random.choice(n_r, p=probs_vp))
                vp_vis[nxt] = True
                path.append(nxt)

            region_paths[next_r] = path
            visited[next_r] = True
            region_order.append(next_r)
            prev_r = next_r
            prev_g = off + path[-1]

        return region_order, region_paths

    # ── VRP algorithms ────────────────────────────────────────────────────────

    def run_greedy(self, ik_result: IKResult, region_offsets: list) -> VRPSolution:
        """Greedy Clustered VRP: nearest-neighbor intra-region, greedy inter-region ordering."""
        N = len(ik_result.q_all)
        dm = self.build_cost_matrix(ik_result)
        dd = self.depot_dists(ik_result)
        sizes = self._region_sizes(region_offsets, N)
        region_order, region_paths = self._greedy_vrp(dm, dd, region_offsets, sizes)
        sol = VRPSolution(
            region_order=region_order,
            region_paths=region_paths,
            cost=0.0, intra_cost=0.0, inter_cost=0.0, depot_cost=0.0,
            algorithm='vrp_greedy',
        )
        sol.cost, sol.intra_cost, sol.inter_cost, sol.depot_cost = \
            self.full_tour_cost(sol, dm, dd, region_offsets)
        return sol

    def run_ils(self, ik_result: IKResult, region_offsets: list,
                n_restarts: int = 50) -> VRPSolution:  # n_restarts is configurable
        """ILS: greedy init → local search → double-bridge perturbation → local search → keep best."""
        N = len(ik_result.q_all)
        dm = self.build_cost_matrix(ik_result)
        dd = self.depot_dists(ik_result)
        sizes = self._region_sizes(region_offsets, N)

        best_order, best_paths = self._greedy_vrp(dm, dd, region_offsets, sizes)
        best_order, best_paths = self._vrp_local_search(best_order, best_paths, dm, dd, region_offsets)
        best_cost = self._compute_tour_cost(best_order, best_paths, dm, dd, region_offsets)

        for _ in range(n_restarts):
            new_order = self._double_bridge_regions(best_order)
            new_paths = dict(best_paths)
            new_order, new_paths = self._vrp_local_search(new_order, new_paths, dm, dd, region_offsets)
            c = self._compute_tour_cost(new_order, new_paths, dm, dd, region_offsets)
            if c < best_cost - 1e-10:
                best_cost, best_order, best_paths = c, new_order, new_paths

        sol = VRPSolution(
            region_order=best_order, region_paths=best_paths,
            cost=0.0, intra_cost=0.0, inter_cost=0.0, depot_cost=0.0,
            algorithm='vrp_ils',
        )
        sol.cost, sol.intra_cost, sol.inter_cost, sol.depot_cost = \
            self.full_tour_cost(sol, dm, dd, region_offsets)
        return sol

    def run_lkh(self, ik_result: IKResult, region_offsets: list,
                n_restarts: int = 100) -> VRPSolution:  # n_restarts is configurable
        """LKH-style: multi-start ILS — each restart seeds from a fresh random-order greedy init."""
        N = len(ik_result.q_all)
        dm = self.build_cost_matrix(ik_result)
        dd = self.depot_dists(ik_result)
        sizes = self._region_sizes(region_offsets, N)

        R = len(region_offsets)
        best_order = best_paths = None
        best_cost = float('inf')

        for _ in range(n_restarts):
            # Random initial region order; intra-region still uses greedy NN from best entry
            rand_order = list(range(R))
            random.shuffle(rand_order)
            # Build greedy intra-region paths for this random ordering
            paths: dict = {}
            prev_g = None
            for r_idx in rand_order:
                cands = {s: self._greedy_path_from(r_idx, s, region_offsets, sizes, dm)
                         for s in range(sizes[r_idx])}
                best_s = min(cands, key=lambda s: (
                    dd[region_offsets[r_idx] + s] if prev_g is None
                    else dm[prev_g, region_offsets[r_idx] + s]) + cands[s][1])
                paths[r_idx] = cands[best_s][0]
                prev_g = region_offsets[r_idx] + paths[r_idx][-1]

            new_order = rand_order
            new_order, paths = self._vrp_local_search(new_order, paths, dm, dd, region_offsets)
            # ILS perturbation loop from this start
            local_best_order, local_best_paths = new_order[:], dict(paths)
            local_best_cost = self._compute_tour_cost(local_best_order, local_best_paths, dm, dd, region_offsets)
            for _ in range(5):
                p_order = self._double_bridge_regions(local_best_order)
                p_paths = dict(local_best_paths)
                p_order, p_paths = self._vrp_local_search(p_order, p_paths, dm, dd, region_offsets)
                c = self._compute_tour_cost(p_order, p_paths, dm, dd, region_offsets)
                if c < local_best_cost - 1e-10:
                    local_best_cost, local_best_order, local_best_paths = c, p_order, p_paths
            if local_best_cost < best_cost - 1e-10:
                best_cost, best_order, best_paths = local_best_cost, local_best_order, local_best_paths

        sol = VRPSolution(
            region_order=best_order, region_paths=best_paths,
            cost=0.0, intra_cost=0.0, inter_cost=0.0, depot_cost=0.0,
            algorithm='vrp_lkh',
        )
        sol.cost, sol.intra_cost, sol.inter_cost, sol.depot_cost = \
            self.full_tour_cost(sol, dm, dd, region_offsets)
        return sol

    def run_2opt(self, ik_result: IKResult, region_offsets: list) -> VRPSolution:
        """Greedy init + intra-region 2-opt + inter-region relocation."""
        N = len(ik_result.q_all)
        dm = self.build_cost_matrix(ik_result)
        dd = self.depot_dists(ik_result)
        sizes = self._region_sizes(region_offsets, N)
        region_order, region_paths = self._greedy_vrp(dm, dd, region_offsets, sizes)
        for r_idx in region_order:
            region_paths[r_idx] = self._intra_two_opt(region_paths[r_idx], region_offsets[r_idx], dm)
        region_order, region_paths = self._inter_region_improve(region_order, region_paths, dm, dd, region_offsets)
        sol = VRPSolution(
            region_order=region_order, region_paths=region_paths,
            cost=0.0, intra_cost=0.0, inter_cost=0.0, depot_cost=0.0,
            algorithm='vrp_2opt',
        )
        sol.cost, sol.intra_cost, sol.inter_cost, sol.depot_cost = \
            self.full_tour_cost(sol, dm, dd, region_offsets)
        return sol

    def run_3opt(self, ik_result: IKResult, region_offsets: list) -> VRPSolution:
        """Greedy init + intra-region 2-opt & Or-opt(1) + inter-region relocation."""
        N = len(ik_result.q_all)
        dm = self.build_cost_matrix(ik_result)
        dd = self.depot_dists(ik_result)
        sizes = self._region_sizes(region_offsets, N)
        region_order, region_paths = self._greedy_vrp(dm, dd, region_offsets, sizes)
        for r_idx in region_order:
            off = region_offsets[r_idx]
            p = self._intra_two_opt(region_paths[r_idx], off, dm)
            region_paths[r_idx] = self._intra_or_opt_1(p, off, dm)
        region_order, region_paths = self._inter_region_improve(region_order, region_paths, dm, dd, region_offsets)
        sol = VRPSolution(
            region_order=region_order, region_paths=region_paths,
            cost=0.0, intra_cost=0.0, inter_cost=0.0, depot_cost=0.0,
            algorithm='vrp_3opt',
        )
        sol.cost, sol.intra_cost, sol.inter_cost, sol.depot_cost = \
            self.full_tour_cost(sol, dm, dd, region_offsets)
        return sol

    def run_aco(self, ik_result: IKResult, region_offsets: list,
                n_ants: int = 20, n_iter: int = 100,    # configurable
                alpha: float = 1.0, beta: float = 2.0,  # pheromone / heuristic weights (configurable)
                rho: float = 0.1,                        # evaporation rate (configurable)
                k_entries: int = 3) -> VRPSolution:      # K nearest entry candidates per region
        """MMAS ACO with two-level pheromone (τ_inter region, τ_intra per-region).

        Each ant selects entry viewpoints from the K nearest candidates (joint-space)
        using pheromone + distance heuristic, then applies full 2-opt + Or-opt local
        search per region — giving near-optimal intra-region paths while ACO globally
        optimises the inter-region sequence and entry choices across iterations.
        GPU acceleration applies to the distance matrix via joint_dist_matrix.
        """
        N = len(ik_result.q_all)
        dm = self.build_cost_matrix(ik_result)
        dd = self.depot_dists(ik_result)
        sizes = self._region_sizes(region_offsets, N)
        R = len(region_offsets)

        # MMAS init: τ_max from greedy cost so pheromone starts at the upper bound
        best_order, best_paths = self._greedy_vrp(dm, dd, region_offsets, sizes)
        best_cost = self._compute_tour_cost(best_order, best_paths, dm, dd, region_offsets)
        tau_init = 1.0 / (rho * best_cost)
        tau_inter = np.full((R + 1, R), tau_init)   # row R = depot transitions
        tau_intra = [np.full((sizes[r], sizes[r]), tau_init) for r in range(R)]

        for _ in range(n_iter):
            iter_cost = float('inf')
            iter_order = iter_paths = None
            for _ant in range(n_ants):
                order, paths = self._aco_construct_ant(
                    dm, dd, region_offsets, sizes, tau_inter, tau_intra, alpha, beta, k_entries)
                for r_idx in order:
                    paths[r_idx] = self._intra_local_search(paths[r_idx], region_offsets[r_idx], dm)
                # Inter-region relocation after intra local search (same as 2opt/3opt/ILS)
                order, paths = self._inter_region_improve(order, paths, dm, dd, region_offsets)
                c = self._compute_tour_cost(order, paths, dm, dd, region_offsets)
                if c < iter_cost:
                    iter_cost, iter_order, iter_paths = c, order[:], dict(paths)
                if c < best_cost - 1e-10:
                    best_cost, best_order, best_paths = c, order[:], dict(paths)

            # Evaporate → deposit iteration best → clip [tau_min, tau_max]
            tau_inter *= (1 - rho)
            for r in range(R):
                tau_intra[r] *= (1 - rho)

            delta = 1.0 / (iter_cost + 1e-10)
            tau_max = 1.0 / (rho * best_cost)
            # tau_min per matrix: inter uses R cities, intra uses n_r viewpoints
            tau_min_inter = tau_max / (R * 2)

            prev_r = R
            for r_idx in iter_order:
                tau_inter[prev_r, r_idx] += delta
                prev_r = r_idx
            for r_idx in iter_order:
                path = iter_paths[r_idx]
                for k in range(len(path) - 1):
                    tau_intra[r_idx][path[k], path[k + 1]] += delta

            np.clip(tau_inter, tau_min_inter, tau_max, out=tau_inter)
            for r in range(R):
                # intra pheromone needs tighter floor: n_r viewpoints, not R regions
                tau_min_intra = tau_max / (sizes[r] * 2)
                np.clip(tau_intra[r], tau_min_intra, tau_max, out=tau_intra[r])

        # Final polish: one full local search pass on the global best
        best_order, best_paths = self._vrp_local_search(best_order, best_paths, dm, dd, region_offsets)
        best_cost = self._compute_tour_cost(best_order, best_paths, dm, dd, region_offsets)

        sol = VRPSolution(
            region_order=best_order, region_paths=best_paths,
            cost=0.0, intra_cost=0.0, inter_cost=0.0, depot_cost=0.0,
            algorithm='vrp_aco',
        )
        sol.cost, sol.intra_cost, sol.inter_cost, sol.depot_cost = \
            self.full_tour_cost(sol, dm, dd, region_offsets)
        return sol

    def run_hierarchical(self, ik_result: IKResult, region_offsets: list,
                         k_entries: int = 3) -> VRPSolution:
        """Hierarchical Sequential VRP.

        Phase 1 (per region): given the arm's current joint config at the exit
        of the previous region (or depot at the start), find the K nearest
        viewpoints in each unvisited region and compute a near-optimal open
        intra-region path from each candidate entry (greedy NN + 2-opt/Or-opt).

        Phase 2 (global): greedily pick the (region, entry) pair that minimises
        transition_cost + intra_cost at each step; the chosen region's near-optimal
        path becomes the fixed plan for that region and its exit config seeds the
        next step.

        This guarantees every region's intra-path is near locally optimal AND the
        inter-region sequence is chosen to minimise the total joint-space cost.
        k_entries controls how many candidate entries are tried per region per step.
        """
        N = len(ik_result.q_all)
        dm = self.build_cost_matrix(ik_result)
        dd = self.depot_dists(ik_result)
        sizes = self._region_sizes(region_offsets, N)
        R = len(region_offsets)

        unvisited = list(range(R))
        region_order, region_paths = [], {}
        prev_g = None  # None = currently at depot

        while unvisited:
            best_r = best_path = None
            best_score = float('inf')

            for r in unvisited:
                off, n_r = region_offsets[r], sizes[r]
                if n_r == 0:
                    continue
                # Distances from current position (depot or last exit) to each
                # viewpoint in region r.
                dists = dd[off:off + n_r] if prev_g is None else dm[prev_g, off:off + n_r]
                for entry in np.argsort(dists)[:min(k_entries, n_r)]:
                    path, intra = self._fixed_start_intra_path(
                        r, int(entry), region_offsets, sizes, dm)
                    score = float(dists[entry]) + intra
                    if score < best_score:
                        best_score, best_r, best_path = score, r, path

            region_order.append(best_r)
            region_paths[best_r] = best_path
            prev_g = region_offsets[best_r] + best_path[-1]
            unvisited.remove(best_r)

        sol = VRPSolution(
            region_order=region_order, region_paths=region_paths,
            cost=0.0, intra_cost=0.0, inter_cost=0.0, depot_cost=0.0,
            algorithm='vrp_hierarchical',
        )
        sol.cost, sol.intra_cost, sol.inter_cost, sol.depot_cost = \
            self.full_tour_cost(sol, dm, dd, region_offsets)
        return sol
