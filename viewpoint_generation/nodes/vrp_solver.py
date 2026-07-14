#!/usr/bin/env python3
import random
import itertools
import multiprocessing as _mp
import numpy as np
from dataclasses import dataclass, field

# ROS/MoveIt are only needed for IK precomputation. Guard the imports so the
# pure solver (cost model + combinatorial algorithms) can be imported and
# unit-tested without a ROS environment.
try:
    from moveit.core.robot_state import RobotState
    from geometry_msgs.msg import Pose
except ImportError:
    RobotState = None
    Pose = None

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
    q_all: np.ndarray               # (N, 7) joint configs; NaN rows where IK failed
    manip_all: np.ndarray           # (N,) Yoshikawa manipulability index
    cartesian_fallback: np.ndarray  # (N,) bool — True where Cartesian proxy is used
    positions: np.ndarray           # (N, 3) viewpoint positions in object_frame
    status: dict = field(default_factory=dict)  # global_idx → {'tier': int}
    q_configs: list = None          # optional: per-viewpoint list of (m_i, 7) candidate configs
                                    # (self-motion manifold / turntable sweep); None → use q_all


@dataclass
class VRPSolution:
    region_order: list    # inter-region visit sequence (region indices)
    region_paths: dict    # region_paths[r] = local cluster visit order for region r
    cost: float
    intra_cost: float     # sum of within-region edge costs
    inter_cost: float     # sum of region-to-region transition costs
    depot_cost: float     # depot→first_vp + last_vp→depot
    algorithm: str = ''
    candidates: list = field(default_factory=list)  # (order, paths, cost) for top-K validation
    region_configs: dict = None  # region_configs[r][k] = chosen config idx for region_paths[r][k]


class SingularityHandler:
    # Turntable offsets (rad) tried as alternative IK seeds to escape near-singular configs
    _SEED_OFFSETS = [np.pi/12, -np.pi/12, np.pi/6, -np.pi/6,
                     np.pi/4, -np.pi/4, np.pi/2, -np.pi/2]

    def __init__(self, robot_model, planning_group='disc_to_ur5e', threshold=0.05):
        self._model = robot_model
        self._group = planning_group
        self.threshold = threshold  # Yoshikawa index below this is near-singular

    def manipulability(self, robot_state) -> float:
        """Yoshikawa index sqrt(det(J @ J.T)) of the (6, 7) geometric Jacobian."""
        J = robot_state.get_jacobian(self._group, np.zeros((3, 1)))
        return float(np.sqrt(max(0.0, np.linalg.det(J @ J.T))))

    def solve_ik_from_seed(self, pose, seed_q: np.ndarray,
                           tip_link: str = 'eoat_camera_link') -> tuple:
        """IK seeded from seed_q (tier 0); falls through to the standard 3-tier on failure."""
        state = RobotState(self._model)
        state.set_joint_group_positions(self._group, seed_q)
        if state.set_from_ik(self._group, pose, tip_link):
            state.update()
            manip = self.manipulability(state)
            if manip >= self.threshold:
                return np.array(state.get_joint_group_positions(self._group)), manip, 0
        return self.solve_ik(pose, tip_link)

    def solve_ik(self, pose, tip_link: str = 'eoat_camera_link') -> tuple:
        """3-tier IK with singularity avoidance. Returns (q, manip, tier) or (None, 0.0, 3)."""
        state = RobotState(self._model)

        # Tier 1: direct IK seeded from DEPOT_Q
        state.set_joint_group_positions(self._group, DEPOT_Q)
        if state.set_from_ik(self._group, pose, tip_link):
            state.update()
            manip = self.manipulability(state)
            if manip >= self.threshold:
                return np.array(state.get_joint_group_positions(self._group)), manip, 1

        # Tier 2: perturb the turntable seed, keep the highest-manipulability result
        best_q, best_manip = None, 0.0
        for offset in self._SEED_OFFSETS:
            seed = DEPOT_Q.copy()
            seed[0] += offset
            state.set_joint_group_positions(self._group, seed)
            if state.set_from_ik(self._group, pose, tip_link):
                state.update()
                manip = self.manipulability(state)
                if manip > best_manip:
                    best_manip, best_q = manip, np.array(
                        state.get_joint_group_positions(self._group))

        if best_q is not None:
            return best_q, best_manip, 2
        return None, 0.0, 3  # Cartesian proxy handled in precompute_ik

    @staticmethod
    def _turntable_seeds(n_samples: int, base: np.ndarray = DEPOT_Q,
                         lo: float = -np.pi, hi: float = np.pi) -> list:
        """IK seeds sweeping the redundant turntable DOF (joint 0) over [lo, hi),
        plus the home seed. Sampling the redundancy is what exposes the multiple
        feasible arm configurations of a single camera pose."""
        seeds = [base.copy()]
        for theta in np.linspace(lo, hi, n_samples, endpoint=False):
            s = base.copy()
            s[0] = theta
            seeds.append(s)
        return seeds

    @staticmethod
    def _dedup_by_config(found: list, dedup_tol: float, max_configs: int) -> list:
        """Keep the highest-manipulability configs, dropping any within dedup_tol
        (joint-space L∞) of an already-kept one. found = [(q, manip), ...]."""
        kept = []
        for q, mp in sorted(found, key=lambda cm: -cm[1]):
            if all(np.max(np.abs(q - kq)) >= dedup_tol for kq, _ in kept):
                kept.append((q, mp))
            if len(kept) >= max_configs:
                break
        return kept

    def sweep_configs(self, pose, tip_link: str = 'eoat_camera_link',
                      n_samples: int = 12, max_configs: int = 8,
                      dedup_tol: float = 0.1) -> list:
        """Turntable-sweep reachability: solve IK from a grid of turntable seeds and
        return the distinct feasible configs as [(q, manip), ...], best manip first.

        This samples the 1-DOF self-motion manifold (turntable) so a pose reachable
        only through a narrow turntable window or a specific arm branch is still found
        — the weight-independent, near-global reachability test. Configs with
        manip ≥ threshold are preferred; if only near-singular solutions exist the
        single best is kept (reachable-but-singular) so the viewpoint is not dropped.
        Returns [] only when no seed yields any IK solution (truly unreachable)."""
        state = RobotState(self._model)
        raw = []
        for seed in self._turntable_seeds(n_samples):
            state.set_joint_group_positions(self._group, seed)
            if state.set_from_ik(self._group, pose, tip_link):
                state.update()
                q = np.array(state.get_joint_group_positions(self._group))
                raw.append((q, self.manipulability(state)))
        if not raw:
            return []
        good = [(q, m) for q, m in raw if m >= self.threshold]
        if good:
            return self._dedup_by_config(good, dedup_tol, max_configs)
        return self._dedup_by_config(raw, dedup_tol, 1)  # near-singular fallback


class VRPSolver:
    """Clustered VRP solver over robot joint space.

    Edge cost is either an execution-time surrogate (default) or a weighted
    joint-space distance:

      cost_mode='time'  — per-joint point-to-point duration under a
                          trapezoidal/triangular velocity profile; the slowest
                          joint governs the segment, so edge_cost = max_j t_j.
                          Approximates MoveIt TOTG time (units: seconds).
      cost_mode='joint' — sqrt(Σ w_j · Δq_j²), weighted Euclidean (units: rad).
    """

    DEFAULT_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    def __init__(
        self,
        joint_weights=None,
        cost_mode='time',
        max_velocity=0.5,        # rad/s, scalar or length-7
        max_acceleration=1.0,    # rad/s², scalar or length-7
        use_gpu=False,
        singularity_threshold=0.05,
        singularity_action='avoid',
        n_turntable_samples=0,       # >0 enables the multi-config turntable sweep
        max_configs_per_vp=8,        # cap on candidate IK configs kept per viewpoint
        config_dedup_tol=0.1,        # rad, L∞ tolerance for treating configs as identical
        seed=None,                   # RNG seed for stochastic algorithms (ACO); None → nondeterministic
        logger=None,
    ):
        if joint_weights is None or len(joint_weights) == 0:
            joint_weights = self.DEFAULT_WEIGHTS
        self.joint_weights = np.array(joint_weights, dtype=float)
        self._w = np.sqrt(self.joint_weights)  # sqrt so ‖√w⊙Δq‖ == weighted L2
        self.cost_mode = cost_mode
        self._vmax = np.broadcast_to(np.asarray(max_velocity, float), (7,)).astype(float)
        self._amax = np.broadcast_to(np.asarray(max_acceleration, float), (7,)).astype(float)
        self._d_crit = self._vmax ** 2 / self._amax  # Δq at which the profile switches tri→trap
        self.use_gpu = use_gpu and _CUPY
        self.singularity_threshold = singularity_threshold
        self.singularity_action = singularity_action
        self.n_turntable_samples = n_turntable_samples
        self.max_configs_per_vp = max_configs_per_vp
        self.config_dedup_tol = config_dedup_tol
        self.seed = seed
        self.logger = logger
        self._cand_cache = {}   # (off, n) → per-region LK candidate sets (per solve)

    @property
    def cost_units(self) -> str:
        return 's' if self.cost_mode == 'time' else 'rad'

    # ── Distance primitives ───────────────────────────────────────────────────

    def _per_joint_time(self, ad, xp=np):
        """Per-joint point-to-point duration for |Δq| = ad (..., 7)."""
        vmax, amax = xp.asarray(self._vmax), xp.asarray(self._amax)
        dcrit = xp.asarray(self._d_crit)
        tri = 2.0 * xp.sqrt(ad / amax)          # short move: accelerate then decelerate
        trap = ad / vmax + vmax / amax           # long move: reaches cruise velocity
        return xp.where(ad <= dcrit, tri, trap)

    def _config_cost(self, dq, xp=np):
        """Scalar edge cost for one or many joint-space moves dq (..., 7) → (...)."""
        if self.cost_mode == 'time':
            return self._per_joint_time(xp.abs(dq), xp).max(axis=-1)
        w = xp.asarray(self._w)
        return xp.sqrt(xp.sum((dq * w) ** 2, axis=-1))

    def _pairwise_cost_matrix(self, q_all: np.ndarray) -> np.ndarray:
        """N×N edge-cost matrix; GPU-accelerated when CuPy is available."""
        if self.use_gpu:
            q = cp.asarray(q_all)
            dq = q[:, None, :] - q[None, :, :]
            return cp.asnumpy(self._config_cost(dq, cp))
        dq = q_all[:, None, :] - q_all[None, :, :]
        return self._config_cost(dq, np)

    def build_cost_matrix(self, ik_result: IKResult) -> np.ndarray:
        """Edge-cost matrix; Cartesian proxy fills rows/cols for IK-failed viewpoints."""
        dm = self._pairwise_cost_matrix(ik_result.q_all)
        fallback = ik_result.cartesian_fallback
        if not np.any(fallback):
            return dm

        cart = np.linalg.norm(
            ik_result.positions[:, None, :] - ik_result.positions[None, :, :], axis=-1)
        finite = dm[np.isfinite(dm) & (dm > 0)]
        pos_cart = cart[cart > 0]
        scale = (float(np.mean(finite) / np.mean(pos_cart))
                 if finite.size > 0 and pos_cart.size > 0 else 2.0)

        penalty = 2.0 * float(np.max(finite)) if finite.size > 0 else 1.0
        for i in np.where(fallback)[0]:
            dm[i, :] = cart[i] * scale
            dm[:, i] = cart[:, i] * scale
            dm[i, i] = 0.0
        dm[~np.isfinite(dm)] = penalty
        np.fill_diagonal(dm, 0.0)
        return dm

    def depot_dists(self, ik_result: IKResult) -> np.ndarray:
        """Cost from DEPOT_Q to each viewpoint; IK-failed ones get 2× the max finite cost."""
        dd = self._config_cost(ik_result.q_all - DEPOT_Q)
        if np.any(np.isnan(dd)):
            finite_max = float(np.nanmax(dd)) if not np.all(np.isnan(dd)) else 10.0
            dd = np.where(np.isnan(dd), finite_max * 2.0, dd)
        return dd

    # ── IK precomputation ─────────────────────────────────────────────────────

    def precompute_ik(self, mesh_entry: dict, robot_model,
                      planning_group: str = 'disc_to_ur5e') -> tuple:
        """IK for every viewpoint across all regions. Returns (IKResult, region_offsets).

        region_offsets[r] is the first global index of region r in IKResult.q_all.
        Rows where IK fails entirely are NaN and flagged in cartesian_fallback.

        When n_turntable_samples > 0, each viewpoint is solved by the turntable sweep,
        which yields multiple feasible IK configs; these populate IKResult.q_configs
        (enabling config-aware P_r) while q_all keeps the best-manipulability config.
        """
        sing = SingularityHandler(robot_model, planning_group, self.singularity_threshold)
        q_rows, manip_vals, fallback_flags, pos_rows, status = [], [], [], [], {}
        sweep = self.n_turntable_samples > 0
        cfg_rows = [] if sweep else None
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
                    pos_rows.append(np.full(3, np.nan))
                    status[g] = {'tier': 3}
                    if sweep:
                        cfg_rows.append(np.zeros((0, 7)))
                    g += 1
                    continue

                if sweep:
                    cms = sing.sweep_configs(
                        self._pose_of(vp), n_samples=self.n_turntable_samples,
                        max_configs=self.max_configs_per_vp, dedup_tol=self.config_dedup_tol)
                    if cms:
                        q, manip = cms[0]
                        cfg_rows.append(np.array([c for c, _ in cms]))
                        status[g] = {'tier': 1, 'n_configs': len(cms)}
                    else:
                        q, manip = None, 0.0
                        cfg_rows.append(np.zeros((0, 7)))
                        status[g] = {'tier': 3}
                else:
                    q, manip, tier = sing.solve_ik(self._pose_of(vp))
                    status[g] = {'tier': tier}

                q_rows.append(q if q is not None else np.full(7, np.nan))
                manip_vals.append(manip)
                fallback_flags.append(q is None)
                pos_rows.append(np.array(vp['position'], dtype=float))
                g += 1

        if sweep and self.logger is not None:
            counts = [len(c) for c in cfg_rows]
            multi = sum(1 for n in counts if n > 1)
            unreach = sum(1 for n in counts if n == 0)
            mean_c = (sum(counts) / len(counts)) if counts else 0.0
            self.logger.info(
                f'turntable sweep: {len(counts)} viewpoints, mean {mean_c:.1f} configs/vp, '
                f'{multi} multi-config, {unreach} unreachable')

        return IKResult(
            q_all=np.array(q_rows),
            manip_all=np.array(manip_vals),
            cartesian_fallback=np.array(fallback_flags, dtype=bool),
            positions=np.array(pos_rows),
            status=status,
            q_configs=cfg_rows,
        ), region_offsets

    def recompute_ik_chained(self, initial_ik: IKResult, region_offsets: list,
                             region_order: list, region_paths: dict,
                             mesh_entry: dict, robot_model,
                             planning_group: str = 'disc_to_ur5e') -> IKResult:
        """Re-solve IK along the VRP path with sequential seeding.

        The first viewpoint is seeded from DEPOT_Q; each subsequent one is seeded
        from the previous viewpoint's solution, so configs form a continuous chain
        through joint space instead of clustering near home. This makes edge costs
        along the path realistic and gives the optimizer an accurate signal. The
        index layout (region_offsets) is preserved; only viewpoints in region_paths
        are recomputed.
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
                q, manip, tier = sing.solve_ik_from_seed(self._pose_of(vp), prev_q)
                if q is not None:
                    g = off + local_i
                    q_new[g] = q
                    manip_new[g] = manip
                    fallback_new[g] = False
                    prev_q = q
                # On IK failure keep the initial solution and do not advance the seed.

        return IKResult(
            q_all=q_new, manip_all=manip_new, cartesian_fallback=fallback_new,
            positions=initial_ik.positions, status=initial_ik.status,
        )

    @staticmethod
    def _pose_of(vp):
        """Build a geometry_msgs Pose from a viewpoint dict."""
        pos, ori = vp['position'], vp['orientation']  # ori = [x, y, z, w]
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = pos[0], pos[1], pos[2]
        pose.orientation.x, pose.orientation.y = ori[0], ori[1]
        pose.orientation.z, pose.orientation.w = ori[2], ori[3]
        return pose

    def tour_cost_breakdown(self, solution: VRPSolution, ik_result: IKResult,
                            region_offsets: list) -> tuple:
        """Per-edge arm vs turntable cost contribution summed over the tour.

        time mode: arm = Σ max_j∈arm t_j, tt = Σ t_turntable (isolated-joint times).
        joint mode: arm = Σ‖w[1:]⊙Δq[1:]‖, tt = Σ w[0]·|Δq[0]|.
        The two do not sum to the total (segment cost is a max, not a sum); they are
        a diagnostic for how much motion each mechanism contributes.
        """
        q = ik_result.q_all

        def _edge(i, j):
            qi, qj = q[i], q[j]
            if np.any(np.isnan(qi)) or np.any(np.isnan(qj)):
                return 0.0, 0.0
            dq = qi - qj
            if self.cost_mode == 'time':
                tj = self._per_joint_time(np.abs(dq))
                return float(tj[1:].max()), float(tj[0])
            return float(np.linalg.norm(dq[1:] * self._w[1:])), float(abs(dq[0]) * self._w[0])

        arm = tt = 0.0
        prev_g = None
        for r_idx in solution.region_order:
            path = solution.region_paths[r_idx]
            off = region_offsets[r_idx]
            if prev_g is not None:
                a, t = _edge(prev_g, off + path[0])
                arm += a; tt += t
            for k in range(len(path) - 1):
                a, t = _edge(off + path[k], off + path[k + 1])
                arm += a; tt += t
            prev_g = off + path[-1]
        return round(arm, 4), round(tt, 4)

    # ── Algorithm helpers ─────────────────────────────────────────────────────

    def _region_sizes(self, region_offsets: list, N: int) -> list:
        return [b - a for a, b in zip(region_offsets, list(region_offsets[1:]) + [N])]

    def _setup(self, ik_result: IKResult, region_offsets: list) -> tuple:
        """Shared preamble for every run_* entry point."""
        N = len(ik_result.q_all)
        dm = self.build_cost_matrix(ik_result)
        dd = self.depot_dists(ik_result)
        sizes = self._region_sizes(region_offsets, N)
        self._cand_cache = {}   # dm is rebuilt here; candidate sets are per solve
        return dm, dd, sizes, len(region_offsets)

    def _finalize(self, order: list, paths: dict, dm, dd, region_offsets: list,
                  algorithm: str, candidates: list = None) -> VRPSolution:
        """Build a VRPSolution and fill its cost breakdown."""
        sol = VRPSolution(
            region_order=order, region_paths=paths,
            cost=0.0, intra_cost=0.0, inter_cost=0.0, depot_cost=0.0,
            algorithm=algorithm, candidates=candidates or [],
        )
        sol.cost, sol.intra_cost, sol.inter_cost, sol.depot_cost = \
            self.full_tour_cost(sol, dm, dd, region_offsets)
        return sol

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

    def _greedy_vrp(self, dm, dd, region_offsets, sizes) -> tuple:
        """Greedy region ordering with jointly optimized entry points.

        At each step picks the (region, entry) pair minimising transition + intra cost.
        """
        active = [r for r in range(len(region_offsets)) if sizes[r] > 0]
        cands = {r: {s: self._greedy_path_from(r, s, region_offsets, sizes, dm)
                     for s in range(sizes[r])} for r in active}

        visited = set()
        region_order, region_paths = [], {}
        prev_g = None

        for _ in range(len(active)):
            best_r = best_s = -1
            best_total = float('inf')
            for r in active:
                if r in visited:
                    continue
                off = region_offsets[r]
                for s, (_, intra) in cands[r].items():
                    trans = dd[off + s] if prev_g is None else dm[prev_g, off + s]
                    if trans + intra < best_total:
                        best_total, best_r, best_s = trans + intra, r, s
            region_order.append(best_r)
            visited.add(best_r)
            region_paths[best_r] = cands[best_r][best_s][0]
            prev_g = region_offsets[best_r] + region_paths[best_r][-1]

        return region_order, region_paths

    # ── Cost functions ────────────────────────────────────────────────────────

    def full_tour_cost(self, solution: VRPSolution, dm, dd, region_offsets) -> tuple:
        """(total, intra, inter, depot) cost for the closed VRP tour."""
        intra = inter = depot = 0.0
        prev_g = None
        for r_idx in solution.region_order:
            path = solution.region_paths[r_idx]
            off = region_offsets[r_idx]
            if prev_g is None:
                depot += dd[off + path[0]]
            else:
                inter += dm[prev_g, off + path[0]]
            for k in range(len(path) - 1):
                intra += dm[off + path[k], off + path[k + 1]]
            prev_g = off + path[-1]
        if prev_g is not None:
            depot += dd[prev_g]
        return intra + inter + depot, intra, inter, depot

    def _compute_tour_cost(self, region_order, region_paths, dm, dd, region_offsets) -> float:
        """Total closed-tour cost (no VRPSolution overhead) for algorithm internals."""
        cost, prev_g = 0.0, None
        for r_idx in region_order:
            path, off = region_paths[r_idx], region_offsets[r_idx]
            cost += dd[off + path[0]] if prev_g is None else dm[prev_g, off + path[0]]
            for k in range(len(path) - 1):
                cost += dm[off + path[k], off + path[k + 1]]
            prev_g = off + path[-1]
        if prev_g is not None:
            cost += dd[prev_g]
        return cost

    # ── Local search ──────────────────────────────────────────────────────────

    def _intra_two_opt(self, path: list, off: int, dm: np.ndarray,
                       lock_end: bool = False) -> list:
        """First-improvement 2-opt on an open intra-region path.

        The start is always held fixed; lock_end also holds the last node, giving
        a fixed-endpoint 2-opt for P_r(e,x) paths."""
        n = len(path)
        jmax = n - 1 if lock_end else n
        improved = True
        while improved:
            improved = False
            for i in range(n - 1):
                a, b = off + path[i], off + path[i + 1]
                for j in range(i + 2, jmax):
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

    def _intra_or_opt(self, path: list, off: int, dm: np.ndarray, seg_len: int = 1,
                      lock_end: bool = False) -> list:
        """Relocate a run of seg_len viewpoints to its best position (open path).

        seg_len 1 is node relocation; 2 and 3 move short chains. Together these
        capture most of the useful moves a full 3-opt would add, at O(n²) cost.
        lock_end restricts moves to interior segments/positions, keeping both
        endpoints fixed (for P_r(e,x) paths).
        """
        improved = True
        while improved:
            improved = False
            n = len(path)
            i_lo, i_hi = (1, n - seg_len) if lock_end else (0, n - seg_len + 1)
            for i in range(i_lo, i_hi):
                seg = path[i:i + seg_len]
                g0, g1 = off + seg[0], off + seg[-1]
                prev_g = off + path[i - 1] if i > 0 else None
                next_g = off + path[i + seg_len] if i + seg_len < n else None
                remove_save = (0.0 if prev_g is None else dm[prev_g, g0]) + \
                              (0.0 if next_g is None else dm[g1, next_g]) - \
                              (0.0 if prev_g is None or next_g is None else dm[prev_g, next_g])
                remaining = path[:i] + path[i + seg_len:]
                nr = len(remaining)
                j_lo, j_hi = (1, nr) if lock_end else (0, nr + 1)
                for j in range(j_lo, j_hi):
                    ip = off + remaining[j - 1] if j > 0 else None
                    ix = off + remaining[j] if j < nr else None
                    insert_cost = (dm[ip, g0] + dm[g1, ix] - dm[ip, ix]
                                   if ip is not None and ix is not None
                                   else (dm[ip, g0] if ip is not None else
                                         (dm[g1, ix] if ix is not None else 0.0)))
                    if insert_cost < remove_save - 1e-10:
                        path = remaining[:j] + seg + remaining[j:]
                        improved = True
                        break
                if improved:
                    break
        return path

    def _region_candidates(self, off: int, n: int, dm: np.ndarray, k: int = 8) -> dict:
        """Cached k-nearest candidate sets (local indices) for region [off, off+n),
        used to restrict Lin-Kernighan's search to promising neighbours."""
        key = (off, n)
        cached = self._cand_cache.get(key)
        if cached is not None:
            return cached
        k = min(k, n - 1)
        cand = {}
        for i in range(n):
            nbrs = sorted((j for j in range(n) if j != i),
                          key=lambda j: dm[off + i, off + j])
            cand[i] = set(nbrs[:k])
        self._cand_cache[key] = cand
        return cand

    def _intra_lk(self, path: list, off: int, dm: np.ndarray, candidates: dict,
                  lock_end: bool = False) -> list:
        """Candidate-restricted Lin-Kernighan (depth-2 sequential moves) on an open
        intra path, ported from tsp_solver._lk_local_search with an offset and an
        optional fixed end.

        The first node is inherently fixed — every sequential move keeps the prefix
        t[:i+1] — so this is naturally a fixed-start path optimiser; lock_end also
        holds the last node (for P_r(e,x)) by keeping j,k off the final index. The
        depth-2 move S1+S2_rev+S3_rev+S4 is the sequential 3-opt that no single 2-opt
        or Or-opt can produce, and each applied move has strictly positive gain, so LK
        never worsens the input path."""
        t = list(path)
        n = len(t)
        end = n - 1 if lock_end else n
        improved = True
        while improved:
            improved = False
            for i in range(n - 2):
                if improved:
                    break
                t_i, t_i1 = t[i], t[i + 1]
                d1 = dm[off + t_i, off + t_i1]
                for j in range(i + 2, end):
                    if improved:
                        break
                    t_j = t[j]
                    if t_j not in candidates[t_i]:
                        continue
                    g1 = d1 - dm[off + t_i, off + t_j]        # sequential gain, depth 1
                    if g1 <= 0:
                        continue
                    t_j1 = t[j + 1] if j + 1 < n else None
                    d2_rem = dm[off + t_j, off + t_j1] if t_j1 is not None else 0.0
                    d2_add = dm[off + t_i1, off + t_j1] if t_j1 is not None else 0.0
                    if g1 + d2_rem - d2_add > 1e-10:          # depth-1 close (2-opt)
                        t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
                        improved = True
                        break
                    if t_j1 is None:
                        continue
                    G2 = g1 + dm[off + t_j, off + t_j1]
                    for k in range(j + 2, end):
                        t_k = t[k]
                        if t_k not in candidates[t_i1]:
                            continue
                        g2 = G2 - dm[off + t_i1, off + t_k]   # sequential gain, depth 2
                        if g2 <= 0:
                            continue
                        t_k1 = t[k + 1] if k + 1 < n else None
                        d3_rem = dm[off + t_k, off + t_k1] if t_k1 is not None else 0.0
                        d3_add = dm[off + t_j1, off + t_k1] if t_k1 is not None else 0.0
                        if g2 + d3_rem - d3_add > 1e-10:      # depth-2 close
                            t = (t[:i + 1] + t[i + 1:j + 1][::-1]
                                 + t[j + 1:k + 1][::-1] + t[k + 1:])
                            n = len(t)
                            improved = True
                            break
        return t

    def _intra_local_search(self, path: list, off: int, dm: np.ndarray,
                            strong: bool = False) -> list:
        """2-opt + Or-opt(1,2,3) to a joint fixpoint.

        When strong=True, candidate-restricted Lin-Kernighan (depth-2 sequential
        moves) is layered on top — this is what makes run_lkh a genuine Lin-Kernighan
        solver, distinct from the 2-opt/Or-opt-based ILS/ACO/hierarchical. The inner
        loop reaches the 2-opt+Or-opt optimum first, so the descent always passes
        through it before LK, and LK only adds strictly-improving moves; the strong
        result is therefore never worse than 2-opt+Or-opt alone."""
        n = len(path)
        cand = self._region_candidates(off, n, dm) if (strong and n >= 4) else None
        while True:
            prev = path[:]
            while True:                       # 2-opt + Or-opt to joint fixpoint
                base = path[:]
                path = self._intra_two_opt(path, off, dm)
                for seg_len in (1, 2, 3):
                    path = self._intra_or_opt(path, off, dm, seg_len)
                if path == base:
                    break
            if cand is None:
                break
            path = self._intra_lk(path, off, dm, cand)
            if path == prev:
                break
        return path

    def _boundary_two_opt(self, path: list, off: int, dm, cin, cout) -> list:
        """2-opt on an open path whose endpoints connect to fixed anchors.

        cin(g)/cout(g) return the cost of joining global index g to the region's
        predecessor exit / successor entry (or the depot). Reversing a segment that
        touches an endpoint changes that connection cost, so unlike the isolated
        intra 2-opt this couples the path to its neighbours — the missing degree of
        freedom that leaves a residual gap after independent intra optimisation.
        """
        n = len(path)
        improved = True
        while improved:
            improved = False
            for i in range(n):
                left_old = cin(off + path[i]) if i == 0 else dm[off + path[i - 1], off + path[i]]
                for j in range(i + 1, n):
                    right_old = (cout(off + path[j]) if j == n - 1
                                 else dm[off + path[j], off + path[j + 1]])
                    left_new = (cin(off + path[j]) if i == 0
                                else dm[off + path[i - 1], off + path[j]])
                    right_new = (cout(off + path[i]) if j == n - 1
                                 else dm[off + path[i], off + path[j + 1]])
                    if (left_new + right_new) - (left_old + right_old) < -1e-10:
                        path[i:j + 1] = path[i:j + 1][::-1]
                        improved = True
                        break
                if improved:
                    break
        return path

    def _boundary_refine(self, order, paths, dm, dd, region_offsets) -> dict:
        """Re-optimise each region's intra path against its actual neighbour connections."""
        R = len(order)
        for idx, r in enumerate(order):
            if len(paths[r]) < 3:
                continue
            pred_g = None if idx == 0 else region_offsets[order[idx - 1]] + paths[order[idx - 1]][-1]
            succ_g = None if idx == R - 1 else region_offsets[order[idx + 1]] + paths[order[idx + 1]][0]
            cin = (lambda g, pg=pred_g: dd[g] if pg is None else dm[pg, g])
            cout = (lambda g, sg=succ_g: dd[g] if sg is None else dm[g, sg])
            paths[r] = self._boundary_two_opt(paths[r], region_offsets[r], dm, cin, cout)
        return paths

    def _fixed_start_intra_path(self, r, start, region_offsets, sizes, dm) -> tuple:
        """Greedy NN from a fixed start + local search → (near-optimal path, intra cost)."""
        path, _ = self._greedy_path_from(r, start, region_offsets, sizes, dm)
        path = self._intra_local_search(path, region_offsets[r], dm)
        off = region_offsets[r]
        cost = sum(dm[off + path[k], off + path[k + 1]] for k in range(len(path) - 1))
        return path, cost

    def _inter_region_improve(self, region_order, region_paths, dm, dd, region_offsets) -> tuple:
        """Region-level Or-opt: relocate each region to its best position, optionally reversed.

        Delta-evaluated. Total tour cost = Σ_r (intra edges of r) + L, where L is the
        connection cost of the depot→entry, exit→entry and exit→depot links. The intra
        sum is invariant under relocation and path reversal (dm is symmetric, so a
        reversed path has the same undirected edge sum), so a candidate move's cost
        change equals the change in L over only the ≤6 links it touches — O(R²) per
        pass instead of O(R³·n̄) full recomputes. The scan order and first-improvement
        acceptance are identical to a full-recompute scan, so the result is unchanged."""
        R = len(region_order)
        if R < 2:
            return region_order, region_paths

        def entry_exit(r):
            off, p = region_offsets[r], region_paths[r]
            return off + p[0], off + p[-1]

        def link(a_exit, b_entry):
            # exit of predecessor → entry of successor; None denotes the depot.
            if a_exit is None:
                return 0.0 if b_entry is None else dd[b_entry]
            if b_entry is None:
                return dd[a_exit]
            return dm[a_exit, b_entry]

        improved = True
        while improved:
            improved = False
            for i in range(R):
                r = region_order[i]
                e_r, x_r = entry_exit(r)
                pred_x = None if i == 0 else entry_exit(region_order[i - 1])[1]
                succ_e = None if i == R - 1 else entry_exit(region_order[i + 1])[0]
                remove_gain = link(pred_x, e_r) + link(x_r, succ_e) - link(pred_x, succ_e)

                remaining = region_order[:i] + region_order[i + 1:]
                nr = len(remaining)
                for j in range(nr + 1):
                    a_x = None if j == 0 else entry_exit(remaining[j - 1])[1]
                    b_e = None if j == nr else entry_exit(remaining[j])[0]
                    base = link(a_x, b_e)
                    # forward (entry e_r, exit x_r) then reversed (entry x_r, exit e_r)
                    for reverse, ins in ((False, link(a_x, e_r) + link(x_r, b_e) - base),
                                         (True, link(a_x, x_r) + link(e_r, b_e) - base)):
                        if ins - remove_gain < -1e-10:
                            region_order = remaining[:j] + [r] + remaining[j:]
                            region_paths = dict(region_paths)
                            if reverse:
                                region_paths[r] = list(reversed(region_paths[r]))
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return region_order, region_paths

    def _vrp_local_search(self, region_order, region_paths, dm, dd, region_offsets,
                          strong: bool = False) -> tuple:
        """Intra-region local search on every region, then inter-region relocation.
        strong=True uses Lin-Kernighan intra (run_lkh); the rest use 2-opt+Or-opt."""
        for r_idx in region_order:
            region_paths[r_idx] = self._intra_local_search(
                region_paths[r_idx], region_offsets[r_idx], dm, strong)
        return self._inter_region_improve(region_order, region_paths, dm, dd, region_offsets)

    def _polish(self, order, paths, dm, dd, region_offsets) -> tuple:
        """Coupled polish: alternate inter-region relocation and boundary-aware intra
        refinement until the total tour cost stops improving."""
        while True:
            before = self._compute_tour_cost(order, paths, dm, dd, region_offsets)
            order, paths = self._inter_region_improve(order, paths, dm, dd, region_offsets)
            paths = self._boundary_refine(order, paths, dm, dd, region_offsets)
            if before - self._compute_tour_cost(order, paths, dm, dd, region_offsets) < 1e-9:
                break
        return order, paths

    def _double_bridge_regions(self, region_order: list) -> list:
        """4-opt double-bridge perturbation (A+C+B+D) of the region sequence."""
        R = len(region_order)
        if R < 4:
            return region_order[:]
        a, b, c = sorted(random.sample(range(1, R), 3))
        return region_order[:a] + region_order[b:c] + region_order[a:b] + region_order[c:]

    # ── ACO ───────────────────────────────────────────────────────────────────

    def _aco_construct_ant(self, dm, dd, region_offsets, sizes, tau_inter, tau_intra,
                           alpha, beta, k_entries=3, rng=None) -> tuple:
        """Build one ant's VRP solution from pheromone + heuristic probabilities.

        The region entry is chosen among the K nearest candidates using pheromone
        (mean outgoing τ_intra) and a distance heuristic, so ACO learns which entry
        yields good downstream paths rather than always taking the nearest.

        rng is an np.random.Generator (or the np.random module) so ant streams are
        explicit and reproducible; workers get independent, seeded generators.
        """
        rng = rng if rng is not None else np.random
        R = len(region_offsets)
        visited = [False] * R
        region_order, region_paths = [], {}
        prev_r, prev_g = R, None  # R = depot row in tau_inter

        for _ in range(sum(1 for r in range(R) if sizes[r] > 0)):
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
            next_r = int(rng.choice(R, p=probs / probs.sum()))

            off, n_r = region_offsets[next_r], sizes[next_r]
            tau_r = tau_intra[next_r]

            dists_e = dd[off:off + n_r] if prev_g is None else dm[prev_g, off:off + n_r]
            k = min(k_entries, n_r)
            k_cand = np.argsort(dists_e)[:k]
            eta_e = 1.0 / (dists_e[k_cand] + 1e-10)
            tau_e = np.array([tau_r[s].mean() for s in k_cand])
            attract_e = (tau_e ** alpha) * (eta_e ** beta)
            s_e = attract_e.sum()
            probs_e = attract_e / s_e if s_e > 1e-10 else np.ones(k) / k
            entry = int(k_cand[rng.choice(k, p=probs_e / probs_e.sum())])

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
                if total_vp > 1e-10:
                    probs_vp = attract_vp / total_vp
                else:
                    mask_vp = np.array([0.0 if vp_vis[j] else 1.0 for j in range(n_r)])
                    probs_vp = mask_vp / mask_vp.sum()
                nxt = int(rng.choice(n_r, p=probs_vp / probs_vp.sum()))
                vp_vis[nxt] = True
                path.append(nxt)

            region_paths[next_r] = path
            visited[next_r] = True
            region_order.append(next_r)
            prev_r, prev_g = next_r, off + path[-1]

        return region_order, region_paths

    # ── VRP algorithms ────────────────────────────────────────────────────────

    def run_greedy(self, ik_result: IKResult, region_offsets: list) -> VRPSolution:
        """Nearest-neighbor intra-region, greedy inter-region ordering."""
        dm, dd, sizes, _ = self._setup(ik_result, region_offsets)
        order, paths = self._greedy_vrp(dm, dd, region_offsets, sizes)
        return self._finalize(order, paths, dm, dd, region_offsets, 'vrp_greedy')

    def run_2opt(self, ik_result: IKResult, region_offsets: list) -> VRPSolution:
        """Greedy init + intra-region 2-opt + inter-region relocation."""
        dm, dd, sizes, _ = self._setup(ik_result, region_offsets)
        order, paths = self._greedy_vrp(dm, dd, region_offsets, sizes)
        for r_idx in order:
            paths[r_idx] = self._intra_two_opt(paths[r_idx], region_offsets[r_idx], dm)
        order, paths = self._polish(order, paths, dm, dd, region_offsets)
        return self._finalize(order, paths, dm, dd, region_offsets, 'vrp_2opt')

    def run_3opt(self, ik_result: IKResult, region_offsets: list) -> VRPSolution:
        """Greedy init + intra-region 2-opt & Or-opt(1) + inter-region relocation."""
        dm, dd, sizes, _ = self._setup(ik_result, region_offsets)
        order, paths = self._greedy_vrp(dm, dd, region_offsets, sizes)
        for r_idx in order:
            paths[r_idx] = self._intra_local_search(paths[r_idx], region_offsets[r_idx], dm)
        order, paths = self._polish(order, paths, dm, dd, region_offsets)
        return self._finalize(order, paths, dm, dd, region_offsets, 'vrp_3opt')

    def run_ils(self, ik_result: IKResult, region_offsets: list,
                n_restarts: int = 50, n_candidates: int = 1) -> VRPSolution:
        """Greedy init → local search → double-bridge perturbation → local search → keep best."""
        dm, dd, sizes, _ = self._setup(ik_result, region_offsets)
        best_order, best_paths = self._greedy_vrp(dm, dd, region_offsets, sizes)
        best_order, best_paths = self._vrp_local_search(
            best_order, best_paths, dm, dd, region_offsets)
        best_cost = self._compute_tour_cost(best_order, best_paths, dm, dd, region_offsets)

        pool = _TopK(n_candidates)
        pool.add(best_order, best_paths, best_cost)
        for _ in range(n_restarts):
            new_order = self._double_bridge_regions(best_order)
            new_paths = {r: p[:] for r, p in best_paths.items()}
            new_order, new_paths = self._vrp_local_search(
                new_order, new_paths, dm, dd, region_offsets)
            c = self._compute_tour_cost(new_order, new_paths, dm, dd, region_offsets)
            pool.add(new_order, new_paths, c)
            if c < best_cost - 1e-10:
                best_cost, best_order, best_paths = c, new_order, new_paths
        best_order, best_paths = self._polish(best_order, best_paths, dm, dd, region_offsets)
        pool.add(best_order, best_paths,
                 self._compute_tour_cost(best_order, best_paths, dm, dd, region_offsets))
        return self._finalize(best_order, best_paths, dm, dd, region_offsets,
                              'vrp_ils', pool.items())

    def run_lkh(self, ik_result: IKResult, region_offsets: list,
                n_restarts: int = 100, n_candidates: int = 1) -> VRPSolution:
        """Multi-start Lin-Kernighan: each restart seeds from a fresh random-order
        greedy init, then optimises with candidate-restricted LK intra (depth-2
        sequential moves) — the strong intra local search that distinguishes this from
        the 2-opt/Or-opt-based run_ils, so LKH vs ILS/ACO/clustered is a fair contrast."""
        dm, dd, sizes, R = self._setup(ik_result, region_offsets)
        best_order = best_paths = None
        best_cost = float('inf')
        pool = _TopK(n_candidates)

        for _ in range(n_restarts):
            rand_order = [r for r in range(R) if sizes[r] > 0]
            random.shuffle(rand_order)
            paths, prev_g = {}, None
            for r_idx in rand_order:
                cands = {s: self._greedy_path_from(r_idx, s, region_offsets, sizes, dm)
                         for s in range(sizes[r_idx])}
                best_s = min(cands, key=lambda s: (
                    dd[region_offsets[r_idx] + s] if prev_g is None
                    else dm[prev_g, region_offsets[r_idx] + s]) + cands[s][1])
                paths[r_idx] = cands[best_s][0]
                prev_g = region_offsets[r_idx] + paths[r_idx][-1]

            order, paths = self._vrp_local_search(
                rand_order, paths, dm, dd, region_offsets, strong=True)
            local_best_order, local_best_paths = order[:], dict(paths)
            local_best_cost = self._compute_tour_cost(
                local_best_order, local_best_paths, dm, dd, region_offsets)
            for _ in range(5):
                p_order = self._double_bridge_regions(local_best_order)
                p_paths = {r: p[:] for r, p in local_best_paths.items()}
                p_order, p_paths = self._vrp_local_search(
                    p_order, p_paths, dm, dd, region_offsets, strong=True)
                c = self._compute_tour_cost(p_order, p_paths, dm, dd, region_offsets)
                if c < local_best_cost - 1e-10:
                    local_best_cost, local_best_order, local_best_paths = c, p_order, p_paths
            pool.add(local_best_order, local_best_paths, local_best_cost)
            if local_best_cost < best_cost - 1e-10:
                best_cost, best_order, best_paths = \
                    local_best_cost, local_best_order, local_best_paths
        best_order, best_paths = self._polish(best_order, best_paths, dm, dd, region_offsets)
        pool.add(best_order, best_paths,
                 self._compute_tour_cost(best_order, best_paths, dm, dd, region_offsets))
        return self._finalize(best_order, best_paths, dm, dd, region_offsets,
                              'vrp_lkh', pool.items())

    def _aco_ant(self, dm, dd, region_offsets, sizes, tau_inter, tau_intra,
                 alpha, beta, k_entries, rng=None) -> tuple:
        """One ant: pheromone-guided construction + intra-region local search.

        Intra-only (no inter-region relocation) so the per-ant cost stays low enough
        to run many ants; the expensive inter-region relocation is applied once per
        iteration to the iteration best.
        """
        order, paths = self._aco_construct_ant(
            dm, dd, region_offsets, sizes, tau_inter, tau_intra, alpha, beta, k_entries, rng)
        for r_idx in order:
            paths[r_idx] = self._intra_local_search(paths[r_idx], region_offsets[r_idx], dm)
        return order, paths, self._compute_tour_cost(order, paths, dm, dd, region_offsets)

    def run_aco(self, ik_result: IKResult, region_offsets: list,
                n_ants: int = 20, n_iter: int = 100,
                alpha: float = 1.0, beta: float = 2.0, rho: float = 0.1,
                k_entries: int = 3, n_candidates: int = 1, n_jobs: int = 1) -> VRPSolution:
        """MMAS ACO with two-level pheromone (τ_inter region, τ_intra per region).

        Each ant selects entries from the K nearest candidates using pheromone +
        distance heuristic, then runs intra-region local search. Once per iteration
        the iteration best gets a full inter-region relocation before pheromone is
        deposited, and the global best is polished at the end. Ant RNG streams are
        explicit and seeded via VRPSolver(seed=…): reproducible and independent across
        parallel workers (E). Ants within an iteration run in parallel when n_jobs > 1.
        """
        dm, dd, sizes, R = self._setup(ik_result, region_offsets)

        best_order, best_paths = self._greedy_vrp(dm, dd, region_offsets, sizes)
        best_cost = self._compute_tour_cost(best_order, best_paths, dm, dd, region_offsets)
        tau_init = 1.0 / (rho * best_cost)
        tau_inter = np.full((R + 1, R), tau_init)   # row R = depot transitions
        tau_intra = [np.full((sizes[r], sizes[r]), tau_init) for r in range(R)]
        pool = _TopK(n_candidates)
        pool.add(best_order, best_paths, best_cost)

        ss = np.random.SeedSequence(self.seed)
        pooler = _AntPool(self, dm, dd, region_offsets, sizes, n_jobs) if n_jobs > 1 else None
        try:
            for _ in range(n_iter):
                iter_seeds = ss.spawn(n_ants)   # independent, reproducible per ant
                if pooler is not None:
                    ants = pooler.run(n_ants, tau_inter, tau_intra,
                                      alpha, beta, k_entries, iter_seeds)
                else:
                    ants = [self._aco_ant(dm, dd, region_offsets, sizes,
                                          tau_inter, tau_intra, alpha, beta, k_entries,
                                          np.random.default_rng(s))
                            for s in iter_seeds]

                iter_order, iter_paths, iter_cost = min(ants, key=lambda a: a[2])
                # Inter-region relocation on the iteration best only (expensive step)
                iter_order, iter_paths = self._inter_region_improve(
                    iter_order, iter_paths, dm, dd, region_offsets)
                iter_cost = self._compute_tour_cost(
                    iter_order, iter_paths, dm, dd, region_offsets)
                pool.add(iter_order, iter_paths, iter_cost)
                if iter_cost < best_cost - 1e-10:
                    best_cost, best_order, best_paths = iter_cost, iter_order[:], dict(iter_paths)

                # Evaporate → deposit iteration best → clip to [tau_min, tau_max]
                tau_inter *= (1 - rho)
                for r in range(R):
                    tau_intra[r] *= (1 - rho)
                delta = 1.0 / (iter_cost + 1e-10)
                tau_max = 1.0 / (rho * best_cost)

                prev_r = R
                for r_idx in iter_order:
                    tau_inter[prev_r, r_idx] += delta
                    prev_r = r_idx
                for r_idx in iter_order:
                    path = iter_paths[r_idx]
                    for k in range(len(path) - 1):
                        a, b = path[k], path[k + 1]
                        # Symmetric deposit: intra paths are reversible (dm is
                        # symmetric and _inter_region_improve may flip them), so
                        # edge {a,b} earns reinforcement in both directions (C).
                        tau_intra[r_idx][a, b] += delta
                        tau_intra[r_idx][b, a] += delta

                np.clip(tau_inter, tau_max / (R * 2), tau_max, out=tau_inter)
                for r in range(R):
                    if sizes[r] > 0:
                        np.clip(tau_intra[r], tau_max / (sizes[r] * 2), tau_max, out=tau_intra[r])
        finally:
            if pooler is not None:
                pooler.close()

        best_order, best_paths = self._polish(best_order, best_paths, dm, dd, region_offsets)
        pool.add(best_order, best_paths,
                 self._compute_tour_cost(best_order, best_paths, dm, dd, region_offsets))
        return self._finalize(best_order, best_paths, dm, dd, region_offsets,
                              'vrp_aco', pool.items())

    def run_hierarchical(self, ik_result: IKResult, region_offsets: list,
                         k_entries: int = 3) -> VRPSolution:
        """Hierarchical sequential VRP.

        At each step, for every unvisited region compute a near-optimal open path
        (greedy NN + 2-opt/Or-opt) from each of its K nearest entries to the current
        arm config, then greedily commit the (region, entry) pair that minimises
        transition + intra cost. Guarantees near-optimal intra paths with a
        cost-minimising inter-region sequence.
        """
        dm, dd, sizes, R = self._setup(ik_result, region_offsets)
        unvisited = [r for r in range(R) if sizes[r] > 0]
        region_order, region_paths = [], {}
        prev_g = None

        while unvisited:
            best_r = best_path = None
            best_score = float('inf')
            for r in unvisited:
                off, n_r = region_offsets[r], sizes[r]
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

        region_order, region_paths = self._polish(
            region_order, region_paths, dm, dd, region_offsets)
        return self._finalize(region_order, region_paths, dm, dd, region_offsets,
                              'vrp_hierarchical')

    # ── Clustered GTSP: exact port-coupled entry/exit + region ordering ────────

    _HK_MAX = 13  # region size below which P_r(e,x) is solved exactly (Held-Karp)

    def _select_ports(self, r, region_offsets, sizes, dm, dd, K) -> list:
        """K candidate entry/exit ports for region r via farthest-point sampling
        (2-approx k-center) in the dm metric, seeded at the depot-nearest node.
        Thin wrapper over _fps_ports on region r's submatrix (shared with the
        config-aware path, which supplies the optimistic-cost metric instead)."""
        off, n = region_offsets[r], sizes[r]
        return self._fps_ports(dm[off:off + n, off:off + n], dd[off:off + n], n, K)

    def _hk_from(self, off, n, dm, start) -> tuple:
        """Held-Karp shortest Hamiltonian path from `start`; returns per-end
        (cost, path) over all nodes."""
        INF = float('inf')
        size = 1 << n
        dp = [[INF] * n for _ in range(size)]
        par = [[-1] * n for _ in range(size)]
        dp[1 << start][start] = 0.0
        for mask in range(size):
            if not (mask >> start) & 1:
                continue
            for j in range(n):
                if not (mask >> j) & 1 or dp[mask][j] == INF:
                    continue
                cur = dp[mask][j]
                for k in range(n):
                    if (mask >> k) & 1:
                        continue
                    nm = mask | (1 << k)
                    nc = cur + dm[off + j, off + k]
                    if nc < dp[nm][k]:
                        dp[nm][k] = nc
                        par[nm][k] = j
        full = size - 1
        costs, paths = {}, {}
        for end in range(n):
            costs[end] = dp[full][end]
            path, mask, j = [], full, end
            while j != -1:
                path.append(j)
                pj = par[mask][j]
                mask ^= (1 << j)
                j = pj
            paths[end] = path[::-1]
        return costs, paths

    def _pinned_path(self, off, n, dm, e, x) -> tuple:
        """Heuristic fixed-endpoint Hamiltonian path e→x (NN reserving x + endpoint-locked
        2-opt / Or-opt)."""
        visited = [False] * n
        visited[e] = visited[x] = True
        path = [e]
        for _ in range(n - 2):
            cur, bd, bj = path[-1], float('inf'), -1
            for j in range(n):
                if visited[j] or j == x:
                    continue
                d = dm[off + cur, off + j]
                if d < bd:
                    bd, bj = d, j
            visited[bj] = True
            path.append(bj)
        path.append(x)
        path = self._intra_two_opt(path, off, dm, lock_end=True)
        for seg_len in (1, 2, 3):
            path = self._intra_or_opt(path, off, dm, seg_len, lock_end=True)
        cost = sum(dm[off + path[k], off + path[k + 1]] for k in range(n - 1))
        return path, cost

    def _intra_endpoint_table(self, r, ports, region_offsets, sizes, dm) -> dict:
        """P_r(e,x) and the achieving path for every ordered port pair (symmetric)."""
        off, n = region_offsets[r], sizes[r]
        table = {}
        if n == 1:
            table[(0, 0)] = (0.0, [0])
            return table
        if n <= self._HK_MAX:
            for e in ports:
                costs, paths = self._hk_from(off, n, dm, e)
                for x in ports:
                    if x != e:
                        table[(e, x)] = (costs[x], paths[x])
        else:
            for a in ports:
                for b in ports:
                    if b <= a:
                        continue
                    path, cost = self._pinned_path(off, n, dm, a, b)
                    table[(a, b)] = (cost, path)
                    table[(b, a)] = (cost, path[::-1])
        return table

    def _port_layer_shortest_path(self, order, tables, ports, region_offsets, dm, dd) -> tuple:
        """Exact entry/exit assignment for a FIXED region order via layered-DAG SP."""
        INF = float('inf')
        dp_list, prev_dp = [], None
        for idx, r in enumerate(order):
            off, prts = region_offsets[r], ports[r]
            pairs = ([(0, 0)] if len(prts) == 1
                     else [(e, x) for e in prts for x in prts if e != x])
            cur, bk = {}, {}
            for (e, x) in pairs:
                if (e, x) not in tables[r]:
                    continue
                pcost = tables[r][(e, x)][0]
                if idx == 0:
                    c, pb = dd[off + e] + pcost, None
                else:
                    poff = region_offsets[order[idx - 1]]
                    best, pb = INF, None
                    for (pe, px), pc in prev_dp.items():
                        cc = pc + dm[poff + px, off + e]
                        if cc < best:
                            best, pb = cc, (pe, px)
                    c = best + pcost
                if c < cur.get((e, x), INF):
                    cur[(e, x)], bk[(e, x)] = c, pb
            prev_dp = cur
            dp_list.append((r, cur, bk))

        off = region_offsets[order[-1]]
        best, state = INF, None
        for (e, x), c in dp_list[-1][1].items():
            cc = c + dd[off + x]
            if cc < best:
                best, state = cc, (e, x)

        paths = {}
        for idx in range(len(order) - 1, -1, -1):
            r, _, bk = dp_list[idx]
            e, x = state
            paths[r] = list(tables[r][(e, x)][1])
            state = bk[(e, x)]
        return best, paths

    def _portdp_exact(self, active, tables, ports, region_offsets, dm, dd) -> tuple:
        """Held-Karp over region subsets with an exit-port state; entry minimized at
        each transition. Globally optimal order + entry/exit given ports and tables."""
        INF = float('inf')
        A, m = active, len(active)

        def enter(i, gp, xi):
            r = A[i]
            off, prts = region_offsets[r], ports[r]
            x_local = prts[xi]
            best, best_e = INF, None
            for e in prts:
                key = (e, x_local)
                if key not in tables[r]:
                    continue
                trans = dd[off + e] if gp is None else dm[gp, off + e]
                c = trans + tables[r][key][0]
                if c < best:
                    best, best_e = c, e
            return best, best_e

        g, back = {}, {}
        for i in range(m):
            for xi in range(len(ports[A[i]])):
                c, e = enter(i, None, xi)
                if c < INF:
                    st = (1 << i, i, xi)
                    if c < g.get(st, INF):
                        g[st], back[st] = c, (None, None, e)

        for S in range(1, 1 << m):
            for i in range(m):
                if not (S >> i) & 1:
                    continue
                r, prts = A[i], ports[A[i]]
                for xi in range(len(prts)):
                    st = (S, i, xi)
                    base = g.get(st, INF)
                    if base == INF:
                        continue
                    gp = region_offsets[r] + prts[xi]
                    for j in range(m):
                        if (S >> j) & 1:
                            continue
                        S2 = S | (1 << j)
                        for xj in range(len(ports[A[j]])):
                            c, e = enter(j, gp, xj)
                            if c == INF:
                                continue
                            nc = base + c
                            st2 = (S2, j, xj)
                            if nc < g.get(st2, INF):
                                g[st2], back[st2] = nc, (i, xi, e)

        full = (1 << m) - 1
        best, best_st = INF, None
        for i in range(m):
            off, prts = region_offsets[A[i]], ports[A[i]]
            for xi in range(len(prts)):
                st = (full, i, xi)
                if st in g:
                    c = g[st] + dd[off + prts[xi]]
                    if c < best:
                        best, best_st = c, st

        order, paths, st = [], {}, best_st
        while st is not None:
            S, i, xi = st
            r, prts = A[i], ports[A[i]]
            prev_i, prev_xi, entry_e = back[st]
            paths[r] = list(tables[r][(entry_e, prts[xi])][1])
            order.append(r)
            st = None if prev_i is None else (S & ~(1 << i), prev_i, prev_xi)
        order.reverse()
        return order, paths, best

    def _portdp_meta(self, active, tables, ports, region_offsets, dm, dd, n_restarts) -> tuple:
        """Large-R fallback: ILS over region orders, each scored by the exact layered SP."""
        best_order = list(active)
        best_cost, best_paths = self._port_layer_shortest_path(
            best_order, tables, ports, region_offsets, dm, dd)
        for _ in range(n_restarts):
            cand = self._double_bridge_regions(best_order)
            c, paths = self._port_layer_shortest_path(
                cand, tables, ports, region_offsets, dm, dd)
            if c < best_cost - 1e-10:
                best_cost, best_order, best_paths = c, cand, paths
        return best_order, best_paths, best_cost

    # ── Config-aware P_r: per-viewpoint IK-config selection along the intra path ─

    def _region_configs(self, ik_result: IKResult, region_offsets: list, sizes: list,
                        active: list) -> dict:
        """cfg_regions[r][i] = (m,7) candidate configs for local viewpoint i of region r.
        Falls back to the single q_all config when q_configs is absent/empty."""
        cfgs = {}
        for r in active:
            off = region_offsets[r]
            reg = []
            for i in range(sizes[r]):
                g = off + i
                c = ik_result.q_configs[g] if ik_result.q_configs is not None else None
                if c is None or len(c) == 0:
                    c = ik_result.q_all[g][None, :]
                reg.append(np.asarray(c, dtype=float))
            cfgs[r] = reg
        return cfgs

    def _config_select(self, ordered_cfgs: list, e=None, x=None) -> tuple:
        """Min-cost config assignment along a FIXED viewpoint order (RoboTSP layered DP).
        e/x pin the first/last config index. Returns (cost, [cfg idx per position])."""
        n = len(ordered_cfgs)
        INF = float('inf')
        if n == 1:
            return 0.0, [e if e is not None else (x if x is not None else 0)]
        m0 = len(ordered_cfgs[0])
        dp = np.array([0.0 if (e is None or c == e) else INF for c in range(m0)])
        par = []
        for i in range(1, n):
            # Vectorised transition: cost of every (prev cfg, cur cfg) pair at once,
            # so one _config_cost call on an (m_prev, m_cur, 7) array replaces
            # m_prev·m_cur scalar calls. argmin keeps the first minimiser, matching the
            # original strict-'<' scan; INF-dp predecessors propagate to INF and lose.
            prev = np.asarray(ordered_cfgs[i - 1], dtype=float)
            cur = np.asarray(ordered_cfgs[i], dtype=float)
            total = dp[:, None] + self._config_cost(prev[:, None, :] - cur[None, :, :])
            pp = np.argmin(total, axis=0)
            ndp = total[pp, np.arange(total.shape[1])]
            par.append(np.where(np.isfinite(ndp), pp, -1))
            dp = ndp
        endc = x if x is not None else int(np.argmin(dp))
        choice = [0] * n
        choice[n - 1], c = endc, endc
        for i in range(n - 1, 0, -1):
            c = int(par[i - 1][c])
            choice[i - 1] = c
        return float(dp[endc]), choice

    def _opt_vp_cost(self, cfg_region: list) -> np.ndarray:
        """Optimistic viewpoint-to-viewpoint cost = min over config pairs (for ordering)."""
        n = len(cfg_region)
        c = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = min(float(self._config_cost(a - b))
                        for a in cfg_region[i] for b in cfg_region[j])
                c[i, j] = c[j, i] = d
        return c

    def _config_endpoint_table(self, cfg_region: list, ports_vp: list,
                               copt: np.ndarray) -> dict:
        """Config-aware P_r keyed by (entry_vp, entry_cfg, exit_vp, exit_cfg).
        Viewpoint order fixed by optimistic cost, then exact config selection."""
        n = len(cfg_region)
        table = {}
        for e_vp in ports_vp:
            if n <= self._HK_MAX:
                _, paths = self._hk_from(0, n, copt, e_vp)
            for x_vp in ports_vp:
                if n > 1 and e_vp == x_vp:
                    continue
                if n == 1:
                    order = [0]
                elif n <= self._HK_MAX:
                    order = paths[x_vp]
                else:
                    order, _ = self._pinned_path(0, n, copt, e_vp, x_vp)
                ordered = [cfg_region[v] for v in order]
                for ce in range(len(cfg_region[e_vp])):
                    for cx in range(len(cfg_region[x_vp])):
                        if n == 1:
                            if ce != cx:
                                continue
                            cost, choice = 0.0, [ce]
                        else:
                            cost, choice = self._config_select(ordered, e=ce, x=cx)
                        if cost < float('inf'):
                            table[(e_vp, ce, x_vp, cx)] = (cost, order, choice)
        return table

    def _cfg_layer_sp(self, order, tables, cfg_regions) -> tuple:
        """Config-aware layered shortest path for a FIXED region order. Ports are
        (viewpoint, config); inter-region transitions use the actual chosen configs."""
        INF = float('inf')
        layers, prev = [], None
        for idx, r in enumerate(order):
            cfgs, tbl = cfg_regions[r], tables[r]
            # Best incoming cost per current-region entry (vp, cfg). This depends on
            # the previous exit and the entry config, NOT on which exit we head to, so
            # it is computed once per entry (batched over prev states) rather than
            # redundantly inside the per-exit loop. The result is cost-optimal and
            # identical to the scalar scan on real (float) configs; on exactly-tied
            # inputs it may pick a different equally-optimal config among the ties.
            entries = list({(k[0], k[1]) for k in tbl})
            best_in = {}
            if entries:
                Qe = np.array([cfgs[ev][ce] for (ev, ce) in entries])
                if idx == 0:
                    incost = self._config_cost(Qe - DEPOT_Q)
                    best_in = {ent: (float(incost[j]), None)
                               for j, ent in enumerate(entries)}
                elif prev:
                    prev_items = list(prev.items())
                    Qprev = np.array([cfg_regions[order[idx - 1]][pxv][pcx]
                                      for (pxv, pcx), _ in prev_items])
                    pcosts = np.array([v[0] for _, v in prev_items])
                    tot = pcosts[:, None] + self._config_cost(
                        Qprev[:, None, :] - Qe[None, :, :])
                    amin = np.argmin(tot, axis=0)
                    best_in = {ent: (float(tot[amin[j], j]), prev_items[amin[j]][0])
                               for j, ent in enumerate(entries)}
            cur, bk = {}, {}
            for (x_vp, cx) in {(k[2], k[3]) for k in tbl}:
                best, barg = INF, None
                for (e_vp, ce) in {(k[0], k[1]) for k in tbl if (k[2], k[3]) == (x_vp, cx)}:
                    if (e_vp, ce) not in best_in:
                        continue
                    bi, pstate = best_in[(e_vp, ce)]
                    val = bi + tbl[(e_vp, ce, x_vp, cx)][0]
                    if val < best:
                        best, barg = val, (e_vp, ce, pstate)
                if barg is not None:
                    cur[(x_vp, cx)], bk[(x_vp, cx)] = (best, barg), barg
            prev = cur
            layers.append((r, cur, bk))

        r_last = order[-1]
        best, state = INF, None
        for (x_vp, cx), (cost, _) in layers[-1][1].items():
            val = cost + float(self._config_cost(cfg_regions[r_last][x_vp][cx] - DEPOT_Q))
            if val < best:
                best, state = val, (x_vp, cx)

        paths, cfg_choice = {}, {}
        for idx in range(len(order) - 1, -1, -1):
            r, _, bk = layers[idx]
            e_vp, ce, prev_state = bk[state]
            x_vp, cx = state
            _, vp_order, choice = tables[r][(e_vp, ce, x_vp, cx)]
            paths[r], cfg_choice[r] = list(vp_order), list(choice)
            state = prev_state
        return best, list(order), paths, cfg_choice

    def _run_clustered_configs(self, ik_result, region_offsets, sizes, active, K,
                               n_restarts) -> VRPSolution:
        cfg_regions = self._region_configs(ik_result, region_offsets, sizes, active)
        copt = {r: self._opt_vp_cost(cfg_regions[r]) for r in active}
        ddv = {r: np.array([min(float(self._config_cost(c - DEPOT_Q)) for c in cfg_regions[r][i])
                            for i in range(sizes[r])]) for r in active}
        ports = {r: self._fps_ports(copt[r], ddv[r], sizes[r], K) for r in active}
        tables = {r: self._config_endpoint_table(cfg_regions[r], ports[r], copt[r])
                  for r in active}

        if len(active) <= 7:
            best = min((self._cfg_layer_sp(list(o), tables, cfg_regions)
                        for o in itertools.permutations(active)), key=lambda t: t[0])
        else:
            best = self._cfg_layer_sp(list(active), tables, cfg_regions)
            for _ in range(n_restarts):
                cand = self._cfg_layer_sp(
                    self._double_bridge_regions(best[1]), tables, cfg_regions)
                if cand[0] < best[0] - 1e-10:
                    best = cand
        cost, order, paths, cfg_choice = best
        sol = VRPSolution(
            region_order=order, region_paths=paths, cost=cost,
            intra_cost=0.0, inter_cost=0.0, depot_cost=0.0,
            algorithm='vrp_clustered_configs', region_configs=cfg_choice)
        sol.intra_cost, sol.inter_cost, sol.depot_cost = self._cfg_breakdown(
            order, paths, cfg_choice, cfg_regions)
        return sol

    @staticmethod
    def _fps_ports(metric, seed_key, n, K):
        """Farthest-point sampling (Gonzalez 1985, 2-approx k-center): pick K ports
        spread apart under the n×n `metric`, seeded at argmin(`seed_key`) — the
        depot-nearest node. Shared by both the plain (metric=dm submatrix, seed=dd)
        and config-aware (metric=optimistic cost, seed=depot config cost) solvers."""
        if n <= K:
            return list(range(n))
        seed = int(np.argmin(seed_key))
        ports, mind = [seed], metric[:, seed].astype(float).copy()
        while len(ports) < K:
            nxt = int(np.argmax(mind))
            if mind[nxt] <= 0:
                break
            ports.append(nxt)
            mind = np.minimum(mind, metric[:, nxt])
        return sorted(set(ports))

    def _cfg_breakdown(self, order, paths, cfg_choice, cfg_regions) -> tuple:
        intra = inter = depot = 0.0
        prev_q = None
        for idx, r in enumerate(order):
            path, choice = paths[r], cfg_choice[r]
            q_entry = cfg_regions[r][path[0]][choice[0]]
            if prev_q is None:
                depot += float(self._config_cost(q_entry - DEPOT_Q))
            else:
                inter += float(self._config_cost(prev_q - q_entry))
            for k in range(len(path) - 1):
                a = cfg_regions[r][path[k]][choice[k]]
                b = cfg_regions[r][path[k + 1]][choice[k + 1]]
                intra += float(self._config_cost(a - b))
            prev_q = cfg_regions[r][path[-1]][choice[-1]]
        if prev_q is not None:
            depot += float(self._config_cost(prev_q - DEPOT_Q))
        return intra, inter, depot

    def select_tour_configs(self, region_order, region_paths, ik_result: IKResult,
                            region_offsets: list):
        """Optimal per-viewpoint IK-config assignment for a FIXED tour (order + paths),
        via the exact config-selection DP over the depot-bracketed viewpoint chain.

        Lets any order-only algorithm (ACO, ILS, LKH, hierarchical, ...) execute smooth,
        swing-free motion: instead of leaving the config to a free-IK re-solve at plan
        time (which can jump to a far branch → large joint swing), it picks the config at
        every viewpoint that minimises joint-space travel along the whole tour, and the
        trajectory planner then plans to those exact joint goals. Unreachable viewpoints
        (no finite config) are skipped, matching how execution skips them.

        Returns (region_configs, total, intra, inter, depot) or None if no configs exist.
        """
        if ik_result.q_configs is None:
            return None
        qc, qa = ik_result.q_configs, ik_result.q_all

        def cfgs_of(g):
            c = qc[g] if g < len(qc) and qc[g] is not None else None
            if c is not None and len(c):
                c = np.asarray(c, dtype=float)
                c = c[np.all(np.isfinite(c), axis=1)]
                if len(c):
                    return c
            q = qa[g]
            return q[None, :] if np.all(np.isfinite(q)) else None

        depot = DEPOT_Q[None, :]
        ordered, slots = [depot], []
        region_configs = {r: [0] * len(region_paths[r]) for r in region_order}
        for r in region_order:
            off = region_offsets[r]
            for k, vp in enumerate(region_paths[r]):
                c = cfgs_of(off + vp)
                if c is None:
                    continue
                ordered.append(c)
                slots.append((r, k))
        if not slots:
            return None
        ordered.append(depot)

        total, choice = self._config_select(ordered, e=0, x=0)
        for i, (r, k) in enumerate(slots):
            region_configs[r][k] = choice[i + 1]

        vecs = [ordered[i][choice[i]] for i in range(len(ordered))]
        depot_c = (float(self._config_cost(vecs[0] - vecs[1]))
                   + float(self._config_cost(vecs[-2] - vecs[-1])))
        intra = inter = 0.0
        for j in range(len(slots) - 1):
            e = float(self._config_cost(vecs[j + 1] - vecs[j + 2]))
            if slots[j][0] == slots[j + 1][0]:
                intra += e
            else:
                inter += e
        return region_configs, total, intra, inter, depot_c

    def run_clustered(self, ik_result: IKResult, region_offsets: list,
                      K: int = 6, exact_budget: float = 5e8,
                      n_restarts: int = 200) -> VRPSolution:
        """Clustered-GTSP solver: FPS ports + fixed-endpoint intra paths P_r(e,x),
        with entry/exit and region order chosen jointly — exact port-DP when
        2^R·R²·K³ ≤ exact_budget, else ILS over orders with exact per-order ports.

        When ik_result carries multiple IK configs per viewpoint (q_configs), P_r
        also selects the per-viewpoint config (RoboTSP layered DP) and the region
        boundary transitions use the actual chosen configs (config-aware path)."""
        dm, dd, sizes, R = self._setup(ik_result, region_offsets)
        active = [r for r in range(R) if sizes[r] > 0]
        if not active:
            return self._finalize([], {}, dm, dd, region_offsets, 'vrp_clustered')

        multi = (ik_result.q_configs is not None
                 and any(len(c) > 1 for c in ik_result.q_configs))
        if multi:
            return self._run_clustered_configs(
                ik_result, region_offsets, sizes, active, max(2, K), n_restarts)

        ports = {r: self._select_ports(r, region_offsets, sizes, dm, dd, K) for r in active}
        tables = {r: self._intra_endpoint_table(r, ports[r], region_offsets, sizes, dm)
                  for r in active}

        m = len(active)
        kmax = max(len(ports[r]) for r in active)
        if (2 ** m) * (m ** 2) * (kmax ** 3) <= exact_budget:
            order, paths, _ = self._portdp_exact(
                active, tables, ports, region_offsets, dm, dd)
            algo = 'vrp_clustered_exact'
        else:
            order, paths, _ = self._portdp_meta(
                active, tables, ports, region_offsets, dm, dd, n_restarts)
            algo = 'vrp_clustered_meta'
        return self._finalize(order, paths, dm, dd, region_offsets, algo)


_ANT_CTX = {}


def _ant_pool_init(cost_mode, jw, vmax, amax, dm, dd, offs, sizes):
    """Worker initializer: rebuild a pure solver and cache read-only inputs."""
    solver = VRPSolver(joint_weights=jw, cost_mode=cost_mode,
                       max_velocity=vmax, max_acceleration=amax)
    _ANT_CTX.update(solver=solver, dm=dm, dd=dd, offs=offs, sizes=sizes)


def _ant_pool_run(args):
    tau_inter, tau_intra, alpha, beta, k_entries, seed = args
    c = _ANT_CTX
    return c['solver']._aco_ant(c['dm'], c['dd'], c['offs'], c['sizes'],
                                tau_inter, tau_intra, alpha, beta, k_entries,
                                np.random.default_rng(seed))


class _AntPool:
    """Spawn-based process pool for independent ACO ants.

    Uses 'spawn' so workers never inherit the parent's ROS/MoveIt state, and an
    initializer so the (large, read-only) distance matrix is pickled once per
    worker instead of once per ant.
    """

    def __init__(self, solver, dm, dd, offs, sizes, n_jobs):
        self._n_jobs = n_jobs
        ctx = _mp.get_context('spawn')
        self._pool = ctx.Pool(
            processes=n_jobs, initializer=_ant_pool_init,
            initargs=(solver.cost_mode, list(solver.joint_weights),
                      solver._vmax, solver._amax, dm, dd, offs, sizes))

    def run(self, n_ants, tau_inter, tau_intra, alpha, beta, k_entries, seeds):
        args = [(tau_inter, tau_intra, alpha, beta, k_entries, seeds[i])
                for i in range(n_ants)]
        chunk = max(1, -(-n_ants // self._n_jobs))
        return self._pool.map(_ant_pool_run, args, chunksize=chunk)

    def close(self):
        self._pool.terminate()
        self._pool.join()


class _TopK:
    """Keeps the K cheapest distinct solutions (by region_order signature)."""

    def __init__(self, k: int):
        self.k = max(1, k)
        self._items = []  # list of (order, paths, cost)
        self._seen = {}   # signature → best cost stored

    def add(self, order, paths, cost):
        if self.k <= 1:
            return
        sig = tuple(order)
        if sig in self._seen and self._seen[sig] <= cost + 1e-10:
            return
        self._seen[sig] = cost
        self._items = [it for it in self._items if tuple(it[0]) != sig]
        self._items.append((order[:], {r: p[:] for r, p in paths.items()}, cost))
        self._items.sort(key=lambda it: it[2])
        del self._items[self.k:]

    def items(self):
        return self._items
