import random
import numpy as np


class TSPSolver:

    def __init__(self, logger=None):
        self.logger = logger
        self.algorithm_results = {}
        self.completed_algorithms = set()
        self._region_gaps = {}

    def _held_karp(self, dm):
        """Exact open-path TSP via bitmask DP.
        Uses numpy arrays for N > 12 to handle up to N = 18 in reasonable time."""
        n = dm.shape[0]
        size = 1 << n
        INF = 1e18
        dp = np.full((size, n), INF)
        for s in range(n):
            dp[1 << s, s] = 0.0

        nexts_for = [np.array([j for j in range(n) if not (mask >> j & 1)], dtype=np.int32)
                     for mask in range(size)]

        for mask in range(1, size):
            nexts = nexts_for[mask]
            if len(nexts) == 0:
                continue
            for last in range(n):
                if not (mask >> last & 1):
                    continue
                cur = dp[mask, last]
                if cur >= INF:
                    continue
                costs = cur + dm[last, nexts]
                new_masks = mask | (1 << nexts)
                for k in range(len(nexts)):
                    nm = int(new_masks[k])
                    nxt = int(nexts[k])
                    if costs[k] < dp[nm, nxt]:
                        dp[nm, nxt] = costs[k]

        full = size - 1
        return float(np.min(dp[full]))

    def _mst_cost(self, dm):
        """Prim's MST — O(N²), lower bound on any open Hamiltonian path."""
        n = dm.shape[0]
        in_mst = np.zeros(n, dtype=bool)
        key = np.full(n, np.inf)
        key[0] = 0.0
        total = 0.0
        for _ in range(n):
            candidates = np.where(~in_mst)[0]
            u = candidates[np.argmin(key[candidates])]
            in_mst[u] = True
            total += key[u]
            mask = ~in_mst
            better = dm[u, mask] < key[mask]
            key[mask] = np.where(better, dm[u, mask], key[mask])
        return float(total)

    def dist_matrix(self, viewpoints):
        pts = np.array(viewpoints)
        diff = pts[:, None, :] - pts[None, :, :]
        return np.linalg.norm(diff, axis=-1)

    def dist_calc(self, dm, path):
        t = path[0] if (isinstance(path, list) and len(path) == 2
                        and isinstance(path[0], list)) else path
        if len(t) > 1 and t[-1] == t[0]:
            t = t[:-1]
        return float(sum(dm[t[k], t[k + 1]] for k in range(len(t) - 1)))

    # ── Core primitives ───────────────────────────────────────────────────────

    def _unpack(self, initial):
        """Accept either a plain list or [path, dist] pair."""
        if isinstance(initial, list) and len(initial) == 2 and isinstance(initial[0], list):
            return list(initial[0])
        return list(initial)

    def _greedy_from(self, dm, start):
        n = dm.shape[0]
        visited = [False] * n
        visited[start] = True
        path = [start]
        cost = 0.0
        cur = start
        for _ in range(n - 1):
            row = dm[cur]
            best_d, best_j = float('inf'), -1
            for j in range(n):
                if not visited[j] and row[j] < best_d:
                    best_d, best_j = row[j], j
            cost += best_d
            visited[best_j] = True
            path.append(best_j)
            cur = best_j
        return path, cost

    def _two_opt(self, path, dm):
        """Best-improvement 2-opt on open path. No tour closure anywhere."""
        t = list(path)
        n = len(t)
        while True:
            improved = False
            for i in range(n - 2):
                for j in range(i + 2, n):
                    # Reverse t[i+1..j]; open path, so the j==n-1 case adds no closing edge.
                    remove = dm[t[i], t[i + 1]]
                    add = dm[t[i], t[j]]
                    if j + 1 < n:
                        remove += dm[t[j], t[j + 1]]
                        add += dm[t[i + 1], t[j + 1]]
                    if add - remove < -1e-10:
                        t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
                        improved = True
            if not improved:
                break
        return t, self.dist_calc(dm, t)

    def _or_opt(self, path, dm, seg_len=1):
        """Relocate every run of seg_len nodes to its best insertion position."""
        t = list(path)
        n = len(t)
        while True:
            improved = False
            i = 0
            while i < n - seg_len:
                prev = i - 1
                nxt = i + seg_len
                # Segment must be strictly interior (has a node before and after)
                if prev < 0 or nxt >= n:
                    i += 1
                    continue
                seg = t[i:i + seg_len]
                # Gain from removing the segment
                remove_gain = (dm[t[prev], seg[0]] + dm[seg[-1], t[nxt]]
                               - dm[t[prev], t[nxt]])
                rest = t[:i] + t[i + seg_len:]
                m = len(rest)
                best_gain = 1e-10
                best_j = -1
                for j in range(m - 1):
                    ins_cost = (dm[rest[j], seg[0]] + dm[seg[-1], rest[j + 1]]
                                - dm[rest[j], rest[j + 1]])
                    gain = remove_gain - ins_cost
                    if gain > best_gain:
                        best_gain = gain
                        best_j = j
                if best_j >= 0:
                    t = rest[:best_j + 1] + seg + rest[best_j + 1:]
                    n = len(t)
                    improved = True
                else:
                    i += 1
            if not improved:
                break
        return t, self.dist_calc(dm, t)

    def _local_search(self, path, dm):
        t = list(path)
        prev_d = float('inf')
        while True:
            t, _ = self._two_opt(t, dm)
            t, _ = self._or_opt(t, dm, seg_len=1)
            t, _ = self._or_opt(t, dm, seg_len=2)
            t, _ = self._or_opt(t, dm, seg_len=3)
            d = self.dist_calc(dm, t)
            if prev_d - d < 1e-10:
                break
            prev_d = d
        return t, d

    def _double_bridge(self, tour):
        n = len(tour)
        if n < 8:
            return tour[:]
        cuts = sorted(random.sample(range(1, n), 4))
        a, b, c, d = cuts
        return tour[:a] + tour[c:d] + tour[b:c] + tour[a:b] + tour[d:]

    # ── Public algorithm entry points ─────────────────────────────────────────

    def nearest_neighbors_tsp(self, points):
        dm = self.dist_matrix(points)
        best_path, best_cost = self._greedy_from(dm, 0)
        for s in range(1, len(points)):
            p, c = self._greedy_from(dm, s)
            if c < best_cost:
                best_cost, best_path = c, p
        return best_path, best_cost

    def local_search_2_opt(self, dm, initial, recursive_seeding=-1, verbose=False):
        best_path, best_dist = self._two_opt(self._unpack(initial), dm)
        if verbose and self.logger:
            self.logger.info(f'  2-opt initial: {best_dist:.4f} m')
        if recursive_seeding > 0:
            n = dm.shape[0]
            nodes = list(range(n))
            for seed in range(recursive_seeding):
                random.shuffle(nodes)
                path, dist = self._two_opt(list(nodes), dm)
                if verbose and self.logger:
                    self.logger.info(f'  2-opt seed {seed + 1}/{recursive_seeding}: {dist:.4f} m')
                if dist < best_dist - 1e-10:
                    best_dist, best_path = dist, path
        return best_path, best_dist

    def local_search_3_opt(self, dm, initial, recursive_seeding=-1):
        def _one_run(start):
            t = list(start)
            n = len(t)
            while True:
                best_t = t[:]
                best_d = self.dist_calc(dm, t)
                improved = False
                for i in range(n - 2):
                    for j in range(i + 1, n - 1):
                        for k in range(j + 1, n):
                            A, B, C, D = t[:i+1], t[i+1:j+1], t[j+1:k+1], t[k+1:]
                            for cand in [
                                A + B[::-1] + C + D,
                                A + B + C[::-1] + D,
                                A + C + B + D,
                                A + B[::-1] + C[::-1] + D,
                                A + C + B[::-1] + D,
                                A + C[::-1] + B + D,
                                A + C[::-1] + B[::-1] + D,
                            ]:
                                d = self.dist_calc(dm, cand)
                                if d < best_d - 1e-10:
                                    best_d, best_t, improved = d, cand, True
                t = best_t
                if not improved:
                    break
            return t, self.dist_calc(dm, t)

        best_path, best_dist = _one_run(self._unpack(initial))
        if recursive_seeding > 0:
            nodes = list(range(dm.shape[0]))
            for _ in range(recursive_seeding):
                random.shuffle(nodes)
                path, dist = _one_run(list(nodes))
                if dist < best_dist - 1e-10:
                    best_dist, best_path = dist, path
        return best_path, best_dist

    def lin_kernighan_helsgaun(self, viewpoints, dm, initial_solution, num_iterations=10, num_candidates=5):
        if num_candidates > 0:
            k = min(num_candidates, len(viewpoints) - 1)
            raw_cands = self.build_candidate_sets(viewpoints, dm, k)
            candidates_set = {i: set(nbrs) for i, nbrs in raw_cands.items()}
            def _local_search(path):
                return self._lk_local_search(path, dm, candidates_set)
        else:
            def _local_search(path):
                return self._local_search(path, dm)

        best_path, best_dist = _local_search(self._unpack(initial_solution))
        n = len(viewpoints)
        max_restarts = max(num_iterations, n)
        patience = max(15, n // 3)
        no_improve = 0
        for _ in range(max_restarts):
            candidate, c_dist = _local_search(self._double_bridge(best_path))
            if c_dist < best_dist - 1e-10:
                best_path, best_dist = candidate, c_dist
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                break
        return best_path, best_dist

    def build_candidate_sets(self, viewpoints, dm, k):
        n = len(viewpoints)
        candidates = {}
        for i in range(n):
            row = sorted((dm[i, j], j) for j in range(n) if j != i)
            candidates[i] = [j for _, j in row[:k]]
        return candidates

    def double_bridge_perturbation(self, tour, num_breaks=4):
        n = len(tour)
        if n <= num_breaks:
            return tour[:]
        if num_breaks == 4:
            return self._double_bridge(tour)
        cuts = sorted(random.sample(range(1, n), num_breaks))
        segs = [tour[a:b] for a, b in zip([0] + cuts, cuts + [n])]
        middle = segs[1:-1]
        random.shuffle(middle)
        return segs[0] + sum(middle, []) + segs[-1]

    def run_greedy(self, viewpoints):
        return self.nearest_neighbors_tsp(viewpoints)

    def run_2opt(self, viewpoints, dm):
        path, _ = self.nearest_neighbors_tsp(viewpoints)
        return self._two_opt(path, dm)

    def run_3opt(self, viewpoints, dm):
        path, _ = self.nearest_neighbors_tsp(viewpoints)
        return self.local_search_3_opt(dm, path)

    def run_ils(self, viewpoints, dm):
        """ILS: multi-start greedy + exhaustive (2-opt + Or-opt) + double-bridge restarts."""
        path, dist = self.nearest_neighbors_tsp(viewpoints)
        return self.lin_kernighan_helsgaun(viewpoints, dm, [path, dist])

    def _lk_local_search(self, path, dm, candidates_set):
        """
        True Lin-Kernighan local search with variable-depth sequential moves.

        For each edge (t[i], t[i+1]), tries a sequential chain:

        Depth 1 — 2-opt with sequential gain criterion:
          Remove (t[i],t[i+1]) and (t[j],t[j+1]).
          Add    (t[i],t[j])   and (t[i+1],t[j+1]).
          Sequential criterion: d(t[i],t[i+1]) - d(t[i],t[j]) > 0  (must hold at depth 1)

        Depth 2 — specific 3-opt S1+S2_rev+S3_rev+S4 (when depth-1 alone fails):
          Remove (t[i],t[i+1]), (t[j],t[j+1]), (t[k],t[k+1]).
          Add    (t[i],t[j]),   (t[i+1],t[k]), (t[j+1],t[k+1]).
          Sequential criterion: ALSO requires g1+d(t[j],t[j+1])-d(t[i+1],t[k]) > 0.
          t[k] must be a candidate of t[i+1] (the "dangling end" from depth 1).

        The depth-2 move is what makes LK distinct: it cannot be produced by any single
        2-opt or Or-opt move. It is the direct product of the LK sequential chain.
        """
        t = list(path)
        n = len(t)

        improved = True
        while improved:
            improved = False

            for i in range(n - 2):
                if improved:
                    break

                t_i  = t[i]
                t_i1 = t[i + 1]
                d1   = dm[t_i, t_i1]   # cost of the edge we want to remove

                for j in range(i + 2, n):
                    if improved:
                        break

                    t_j = t[j]
                    if t_j not in candidates_set.get(t_i, set()):
                        continue

                    # LK sequential gain criterion at depth 1
                    g1 = d1 - dm[t_i, t_j]
                    if g1 <= 0:
                        continue

                    t_j1 = t[j + 1] if j + 1 < n else None

                    # ── Depth-1 close: standard 2-opt ─────────────────────
                    d2_rem = dm[t_j,  t_j1] if t_j1 is not None else 0.0
                    d2_add = dm[t_i1, t_j1] if t_j1 is not None else 0.0
                    if g1 + d2_rem - d2_add > 1e-10:
                        t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
                        improved = True
                        break

                    # ── Depth-2 close: S1+S2_rev+S3_rev+S4 ───────────────
                    if t_j1 is None:
                        continue   # need the j+1 edge to go deeper

                    # Accumulated gain after removing first two edges and
                    # adding the first new edge (t_i → t_j).
                    G2 = g1 + dm[t_j, t_j1]

                    for k in range(j + 2, n):
                        t_k = t[k]
                        if t_k not in candidates_set.get(t_i1, set()):
                            continue

                        # LK sequential gain criterion at depth 2
                        g2 = G2 - dm[t_i1, t_k]
                        if g2 <= 0:
                            continue

                        t_k1 = t[k + 1] if k + 1 < n else None
                        d3_rem = dm[t_k,  t_k1] if t_k1 is not None else 0.0
                        d3_add = dm[t_j1, t_k1] if t_k1 is not None else 0.0
                        if g2 + d3_rem - d3_add > 1e-10:
                            # Apply S1 + S2_rev + S3_rev + S4
                            t = (t[:i + 1]
                                 + t[i + 1:j + 1][::-1]
                                 + t[j + 1:k + 1][::-1]
                                 + t[k + 1:])
                            n = len(t)
                            improved = True
                            break

        return t, self.dist_calc(dm, t)

    def run_lkh(self, viewpoints, dm, n_runs=50):
        """
        Lin-Kernighan solver: n_runs independent ILS passes with candidate-restricted local search.

        Each run starts from a different greedy seed node so the double-bridge
        restarts explore different regions of solution space. The global best
        across all runs is returned.
        """
        n = len(viewpoints)
        k = min(n - 1, 7)
        raw_cands = self.build_candidate_sets(viewpoints, dm, k)
        candidates_set = {i: set(nbrs) for i, nbrs in raw_cands.items()}

        patience = max(15, n // 3)
        global_best_path, global_best_dist = None, float('inf')

        for run in range(n_runs):
            # Spread starting seeds evenly across the node indices
            seed = (run * max(1, n // n_runs)) % n
            path, _ = self._greedy_from(dm, seed)
            path, dist = self._lk_local_search(path, dm, candidates_set)

            no_improve = 0
            for _ in range(max(15, n)):
                candidate, c_dist = self._lk_local_search(
                    self._double_bridge(path), dm, candidates_set)
                if c_dist < dist - 1e-10:
                    path, dist = candidate, c_dist
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= patience:
                    break

            if dist < global_best_dist - 1e-10:
                global_best_path, global_best_dist = path, dist
            self.logger.debug(f'    LKH run {run + 1}/{n_runs}: {dist:.4f} m')

        return global_best_path, global_best_dist

    def _run_algorithm(self, algo, viewpoints, dist_matrix):
        if algo == 'greedy':
            return self.run_greedy(viewpoints)
        elif algo == '2opt':
            return self.run_2opt(viewpoints, dist_matrix)
        elif algo == '3opt':
            return self.run_3opt(viewpoints, dist_matrix)
        elif algo == 'ILS':
            return self.run_ils(viewpoints, dist_matrix)
        elif algo == 'LKH':
            return self.run_lkh(viewpoints, dist_matrix)
        else:
            self.logger.error(f"Unknown TSP algorithm: {algo}")
            return None, None
