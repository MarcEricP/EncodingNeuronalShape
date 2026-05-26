import math
import numpy as np
from numba import types, njit
from numba.typed import List


i64 = types.int64
f64 = types.float64

@njit(cache=True, fastmath=False)
def _interp_age_multiplier_nb(age: float, ages, mults, use_age_dependence: bool) -> float:
    """
    useful for the case when branching rate depends on parent's age
    """
    if (not use_age_dependence) or len(ages) == 0 or len(mults) == 0:
        return 1.0
    n = len(ages)
    if age <= ages[0]:
        return mults[0]
    if age >= ages[n - 1]:
        return mults[n - 1]
    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ages[mid] <= age:
            lo = mid
        else:
            hi = mid
    x0 = ages[lo]
    x1 = ages[hi]
    y0 = mults[lo]
    y1 = mults[hi]
    if x1 <= x0:
        return y0
    t = (age - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


@njit(cache=True, fastmath=False)
def decide_branches(
    # state (read-only)
    position, parent, alive, alive_nodes,
    # sim controls
    dt: float,
    branching_lambda: float,
    new_branch_length_mean: float,
    new_branch_length_std: float,
    new_branch_angle_kappa:float,
    # lineage counter
    max_tip_id_current: int,
    # age-dependent multiplier
    birth_time,
    simulation_time: float,
    branching_age_ages,
    branching_age_mults,
    use_branching_age_dependence: bool,
    dist_to_center,
    angle_bin,
    front_only: float,
    front_only_bins: int,
    max_r_per_bin,
    max_dist_list,
    # rng seeded via np.random outside
):
    """
    Build external branching events for this step.

    Returns:
        parent_nodes (List[int])
        prop_splits  (List[float]) in (0,1)
        angles       (List[float]) signed radians
        lengths      (List[float]) >= 0
        new_tip_ids  (List[int])   strictly increasing new lineages
        max_tip_id_next (int)      updated counter
    Each tuple (parent_nodes[i], prop_splits[i], angles[i], lengths[i], new_tip_ids[i])
    describes one branch to create during this step.
    """
    parent_nodes = List.empty_list(i64)
    prop_splits  = List.empty_list(f64)
    angles       = List.empty_list(f64)
    lengths      = List.empty_list(f64)
    new_tip_ids  = List.empty_list(i64)

    max_tip_id_next = int(max_tip_id_current)

    for q in range(len(alive_nodes)):
        node_idx = alive_nodes[q]
        if not alive[node_idx]:
            continue
        if front_only >= 0.0:
            if front_only_bins > 0:
                b = angle_bin[node_idx]
                if b >= 0 and b < front_only_bins:
                    cutoff = max_r_per_bin[b] - front_only
                else:
                    cutoff = -1e30
            else:
                cutoff = max_dist_list[0] - front_only
            if dist_to_center[node_idx] < cutoff:
                continue
        p_idx = parent[node_idx]
        if p_idx < 0:
            continue  # skip roots
        dseg = position[node_idx] - position[p_idx]
        L = math.hypot(dseg.real, dseg.imag)
        if L <= 0.0:
            continue

        # Age (current time minus compartment birth)
        age = float(simulation_time) - float(birth_time[node_idx])
        mult = _interp_age_multiplier_nb(
            age, branching_age_ages, branching_age_mults, use_branching_age_dependence
        )
        cur_lambda = float(branching_lambda) * float(mult)
        pois = dt * L * cur_lambda
        n_branches = np.random.poisson(pois)
        if n_branches <= 0:
            continue

        for _ in range(n_branches):
            max_tip_id_next += 1  # reserve a brand-new lineage id

            prop = np.random.random()
            if prop < 1e-4:
                prop = 1e-4
            elif prop > 1.0 - 1e-4:
                prop = 1.0 - 1e-4
            new_L = float(max(np.random.normal(new_branch_length_mean, new_branch_length_std), 0.0))
            vm = np.random.vonmises(0.0, new_branch_angle_kappa, 1)[0]
            angle = (2*np.random.randint(2) - 1) * (vm / 2.0 + np.pi / 2.0)

            parent_nodes.append(int(node_idx))
            prop_splits.append(prop)
            angles.append(angle)
            lengths.append(new_L)
            new_tip_ids.append(int(max_tip_id_next))

    return parent_nodes, prop_splits, angles, lengths, new_tip_ids, max_tip_id_next
