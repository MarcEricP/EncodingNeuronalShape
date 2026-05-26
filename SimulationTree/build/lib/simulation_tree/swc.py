import numpy as np
from numba import njit, types
from numba.typed import Dict

i64 = types.int64

@njit(cache=True)
def _assign_ids(parent, alive):
    id_map = Dict.empty(key_type=i64, value_type=i64)
    m = 0
    n = len(parent)
    for i in range(n):  # roots first
        if alive[i] and parent[i] < 0:
            m += 1
            id_map[i] = m
    for i in range(n):  # then others
        if alive[i] and parent[i] >= 0:
            m += 1
            id_map[i] = m
    return id_map, m

@njit(cache=True)
def build_swc_arrays_extended(position, width, parent, alive, birth_time, tip_id, is_contact):
    """
    Returns an array for each swc column:
    id type x y z radius parent  birth_time  width  tip_id
    All arrays are ordered by the new SWC id.
    """
    id_map, m = _assign_ids(parent, alive)

    ids       = np.empty(m, dtype=np.int64)
    types_arr = np.empty(m, dtype=np.int64)
    xs        = np.empty(m, dtype=np.float64)
    ys        = np.empty(m, dtype=np.float64)
    zs        = np.empty(m, dtype=np.float64)
    rs        = np.empty(m, dtype=np.float64)
    pids      = np.empty(m, dtype=np.int64)

    birth_out = np.empty(m, dtype=np.float64)
    width_out = np.empty(m, dtype=np.float64)
    tipid_out = np.empty(m, dtype=np.int64)
    is_contact_out = np.empty(m,dtype = np.int64)

    for i in range(len(parent)):
        if not alive[i]:
            continue
        idx = id_map[i] - 1

        # geometry
        x = position[i].real
        y = position[i].imag
        z = 0.0
        r = 0.5 * width[i]
        if r < 0.0:
            r = 0.0

        # type: root(s) = 1, others = 3
        t = 1 if parent[i] < 0 else 3

        # map parent to SWC id (or -1)
        if parent[i] < 0:
            pid = -1
        else:
            par = parent[i]
            if 0 <= par < len(parent) and alive[par]:
                pid = id_map[par]
            else:
                pid = -1

        ids[idx]       = id_map[i]
        types_arr[idx] = t
        xs[idx]        = x
        ys[idx]        = y
        zs[idx]        = z
        rs[idx]        = r
        pids[idx]      = pid

        birth_out[idx] = birth_time[i]
        width_out[idx] = width[i]
        tipid_out[idx] = tip_id[i]
        is_contact_out[idx] = int(is_contact[idx])

    return ids, types_arr, xs, ys, zs, rs, pids, birth_out, width_out, tipid_out, is_contact_out

def save_tree_to_swc_extended(
    filepath,
    position, width, parent, alive,
    birth_time, tip_id,is_contact,
    header_comment: str = None
):
    """
    Write a SWC:
    columns: id type x y z radius parent  birth_time  width  tip_id
    type: root(s) = 1, others = 3
    """
    (ids, types_arr, xs, ys, zs, rs, pids,
     birth_out, width_out, tipid_out,is_contact_out) = build_swc_arrays_extended(
        position, width, parent, alive, birth_time, tip_id, is_contact
    )

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("# SWC file generated from simulation_tree\n")
        if header_comment:
            for line in header_comment.splitlines():
                f.write(f"# {line}\n")
        f.write("# Columns: id type x y z radius parent  birth_time width tip_id is_contact\n")
        for k in range(len(ids)):
            f.write(
                f"{ids[k]:d} {types_arr[k]:d} "
                f"{xs[k]:.6f} {ys[k]:.6f} {zs[k]:.6f} {rs[k]:.6f} {pids[k]:d} "
                f"{birth_out[k]:.6f} {width_out[k]:.6f} {tipid_out[k]:d} {is_contact_out[k]:d}\n"
            )
