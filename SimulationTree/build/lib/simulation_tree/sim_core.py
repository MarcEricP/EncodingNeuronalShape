import math
import numpy as np
from numba import njit, types
from numba.typed import List, Dict

from . import collisioner as cn  

i64 = types.int64
f64 = types.float64
c128 = types.complex128
k2 = types.UniTuple(i64, 2)
b1 = types.boolean

list_i64 = types.ListType(i64)
list_f64 = types.ListType(f64)
list_c128 = types.ListType(c128)
list_k2 = types.ListType(k2)

children_sig = types.DictType(i64, list_i64)

elem2box_sig = types.DictType(i64, list_k2)

box2elem_sig = types.DictType(k2, list_i64)

# -Small math helpers -
@njit(cache=True, fastmath=False)
def seg_len(a: c128, b: c128) -> float:
    d = b - a
    return math.hypot(d.real, d.imag)

@njit(cache=True, fastmath=False)
def unit_vec(a: c128, b: c128) -> c128:
    d = b - a
    n = math.hypot(d.real, d.imag)
    if n > 1e-4:
        return (d.real / n) + 1j * (d.imag / n)
    return 0.0 + 0.0j

@njit(cache=True, fastmath=False)
def area_parallelogram_style(p_parent: c128, p_child: c128, w_parent: float, w_child: float) -> float:
    return 2.0 * seg_len(p_parent, p_child) * w_child * w_parent

@njit(cache=True, fastmath=False)
def _angle_bin(center: c128, pos: c128, n_bins: int) -> int:
    if n_bins <= 0:
        return 0
    dx = pos.real - center.real
    dy = pos.imag - center.imag
    theta = math.atan2(dy, dx)
    if theta < 0.0:
        theta += 2.0 * math.pi
    b = int(theta * n_bins / (2.0 * math.pi))
    if b >= n_bins:
        b = n_bins - 1
    return b

@njit(cache=True, fastmath=False)
def recompute_max_dist_alive(dist_to_center, alive, max_dist_list):
    maxd = 0.0
    for i in range(len(dist_to_center)):
        if alive[i] and dist_to_center[i] > maxd:
            maxd = dist_to_center[i]
    max_dist_list[0] = maxd

@njit(cache=True, fastmath=False)
def recompute_max_r_per_bin(dist_to_center, angle_bin, alive, n_bins: int, max_r_per_bin):
    for b in range(n_bins):
        max_r_per_bin[b] = 0.0
    for i in range(len(dist_to_center)):
        if not alive[i]:
            continue
        b = angle_bin[i]
        if b < 0 or b >= n_bins:
            continue
        r = dist_to_center[i]
        if r > max_r_per_bin[b]:
            max_r_per_bin[b] = r

@njit(cache=True, fastmath=False)
def init_angle_bins(center: c128, position, angle_bin, n_bins: int):
    for i in range(len(position)):
        angle_bin[i] = _angle_bin(center, position[i], n_bins)

# -Boundary helpers -
@njit(cache=True, fastmath=False)
def _is_inside_bounds(z: c128, x0: float, y_min: float, y_max: float, mode: int) -> b1:
    if mode == 0:
        return True
    x = z.imag
    y = z.real
    if x < x0:
        return False
    return (y >= y_min) and (y <= y_max)

@njit(cache=True, fastmath=False)
def _is_on_trunk(z: c128, x0: float, y_min: float, y_max: float, eps: float = 1e-6) -> b1:
    x = z.imag
    y = z.real
    return (abs(x - x0) <= eps) and (y >= y_min) and (y <= y_max)

@njit(cache=True, fastmath=False)
def _clip_target_to_bounds(root_pos: c128, target_pos: c128,
                           x0: float, y_min: float, y_max: float,
                           mode: int):
    """
    Clip target to the allowed region. Returns (target_pos, boundary_contact).
    mode: 0=off, 1=hard y bounds.
    """
    if mode == 0:
        return target_pos, False

    boundary_contact = False


    if (target_pos.imag >= x0) and (target_pos.real >= y_min) and (target_pos.real <= y_max):
        return target_pos, False

    t_min = 1e30

    if target_pos.imag < x0:
        dx = target_pos.imag - root_pos.imag
        if abs(dx) > 1e-12:
            t = (x0 - root_pos.imag) / dx
            if (t >= 0.0) and (t <= 1.0) and (t < t_min):
                t_min = t

    if target_pos.real < y_min:
        dy = target_pos.real - root_pos.real
        if abs(dy) > 1e-12:
            t = (y_min - root_pos.real) / dy
            if (t >= 0.0) and (t <= 1.0) and (t < t_min):
                t_min = t

    if target_pos.real > y_max:
        dy = target_pos.real - root_pos.real
        if abs(dy) > 1e-12:
            t = (y_max - root_pos.real) / dy
            if (t >= 0.0) and (t <= 1.0) and (t < t_min):
                t_min = t

    if t_min < 1e20:
        target_pos = root_pos + t_min * (target_pos - root_pos)
    else:
        target_pos = root_pos

    return target_pos, True


def init_state_primary(box_size_contact: float, new_branch_width: float):
    """
    Build a minimal tree with a root and one child.
    """
    # Node parallel arrays (per-compartment)
    position = List.empty_list(c128)
    width = List.empty_list(f64)
    birth_time = List.empty_list(f64)
    tip_id = List.empty_list(i64)            # lineage id (branch tip id)
    is_contact = List.empty_list(types.boolean)
    is_protected_scaffold = List.empty_list(types.boolean)
    parent = List.empty_list(i64)

    # children adjacency
    children = Dict.empty(key_type=i64, value_type=list_i64)

    # bookkeeping
    tip_list = List.empty_list(i64)      # list of node indices that are current tips
    alive = List.empty_list(types.boolean)
    alive_nodes = List.empty_list(i64)
    alive_pos = List.empty_list(i64)
    dist_to_center = List.empty_list(f64)
    angle_bin = List.empty_list(i64)
    max_dist_list = List.empty_list(f64)
    max_dist_list.append(0.0)

    # Collisioner typed dicts
    unit_size, elem2box, box2elem = cn.initialize(box_size_contact)

    # Create root (idx 0)
    root_idx = 0
    position.append(15.0 + 0.0j)
    width.append(0.5)
    birth_time.append(-64)
    tip_id.append(1)         # initial lineage id
    is_contact.append(False)
    is_protected_scaffold.append(False)
    parent.append(-1)
    alive.append(True)
    alive_nodes.append(root_idx)
    alive_pos.append(0)
    children[root_idx] = List.empty_list(i64)

    # Create first tip (idx 1), inherits lineage 1
    tip_idx = 1
    position.append(-15.0 + 0.0j)
    width.append(new_branch_width)
    birth_time.append(-64)
    tip_id.append(1)
    is_contact.append(False)
    is_protected_scaffold.append(False)
    parent.append(root_idx)
    alive.append(True)
    alive_nodes.append(tip_idx)
    alive_pos.append(1)
    children[tip_idx] = List.empty_list(i64)

    # hook edges
    children[root_idx].append(tip_idx)
    tip_list.append(tip_idx)

    # collisioner uses compartment ids (node indices)
    cn.add_segment(unit_size, box2elem, elem2box,
                   position[root_idx], position[tip_idx],
                   width[root_idx], width[tip_idx],
                   tip_idx)

    simulation_time = 0.0
    max_tip_id = 1  # lineage counter
    center = (position[root_idx] + position[tip_idx]) * 0.5
    for i in range(len(position)):
        d = seg_len(center, position[i])
        dist_to_center.append(d)
        angle_bin.append(0)
        if d > max_dist_list[0]:
            max_dist_list[0] = d

    return (unit_size, elem2box, box2elem,
            position, width, birth_time, tip_id, is_contact, is_protected_scaffold, parent, children, alive,
            alive_nodes, alive_pos,
            tip_list, simulation_time, max_tip_id, root_idx,
            center, dist_to_center, angle_bin, max_dist_list)

def init_state_vertical_branch(box_size_contact: float, new_branch_width: float,
                               x0: float, y_min: float, y_max: float):
    """
    Initialize a vertical branch at x=x0 from y_min to y_max.
    The trunk is fixed: no initial tips are added to tip_list.
    """
    position = List.empty_list(c128)
    width = List.empty_list(f64)
    birth_time = List.empty_list(f64)
    tip_id = List.empty_list(i64)
    is_contact = List.empty_list(types.boolean)
    is_protected_scaffold = List.empty_list(types.boolean)
    parent = List.empty_list(i64)

    children = Dict.empty(key_type=i64, value_type=list_i64)
    tip_list = List.empty_list(i64)
    alive = List.empty_list(types.boolean)
    alive_nodes = List.empty_list(i64)
    alive_pos = List.empty_list(i64)
    dist_to_center = List.empty_list(f64)
    angle_bin = List.empty_list(i64)
    max_dist_list = List.empty_list(f64)
    max_dist_list.append(0.0)

    unit_size, elem2box, box2elem = cn.initialize(box_size_contact)

    root_idx = 0
    position.append(y_min + 1j * x0)
    width.append(0.5)
    birth_time.append(-64.0)
    tip_id.append(0)
    is_contact.append(False)
    is_protected_scaffold.append(True)
    parent.append(-1)
    alive.append(True)
    alive_nodes.append(root_idx)
    alive_pos.append(0)
    children[root_idx] = List.empty_list(i64)

    tip_idx = 1
    position.append(y_max + 1j * x0)
    width.append(new_branch_width)
    birth_time.append(-64.0)
    tip_id.append(0)
    is_contact.append(False)
    is_protected_scaffold.append(True)
    parent.append(root_idx)
    alive.append(True)
    alive_nodes.append(tip_idx)
    alive_pos.append(1)
    children[tip_idx] = List.empty_list(i64)

    children[root_idx].append(tip_idx)

    cn.add_segment(unit_size, box2elem, elem2box,
                   position[root_idx], position[tip_idx],
                   width[root_idx], width[tip_idx],
                   tip_idx)

    simulation_time = 0.0
    max_tip_id = 0
    center = (position[root_idx] + position[tip_idx]) * 0.5
    for i in range(len(position)):
        d = seg_len(center, position[i])
        dist_to_center.append(d)
        angle_bin.append(0)
        if d > max_dist_list[0]:
            max_dist_list[0] = d

    return (unit_size, elem2box, box2elem,
            position, width, birth_time, tip_id, is_contact, is_protected_scaffold, parent, children, alive,
            alive_nodes, alive_pos,
            tip_list, simulation_time, max_tip_id, root_idx,
            center, dist_to_center, angle_bin, max_dist_list)

def init_state_ring(box_size_contact: float,
                    new_branch_width: float,
                    radius: float,
                    n_nodes: int = 32,
                    center: complex = 0.0 + 0.0j,
                    root_angle: float = 0.0,
                    close_in_topology: bool = False):
    """
    Initialize a ring-shaped scaffold with NO initial tips.
    - Geometry: n_nodes equally spaced on a circle of given radius (closed loop).
    - Rooting:
         Adds a central root node at center (idx 0).
         The first ring node is a child of this center root.
    - Topology:
         By default (close_in_topology=False): ring nodes form a simple chain
          (ring0 -> ring1 -> ... -> ring(n-1)), attached to center root.
          We still close the geometry in the
          collisioner (segment ring(n-1) -> ring0) so the shape is a ring, but we avoid a
          cycle in the children graph. This keeps typical tree traversals safe.
         If close_in_topology=True: also add children[ring(n-1)].append(ring0), creating
          a cycle in children (use only if your logic never DFS's children blindly).
    - tip_list is intentionally left EMPTY so growth must come from side-branching.

    Returns the same tuple as init_state_primary.
    """

    # Node parallel arrays (per-compartment)
    position   = List.empty_list(c128)
    width      = List.empty_list(f64)
    birth_time = List.empty_list(f64)
    tip_id     = List.empty_list(i64)              # lineage id (branch tip id)
    is_contact = List.empty_list(types.boolean)
    is_protected_scaffold = List.empty_list(types.boolean)
    parent     = List.empty_list(i64)

    # children adjacency
    children = Dict.empty(key_type=i64, value_type=list_i64)

    # bookkeeping
    tip_list = List.empty_list(i64)                
    alive    = List.empty_list(types.boolean)
    alive_nodes = List.empty_list(i64)
    alive_pos = List.empty_list(i64)
    dist_to_center = List.empty_list(f64)
    angle_bin = List.empty_list(i64)
    max_dist_list = List.empty_list(f64)
    max_dist_list.append(0.0)

    # Collisioner typed dicts
    unit_size, elem2box, box2elem = cn.initialize(box_size_contact)

    # Center root (idx 0): structural root, not part of the protected ring seam.
    root_idx = 0
    position.append(center)
    width.append(0.5)
    birth_time.append(-64.0)
    tip_id.append(0)
    is_contact.append(False)
    is_protected_scaffold.append(False)
    parent.append(-1)
    alive.append(True)
    alive_nodes.append(root_idx)
    alive_pos.append(root_idx)
    children[root_idx] = List.empty_list(i64)

    # Angles for ring nodes
    two_pi = 2.0 * np.pi
    angles = [root_angle + two_pi * (k / n_nodes) for k in range(n_nodes)]

    # Create protected ring nodes on the circle (indices 1..n_nodes)
    for k, th in enumerate(angles):
        idx = k + 1
        z = center + radius * (np.cos(th) + 1j * np.sin(th))
        position.append(z)
        width.append(new_branch_width)
        birth_time.append(-64.0)
        tip_id.append(0)    # no initial tip lineages
        is_contact.append(False)
        is_protected_scaffold.append(True)
        parent.append(root_idx if k == 0 else (idx - 1))
        alive.append(True)
        alive_nodes.append(idx)
        alive_pos.append(idx)
        children[idx] = List.empty_list(i64)

    first_ring_idx = 1
    last_ring_idx = n_nodes

    # Attach first ring node to center root (topology + collisioner)
    children[root_idx].append(first_ring_idx)
    cn.add_segment(unit_size, box2elem, elem2box,
                   position[root_idx], position[first_ring_idx],
                   width[root_idx], width[first_ring_idx],
                   first_ring_idx)

    # Build ring-node chain ring0->ring1->...->ring(n-1)
    for k in range(1, n_nodes):
        children[k].append(k + 1)

    # Geometric segments for collisioner along ring-node chain
    for k in range(first_ring_idx, last_ring_idx):
        cn.add_segment(unit_size, box2elem, elem2box,
                       position[k], position[k + 1],
                       width[k], width[k + 1],
                       k + 1)                     # element id = head node

    # Close the geometry (ring end -> first ring node) so it looks like a ring for contacts
    # Use id=0 (center root id), which is otherwise unused as a topological edge id.
    cn.add_segment(unit_size, box2elem, elem2box,
                   position[last_ring_idx], position[first_ring_idx],
                   width[last_ring_idx], width[first_ring_idx],
                   0)

    # Optionally also close the topology (cycle in children)
    if close_in_topology:
        children[last_ring_idx].append(first_ring_idx)
        # NOTE: parent[first_ring_idx] stays root_idx to keep a unique parent array.

    # No initial tips
    simulation_time = 0.0
    max_tip_id = 0

    init_center = 0.0 + 0.0j
    for i in range(len(position)):
        init_center += position[i]
    if len(position) > 0:
        init_center /= len(position)
    for i in range(len(position)):
        d = seg_len(init_center, position[i])
        dist_to_center.append(d)
        angle_bin.append(0)
        if d > max_dist_list[0]:
            max_dist_list[0] = d
    return (unit_size, elem2box, box2elem,
            position, width, birth_time, tip_id, is_contact, is_protected_scaffold, parent, children, alive,
            alive_nodes, alive_pos,
            tip_list, simulation_time, max_tip_id, root_idx,
            init_center, dist_to_center, angle_bin, max_dist_list)


def init_state_from_swc_string(box_size_contact: float, new_branch_width_default: float, swc_text: str):
    """
    Build a full simulation state from an SWC-formatted string.
    Supports classic 7-column SWC and this project's extended 10-column format:
      id type x y z radius parent  [birth_time width tip_id]
    Rules:
    - Positions use (x,y) -> complex; z is ignored.
    - width = 2*radius if 'width' column missing or <= 0.
    - If 'tip_id' column missing, we assign a UNIQUE lineage id to each branch tip;
      internal nodes get tip_id = 0 (not used for lineage). Growth inherits the TIP's lineage id (see grow_tip).
    - Active tips for the simulation are all leaves in the parsed tree.
    """
    # Containers (same as init_state_primary), Python side (not jitted)
    position = List.empty_list(c128)
    width    = List.empty_list(f64)
    birth_time = List.empty_list(f64)
    tip_id_arr = List.empty_list(i64)
    is_contact = List.empty_list(types.boolean)
    is_protected_scaffold = List.empty_list(types.boolean)
    parent     = List.empty_list(i64)
    children   = Dict.empty(key_type=i64, value_type=list_i64)
    alive      = List.empty_list(types.boolean)
    alive_nodes = List.empty_list(i64)
    alive_pos = List.empty_list(i64)
    tip_list   = List.empty_list(i64)
    dist_to_center = List.empty_list(f64)
    angle_bin = List.empty_list(i64)
    max_dist_list = List.empty_list(f64)
    max_dist_list.append(0.0)

    # Collisioner
    unit_size, elem2box, box2elem = cn.initialize(box_size_contact)

    # Parse SWC string
    lines = [ln.strip() for ln in swc_text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if not lines:
        # fallback to primary init if empty
        return init_state_primary(box_size_contact, new_branch_width_default)

    # Read rows and sort by id to ensure deterministic index order
    rows = []
    for ln in lines:
        parts = ln.split()
        if len(parts) < 7:
            continue
        try:
            idx = int(float(parts[0]))
            typ = int(float(parts[1]))
            x   = float(parts[2]); y = float(parts[3]); z = float(parts[4])
            r   = float(parts[5])
            pid = int(float(parts[6]))
            bt  = None; ww = None; tid = None
            if len(parts) >= 8:
                bt = float(parts[7])
            if len(parts) >= 9:
                ww = float(parts[8])
            if len(parts) >= 10:
                tid = int(float(parts[9]))
            rows.append((idx, typ, x, y, z, r, pid, bt, ww, tid))
        except Exception:
            # skip malformed
            continue
    rows.sort(key=lambda r: r[0])

    # Map SWC id -> new index (0..n-1) in the order of sorted ids
    id2idx = {rid: k for k, (rid, *_rest) in enumerate(rows)}

    # Pre-allocate per-index python lists first for convenience
    n = len(rows)
    pos_py = [0j] * n
    wid_py = [0.0] * n
    birth_py = [0.0] * n
    tipid_py = [0] * n
    par_py   = [-1] * n
    children_py = [[] for _ in range(n)]
    alive_py = [True] * n

    for k, (rid, typ, x, y, z, r, pid, bt, ww, tid) in enumerate(rows):
        pos_py[k] = complex(x, y)
        # width: prefer explicit width; else 2*radius
        w = ww if (ww is not None and ww > 0.0) else max(2.0 * r, 0.0)
        wid_py[k] = w if w > 0.0 else (new_branch_width_default if new_branch_width_default > 0 else 0.5)
        birth_py[k] = bt if (bt is not None) else -64
        # parent mapping: SWC parent_id of -1 -> no parent; otherwise map via id2idx
        if pid is None or pid < 0:
            par_py[k] = -1
        else:
            par_py[k] = id2idx.get(pid, -1)
        # tip_id: keep if provided; else 0 for now (we'll set tips after)
        tipid_py[k] = int(tid) if (tid is not None) else 0

    # Build children lists
    for i in range(n):
        p = par_py[i]
        if p >= 0:
            children_py[p].append(i)

    # Determine branch tips (leaves)
    leaves = [i for i in range(n) if len(children_py[i]) == 0 and par_py[i] >= 0 and alive_py[i]]

    # If no explicit tip_id provided, assign a unique lineage id per leaf
    if all(tid == 0 for tid in tipid_py):
        cur = 0
        for leaf in leaves:
            cur += 1
            tipid_py[leaf] = cur
        max_tip_id = cur
    else:
        # Otherwise, compute max over provided positive ids (fallback to #leaves if none)
        pos_ids = [tid for tid in tipid_py if tid is not None]
        max_tip_id = max(pos_ids) if pos_ids else len(leaves)

    # Compute centroid from initial alive positions (once at init)
    init_center = 0.0 + 0.0j
    alive_count = 0
    for i in range(n):
        if alive_py[i]:
            init_center += pos_py[i]
            alive_count += 1
    if alive_count > 0:
        init_center /= alive_count

    # Move from python lists to numba typed containers (same layout as primary init)
    for i in range(n):
        position.append(pos_py[i])
        width.append(wid_py[i])
        birth_time.append(birth_py[i])
        tip_id_arr.append(tipid_py[i])
        is_contact.append(False)
        is_protected_scaffold.append(False)
        parent.append(par_py[i])
        alive.append(alive_py[i])
        if alive_py[i]:
            alive_nodes.append(i)
            alive_pos.append(len(alive_nodes) - 1)
        else:
            alive_pos.append(-1)
        children[i] = List.empty_list(i64)
        d = seg_len(init_center, pos_py[i])
        dist_to_center.append(d)
        angle_bin.append(0)
        if d > max_dist_list[0]:
            max_dist_list[0] = d
    for i in range(n):
        for ch in children_py[i]:
            children[i].append(ch)
    
    # Build collision segments (for every non-root alive node)
    for i in range(n):
        if not alive[i]: 
            continue
        p = parent[i]
        if p >= 0 and alive[p]:
            cn.add_segment(unit_size, box2elem, elem2box,
                           position[p], position[i], width[p], width[i], i)

    # Active tips list = leaves (alive, nonroot)
    for leaf in leaves:
        tip_list.append(leaf)

    # Root index: first node with parent < 0 (if multiple, pick first)
    root_idx = next((i for i in range(n) if parent[i] < 0 and alive[i]), 0)

    simulation_time = 0.0
    return (unit_size, elem2box, box2elem,
            position, width, birth_time, tip_id_arr, is_contact, is_protected_scaffold, parent, children, alive,
            alive_nodes, alive_pos,
            tip_list, simulation_time, max_tip_id, root_idx,
            init_center, dist_to_center, angle_bin, max_dist_list)

def compact_state(
    unit_size,
    position, width, birth_time, tip_id, is_contact, is_protected_scaffold, parent, children, alive,
    alive_nodes, alive_pos,
    tip_list,
    center: complex,
    front_only_bins: int,
):
    """
    Rebuild all node-indexed structures from currently alive nodes only.
    Returns:
      unit_size, elem2box, box2elem,
      position, width, birth_time, tip_id, is_contact, is_protected_scaffold, parent, children, alive,
      alive_nodes, alive_pos, tip_list, root_idx, dist_to_center, angle_bin, max_dist_list
    """
    n_alive = len(alive_nodes)
    if n_alive == 0:
        unit_size_new, elem2box_new, box2elem_new = cn.initialize(unit_size)
        position_new = List.empty_list(c128)
        width_new = List.empty_list(f64)
        birth_time_new = List.empty_list(f64)
        tip_id_new = List.empty_list(i64)
        is_contact_new = List.empty_list(types.boolean)
        is_protected_scaffold_new = List.empty_list(types.boolean)
        parent_new = List.empty_list(i64)
        alive_new = List.empty_list(types.boolean)
        alive_nodes_new = List.empty_list(i64)
        alive_pos_new = List.empty_list(i64)
        children_new = Dict.empty(key_type=i64, value_type=list_i64)
        tip_list_new = List.empty_list(i64)
        dist_to_center_new = List.empty_list(f64)
        angle_bin_new = List.empty_list(i64)
        max_dist_list_new = List.empty_list(f64)
        max_dist_list_new.append(0.0)
        return (
            unit_size_new, elem2box_new, box2elem_new,
            position_new, width_new, birth_time_new, tip_id_new, is_contact_new, is_protected_scaffold_new, parent_new, children_new, alive_new,
            alive_nodes_new, alive_pos_new, tip_list_new, -1, dist_to_center_new, angle_bin_new, max_dist_list_new
        )

    old_to_new = {}
    for new_idx, old_idx in enumerate(alive_nodes):
        old_to_new[int(old_idx)] = int(new_idx)

    position_new = List.empty_list(c128)
    width_new = List.empty_list(f64)
    birth_time_new = List.empty_list(f64)
    tip_id_new = List.empty_list(i64)
    is_contact_new = List.empty_list(types.boolean)
    is_protected_scaffold_new = List.empty_list(types.boolean)
    parent_new = List.empty_list(i64)
    alive_new = List.empty_list(types.boolean)
    alive_nodes_new = List.empty_list(i64)
    alive_pos_new = List.empty_list(i64)
    children_new = Dict.empty(key_type=i64, value_type=list_i64)
    dist_to_center_new = List.empty_list(f64)
    angle_bin_new = List.empty_list(i64)
    max_dist_list_new = List.empty_list(f64)
    max_dist_list_new.append(0.0)

    for new_idx, old_idx in enumerate(alive_nodes):
        oi = int(old_idx)
        position_new.append(position[oi])
        width_new.append(width[oi])
        birth_time_new.append(birth_time[oi])
        tip_id_new.append(tip_id[oi])
        # run_simu uses is_contact as per-tip array, keep compact state node flags neutral
        is_contact_new.append(False)
        is_protected_scaffold_new.append(is_protected_scaffold[oi])

        p_old = parent[oi]
        if p_old < 0:
            parent_new.append(-1)
        else:
            p_new = old_to_new.get(int(p_old), -1)
            parent_new.append(p_new)

        alive_new.append(True)
        alive_nodes_new.append(new_idx)
        alive_pos_new.append(new_idx)
        children_new[new_idx] = List.empty_list(i64)

        d = seg_len(center, position[oi])
        dist_to_center_new.append(d)
        angle_bin_new.append(_angle_bin(center, position[oi], front_only_bins))
        if d > max_dist_list_new[0]:
            max_dist_list_new[0] = d

    for new_idx, old_idx in enumerate(alive_nodes):
        oi = int(old_idx)
        old_kids = children[oi]
        new_kids = List.empty_list(i64)
        for j in range(len(old_kids)):
            ch_old = int(old_kids[j])
            if ch_old in old_to_new:
                new_kids.append(old_to_new[ch_old])
        children_new[new_idx] = new_kids

    tip_list_new = List.empty_list(i64)
    seen_tip = set()
    for j in range(len(tip_list)):
        t_old = int(tip_list[j])
        t_new = old_to_new.get(t_old, -1)
        if t_new >= 0 and (t_new not in seen_tip):
            seen_tip.add(t_new)
            tip_list_new.append(t_new)

    root_idx_new = -1
    for i in range(len(parent_new)):
        if parent_new[i] < 0:
            root_idx_new = i
            break

    unit_size_new, elem2box_new, box2elem_new = cn.initialize(unit_size)
    for child_idx in range(len(parent_new)):
        p = parent_new[child_idx]
        if p >= 0:
            cn.add_segment(
                unit_size_new, box2elem_new, elem2box_new,
                position_new[p], position_new[child_idx],
                width_new[p], width_new[child_idx],
                child_idx
            )

    # Preserve ring geometric closure after compaction.
    # Ring init uses an extra non-topological edge (protected ring leaf -> first protected ring node).
    # Compaction rebuilds only topological parent edges, so re-add this seam when detected.
    first_protected_idx = -1
    protected_leaf_idx = -1
    n_protected = 0
    for i in range(len(parent_new)):
        if not is_protected_scaffold_new[i]:
            continue
        n_protected += 1
        p = parent_new[i]
        if (first_protected_idx < 0) and ((p < 0) or (not is_protected_scaffold_new[p])):
            first_protected_idx = i
        if (len(children_new[i]) == 0) and (protected_leaf_idx < 0):
            protected_leaf_idx = i
    seam_id = root_idx_new if root_idx_new >= 0 else 0
    if (n_protected >= 3) and (first_protected_idx >= 0) and (protected_leaf_idx >= 0) and (protected_leaf_idx != first_protected_idx):
        cn.add_segment(
            unit_size_new, box2elem_new, elem2box_new,
            position_new[protected_leaf_idx], position_new[first_protected_idx],
            width_new[protected_leaf_idx], width_new[first_protected_idx],
            seam_id
        )

    return (
        unit_size_new, elem2box_new, box2elem_new,
        position_new, width_new, birth_time_new, tip_id_new, is_contact_new, is_protected_scaffold_new, parent_new, children_new, alive_new,
        alive_nodes_new, alive_pos_new, tip_list_new, root_idx_new, dist_to_center_new, angle_bin_new, max_dist_list_new
    )


@njit(cache=True)
def is_root(parent, idx: int) -> bool:
    return parent[idx] == -1

@njit(cache=True)
def is_branch_tip(children, idx: int) -> bool:
    return len(children[idx]) == 0

@njit(cache=True)
def is_branching_point(children, idx: int) -> bool:
    return len(children[idx]) > 1

@njit(cache=True)
def _register_alive_node(alive_nodes, alive_pos, idx: int):
    alive_pos.append(len(alive_nodes))
    alive_nodes.append(idx)

@njit(cache=True)
def _unregister_alive_node(alive_nodes, alive_pos, idx: int):
    pos = alive_pos[idx]
    if pos < 0:
        return
    last_pos = len(alive_nodes) - 1
    last_idx = alive_nodes[last_pos]
    alive_nodes[pos] = last_idx
    alive_pos[last_idx] = pos
    alive_nodes.pop()
    alive_pos[idx] = -1

@njit(cache=True)
def add_child_node(unit_size, box2elem, elem2box,
                   position, width, birth_time, tip_id, is_contact, is_protected_scaffold, parent, children, alive,
                   alive_nodes, alive_pos,
                   center: c128, dist_to_center, angle_bin, max_dist_list, front_only_bins: int,
                   parent_idx: int, new_payload_position: c128, new_payload_width: float,
                   new_payload_birth_time: float, new_payload_tip_id: int) -> int:
    """
    Append a child node to parent_idx, update collision maps, return child idx.
    The child inherits the tip_id lineage passed as new_payload_tip_id.
    """
    idx = len(position)
    position.append(new_payload_position)
    width.append(new_payload_width)
    birth_time.append(new_payload_birth_time)
    tip_id.append(new_payload_tip_id)     # lineage assignment
    is_contact.append(False)
    is_protected_scaffold.append(False)
    parent.append(parent_idx)
    alive.append(True)
    _register_alive_node(alive_nodes, alive_pos, idx)
    children[idx] = List.empty_list(i64)
    d = seg_len(center, new_payload_position)
    dist_to_center.append(d)
    angle_bin.append(_angle_bin(center, new_payload_position, front_only_bins))
    if d > max_dist_list[0]:
        max_dist_list[0] = d

    # add child in adjacency
    children[parent_idx].append(idx)

    # collisioner segment = edge parent_idx -> idx (use compartment id idx)
    cn.add_segment(unit_size, box2elem, elem2box,
                   position[parent_idx], position[idx],
                   width[parent_idx], width[idx], idx)
    return idx

@njit(cache=True)
def delete_node(unit_size, box2elem, elem2box,
                position, width, birth_time, tip_id, is_contact, is_protected_scaffold, parent, children, alive,
                alive_nodes, alive_pos,
                idx: int):
    """
    Remove a leaf node (assume it's a tip), update collisioner and parent adjacency.
    """
    if not alive[idx]:
        return
    if is_protected_scaffold[idx]:
        return
    p = parent[idx]
    if p >= 0:
        cn.delete_segment(box2elem, elem2box, idx)
        newkids = List.empty_list(i64)
        for k in range(len(children[p])):
            if children[p][k] != idx:
                newkids.append(children[p][k])
        children[p] = newkids
    alive[idx] = False
    _unregister_alive_node(alive_nodes, alive_pos, idx)

# -Split node -
@njit(cache=True, fastmath=False)
def split_node(unit_size, box2elem, elem2box,
               position, width, birth_time, tip_id, is_contact, is_protected_scaffold, parent, children, alive,
               alive_nodes, alive_pos,
               center: c128, dist_to_center, angle_bin, max_dist_list, front_only_bins: int,
               node_idx_split: int, proportion_length_split: float,
               min_dist_nodes: float = 0.0, dump_end_node: bool = False) -> int:
    """
    Returns middle_node_idx (or parent/end idx in edge cases).
    The new middle node inherits the lineage tip_id of the edge's end node.
    """
    end_idx = node_idx_split
    p_idx = parent[end_idx]
    assert p_idx >= 0
    if not alive[end_idx] or not alive[p_idx]:
        return end_idx

    seg_L = seg_len(position[p_idx], position[end_idx])
    if not dump_end_node:
        if proportion_length_split * seg_L < min_dist_nodes:
            return p_idx
        if (1.0 - proportion_length_split) * seg_L < min_dist_nodes:
            return end_idx

    t = proportion_length_split
    middle_pos = (1.0 - t) * position[p_idx] + t * position[end_idx]
    middle_width = (1.0 - t) * width[p_idx] + t * width[end_idx]
    middle_birth = (1.0 - t) * birth_time[p_idx] + t * birth_time[end_idx]

    cn.delete_segment(box2elem, elem2box, end_idx)

    newkids = List.empty_list(i64)
    for k in range(len(children[p_idx])):
        if children[p_idx][k] != end_idx:
            newkids.append(children[p_idx][k])
    children[p_idx] = newkids

    # middle inherits lineage of the end node
    mid_idx = add_child_node(
        unit_size, box2elem, elem2box,
        position, width, birth_time, tip_id, is_contact, is_protected_scaffold, parent, children, alive,
        alive_nodes, alive_pos,
        center, dist_to_center, angle_bin, max_dist_list, front_only_bins,
        p_idx, middle_pos, middle_width, middle_birth, tip_id[end_idx]
    )

    if dump_end_node:
        delete_node(
            unit_size, box2elem, elem2box,
            position, width, birth_time, tip_id, is_contact, is_protected_scaffold, parent, children, alive,
            alive_nodes, alive_pos,
            end_idx
        )
    else:
        parent[end_idx] = mid_idx
        children[mid_idx].append(end_idx)
        cn.add_segment(
            unit_size, box2elem, elem2box,
            position[mid_idx], position[end_idx],
            width[mid_idx], width[end_idx], end_idx
    )

    return mid_idx

# -Grow tip -
@njit(cache=True, fastmath=False)
def grow_tip(unit_size, box2elem, elem2box,
             position, width, birth_time, tip_id, is_contact, is_protected_scaffold, parent, children, alive,
             alive_nodes, alive_pos,
             center: c128, dist_to_center, angle_bin, max_dist_list, front_only_bins: int,
             node_idx: int, target_position: c128, list_exclusion: np.ndarray,
             birth_time_new: float, width_new: float,
             degree_exempt_collision: int,
             proba_overlap: float, min_dist_nodes: float = 0.0, max_dist_node: float = 0.1):
    """
    Returns (new_tip_node_idx, is_contact_flag, overlap_flag).
    Child nodes inherit the same lineage tip_id as their parent.
    """
    p_idx = node_idx
    root_position = position[p_idx]
    root_width = width[p_idx]

    pot = cn.get_potential_collisions(unit_size, box2elem,
                                      root_position, target_position,
                                      root_width, width_new, 1)  # polygon

    min_dist = 1e300
    final_intersection = target_position
    overlap = False

    # Build exemption list as a depth-k neighborhood starting from the parent,
    # walking only down the children graph.
    exempt_nodes = List.empty_list(i64)
    if degree_exempt_collision > 0:
        if p_idx >= 0:
            # visited dedup to avoid cycles / repeated nodes in dense branching
            visited = Dict.empty(key_type=i64, value_type=b1)
            frontier = List.empty_list(i64)
            frontier.append(p_idx)
            visited[p_idx] = True
            depth = 0
            while depth <= degree_exempt_collision and len(frontier) > 0:
                next_frontier = List.empty_list(i64)
                for i in range(len(frontier)):
                    node = frontier[i]
                    exempt_nodes.append(node)
                    kids = children[node]
                    for s in range(len(kids)):
                        child = kids[s]
                        if child in visited:
                            continue
                        visited[child] = True
                        exempt_nodes.append(child)
                        if depth < degree_exempt_collision:
                            next_frontier.append(child)
                frontier = next_frontier
                depth += 1

    for k in range(len(pot)):
        coll_node_idx = pot[k]
        skip = False
        for e in range(len(list_exclusion)):
            if coll_node_idx == list_exclusion[e]:
                skip = True
                break
        if skip:
            continue

        if degree_exempt_collision > 0:
            for e in range(len(exempt_nodes)):
                if coll_node_idx == exempt_nodes[e]:
                    skip = True
                    break
            if skip:
                continue

        coll_parent_idx = parent[coll_node_idx]
        if coll_parent_idx < 0:
            continue

        inter = cn.fine_intersection_polygon(root_position, target_position, root_width, width_new,
                                             position[coll_parent_idx], position[coll_node_idx],
                                             width[coll_parent_idx], width[coll_node_idx])

        if not (math.isfinite(inter.real) and math.isfinite(inter.imag)):
            L = seg_len(root_position, target_position)
            if L > 0.0:
                touch_eps = 1e-7
                for endpoint, _w_ep in ((position[coll_parent_idx], width[coll_parent_idx]),
                                        (position[coll_node_idx], width[coll_node_idx])):
                    dsum = seg_len(root_position, endpoint) + seg_len(endpoint, target_position)
                    if abs(dsum - L) <= touch_eps:
                        inter = endpoint
                        d = seg_len(root_position, inter)
                        if d < min_dist:
                            rnd = np.random.random()
                            if rnd > proba_overlap:
                                min_dist = d
                                final_intersection = inter
                                overlap = False
                            else:
                                overlap = True
                        continue
            continue

        d = seg_len(root_position, inter)
        if d < min_dist:
            rnd = np.random.random()
            if rnd > proba_overlap:
                min_dist = d
                final_intersection = inter
                overlap = False
            else:
                overlap = True

    is_cont = (min_dist < 1e300)

    parent_dist = 0.0
    ppar = parent[p_idx]
    if ppar >= 0:
        parent_dist = seg_len(position[p_idx], position[ppar])

    new_length = seg_len(root_position, final_intersection)

    if (new_length >= min_dist_nodes) or ((parent_dist > max_dist_node) and (new_length > 0.0)):
        # create child (inherits lineage)
        new_tip_idx = add_child_node(unit_size, box2elem, elem2box,
                                     position, width, birth_time, tip_id, is_contact, is_protected_scaffold, parent, children, alive,
                                     alive_nodes, alive_pos,
                                     center, dist_to_center, angle_bin, max_dist_list, front_only_bins,
                                     p_idx, final_intersection, width_new, birth_time_new, tip_id[node_idx])
    else:
        # shift the node itself
        position[p_idx] = final_intersection
        dist_to_center[p_idx] = seg_len(center, final_intersection)
        angle_bin[p_idx] = _angle_bin(center, final_intersection, front_only_bins)
        if dist_to_center[p_idx] > max_dist_list[0]:
            max_dist_list[0] = dist_to_center[p_idx]
        cn.delete_segment(box2elem, elem2box, p_idx)
        if parent[p_idx] >= 0:
            cn.add_segment(unit_size, box2elem, elem2box,
                           position[parent[p_idx]], position[p_idx],
                           width[parent[p_idx]], width[p_idx], p_idx)
        new_tip_idx = p_idx

    return new_tip_idx, is_cont, overlap

@njit(cache=True, fastmath=False)
def _splice_out_degree2_parent_if_safe(unit_size, box2elem, elem2box,
                                       position, width, birth_time, tip_id, is_contact, is_protected_scaffold, parent, children, alive,
                                       alive_nodes, alive_pos,
                                       parent_idx: int, max_length_merge: float,
                                       max_merge_angle_rad: float) -> b1:
    """
    If parent_idx has exactly one child (after a tip was deleted) and has a valid grandparent,
    and the length of the would-be merged segment (grandparent -> only_child) is <= max_length_merge,
    then:
      - remove collision segments (grandparent->parent) [id=parent_idx] and (parent->only_child) [id=only_child]
      - rewire only_child under grandparent
      - add new segment (grandparent->only_child) [id=only_child]
      - mark parent as not alive
    Returns True if a merge was performed.
    """
    if not alive[parent_idx]:
        return False
    if is_protected_scaffold[parent_idx]:
        return False

    # Grandparent must exist to splice parent out
    gp = parent[parent_idx]
    if gp < 0 or (not alive[gp]):
        return False

    # Parent must have exactly one remaining child
    if len(children[parent_idx]) != 1:
        return False

    only_child = children[parent_idx][0]
    if (only_child < 0) or (not alive[only_child]):
        return False
    if is_protected_scaffold[only_child]:
        return False

    # New merged length would be grandparent -> only_child
    new_len = seg_len(position[gp], position[only_child])
    if new_len > max_length_merge:
        return False

    # Alignment guard: require near-collinearity between gp->parent and parent->child
    v1 = position[parent_idx] - position[gp]
    v2 = position[only_child] - position[parent_idx]
    n1 = math.hypot(v1.real, v1.imag)
    n2 = math.hypot(v2.real, v2.imag)
    if n1 > 1e-10 and n2 > 1e-10:
        c = (v1.real * v2.real + v1.imag * v2.imag) / (n1 * n2)
        if c > 1.0:
            c = 1.0
        elif c < -1.0:
            c = -1.0
        ang = math.acos(c)
        if ang > max_merge_angle_rad:
            return False

    # Collisioner: remove old segments
    # Segment id is the child's compartment id
    # gp -> parent has id = parent_idx (child id of that edge)
    cn.delete_segment(box2elem, elem2box, parent_idx)
    # parent -> only_child has id = only_child
    cn.delete_segment(box2elem, elem2box, only_child)

    # Rewire children lists
    # Remove 'parent_idx' from gp's children, and add 'only_child' instead
    newkids_gp = List.empty_list(i64)
    for k in range(len(children[gp])):
        if children[gp][k] != parent_idx:
            newkids_gp.append(children[gp][k])
    # append the child
    newkids_gp.append(only_child)
    children[gp] = newkids_gp

    # Parent loses its child list (optional, not strictly necessary)
    children[parent_idx] = List.empty_list(i64)

    # Update parent pointer of the only_child
    parent[only_child] = gp

    # Mark parent dead
    alive[parent_idx] = False
    _unregister_alive_node(alive_nodes, alive_pos, parent_idx)

    # Collisioner: add the merged segment gp -> only_child, id = only_child
    cn.add_segment(unit_size, box2elem, elem2box,
                   position[gp], position[only_child],
                   width[gp], width[only_child], only_child)

    return True

@njit(cache=True, fastmath=False)
def shrink_tip(unit_size, box2elem, elem2box,
               position, width, birth_time, tip_id, is_contact, is_protected_scaffold, parent, children, alive,
               alive_nodes, alive_pos,
               center: c128, dist_to_center, angle_bin, max_dist_list, front_only_bins: int,
               node_idx: int, shrink_length: float, min_dist_nodes: float = 0.0,
               max_length_merge: float = 0.25,
               max_merge_angle_rad: float = math.pi,
               boundary_mode: int = 0, boundary_x0: float = 0.0,
               boundary_y_min: float = 0.0, boundary_y_max: float = 0.0):
    """
    Shrink along the tip's edge(s). When a tip node disappears or the edge is split,
    the compartment that becomes the new tip inherits the same tip_id lineage. After
    deleting a tip, if its parent now has exactly one other child and the
    merged (grandparent -> that child) segment would be <= max_length_merge,
    we splice the parent out and rewire that child directly to the grandparent.
    """
    length_left = shrink_length
    current_idx = node_idx

    while (length_left > 0.0) and is_branch_tip(children, current_idx) and alive[current_idx]:
        p_idx = parent[current_idx]
        if p_idx < 0:
            break
        dist_parent = seg_len(position[current_idx], position[p_idx])
        if is_protected_scaffold[p_idx]:
            # Do not let retraction consume an edge that would move the tip into protected scaffold.
            min_keep = max(min_dist_nodes, 1e-6)
            if (length_left >= dist_parent) and (dist_parent > min_keep):
                prop_keep = min_keep / dist_parent
                new_idx = split_node(unit_size, box2elem, elem2box,
                                     position, width, birth_time, tip_id, is_contact, is_protected_scaffold, parent, children, alive,
                                     alive_nodes, alive_pos,
                                     center, dist_to_center, angle_bin, max_dist_list, front_only_bins,
                                     current_idx, prop_keep, min_dist_nodes, True)
                tip_id[new_idx] = tip_id[current_idx]
                current_idx = new_idx
            tip_alive_guard = is_branch_tip(children, current_idx) and alive[current_idx]
            return current_idx, length_left, tip_alive_guard
        if boundary_mode == 1:
            if _is_on_trunk(position[current_idx], boundary_x0, boundary_y_min, boundary_y_max) and \
               _is_on_trunk(position[p_idx], boundary_x0, boundary_y_min, boundary_y_max):
                return current_idx, length_left, False

        if length_left < dist_parent:
            # split at the right point and delete end
            if dist_parent <= 1e-8:
                delete_node(unit_size, box2elem, elem2box,
                            position, width, birth_time, tip_id, is_contact, is_protected_scaffold, parent, children, alive,
                            alive_nodes, alive_pos,
                            current_idx)
                # lineage inheritance up to parent
                tip_id[p_idx] = tip_id[current_idx]
                # After removing the tiny edge, try splice (parent might have exactly one child now)
                _splice_out_degree2_parent_if_safe(unit_size, box2elem, elem2box,
                                                   position, width, birth_time, tip_id, is_contact, is_protected_scaffold, parent, children, alive,
                                                   alive_nodes, alive_pos,
                                                   p_idx, max_length_merge, max_merge_angle_rad)
                current_idx = p_idx
                continue

            prop = (dist_parent - length_left) / dist_parent
            new_idx = split_node(unit_size, box2elem, elem2box,
                                 position, width, birth_time, tip_id, is_contact, is_protected_scaffold, parent, children, alive,
                                 alive_nodes, alive_pos,
                                 center, dist_to_center, angle_bin, max_dist_list, front_only_bins,
                                 current_idx, prop, min_dist_nodes, True)
            # new tip inherits lineage
            tip_id[new_idx] = tip_id[current_idx]
            current_idx = new_idx
            length_left = 0.0
        else:
            # delete current tip and move up
            delete_node(unit_size, box2elem, elem2box,
                        position, width, birth_time, tip_id, is_contact, is_protected_scaffold, parent, children, alive,
                        alive_nodes, alive_pos,
                        current_idx)
            
            # If parent now has exactly one other child, and grandparent->child length <= max_length_merge,
            # splice parent out (so parent does not become the tip).
            merged = _splice_out_degree2_parent_if_safe(unit_size, box2elem, elem2box,
                                                        position, width, birth_time, tip_id, is_contact, is_protected_scaffold, parent, children, alive,
                                                        alive_nodes, alive_pos,
                                                        p_idx, max_length_merge, max_merge_angle_rad)

            if (not merged) and is_branch_tip(children, p_idx):
                # If no merge happened and parent actually became a tip, pass the lineage up
                tip_id[p_idx] = tip_id[current_idx]

            # climb one step up in any case (parent may have been spliced out)
            current_idx = parent[p_idx] if (merged and alive[p_idx] == False) else p_idx
            length_left -= dist_parent

    tip_alive = is_branch_tip(children, current_idx) and alive[current_idx]
    return current_idx, length_left, tip_alive


@njit(cache=True, fastmath=False)
def make_new_branch(unit_size, box2elem, elem2box,
                    position, width, birth_time, tip_id, is_contact, is_protected_scaffold, parent, children, alive,
                    alive_nodes, alive_pos,
                    center: c128, dist_to_center, angle_bin, max_dist_list, front_only_bins: int,
                    parent_node_idx: int, proportion_parent_node_split: float,
                    angle_wrt_parent_node: float, target_length: float, birth_time_new: float,
                    new_tip_id: int, new_branch_width: float,
                    degree_exempt_collision: int,
                    proba_overlap: float, min_dist_nodes: float = 0.0,
                    boundary_mode: int = 0, boundary_x0: float = 0.0,
                    boundary_y_min: float = 0.0, boundary_y_max: float = 0.0):
    """
    Create a side branch with a brand new tip_id lineage = new_tip_id.
    Every compartment spawned by this branch carries that lineage id.
    """
    new_root_idx = split_node(unit_size, box2elem, elem2box,
                              position, width, birth_time, tip_id, is_contact, is_protected_scaffold, parent, children, alive,
                              alive_nodes, alive_pos,
                              center, dist_to_center, angle_bin, max_dist_list, front_only_bins,
                              parent_node_idx, proportion_parent_node_split, min_dist_nodes, False)
    u = unit_vec(position[parent_node_idx if parent[new_root_idx] == parent_node_idx else parent[new_root_idx]],
                 position[new_root_idx])
    new_side_pos = position[new_root_idx] + (1j * (1.0 if angle_wrt_parent_node >= 0 else -1.0)) * u * width[parent_node_idx]
    # first compartment of the new branch uses new_tip_id lineage
    side_idx = add_child_node(unit_size, box2elem, elem2box,
                              position, width, birth_time, tip_id, is_contact, is_protected_scaffold, parent, children, alive,
                              alive_nodes, alive_pos,
                              center, dist_to_center, angle_bin, max_dist_list, front_only_bins,
                              new_root_idx, new_side_pos, new_branch_width, birth_time_new, new_tip_id)

    target_pos = new_side_pos + target_length * u * np.exp(1j * angle_wrt_parent_node)
    target_pos, boundary_contact = _clip_target_to_bounds(
        new_side_pos, target_pos,
        boundary_x0, boundary_y_min, boundary_y_max,
        boundary_mode
    )
    exclusion = np.empty(3, dtype=np.int64)
    exclusion[0] = parent_node_idx
    exclusion[1] = new_root_idx
    exclusion[2] = side_idx

    new_tip_idx, is_cont, overlap = grow_tip(unit_size, box2elem, elem2box,
                                             position, width, birth_time, tip_id, is_contact, is_protected_scaffold, parent, children, alive,
                                             alive_nodes, alive_pos,
                                             center, dist_to_center, angle_bin, max_dist_list, front_only_bins,
                                             side_idx, target_pos, exclusion, birth_time_new, new_branch_width,
                                             degree_exempt_collision,
                                             proba_overlap, min_dist_nodes, 0.1)
    if boundary_contact:
        is_cont = True
    return new_tip_idx, is_cont, overlap


@njit(cache=True, fastmath=False)
def step_kernel(
    # state
    unit_size, elem2box, box2elem,max_length_merge,
    position, width, birth_time, tip_id, is_contact, is_protected_scaffold, parent, children, alive,
    alive_nodes, alive_pos,
    center, dist_to_center, angle_bin, max_dist_list,
    tip_list,
    # sim controls
    dt: float,
    simulation_time: float,
    proba_overlap: float,
    persistence_length: float,
    new_branch_width: float,
    min_dist_node: float,
    degree_exempt_collision: int,
    # outputs/updates
    max_tip_id_inout: int,
    step_increments,
    # external branching events
    branch_parent_nodes,
    branch_prop_splits,
    branch_angles,
    branch_lengths,
    branch_new_tip_ids,
    # boundary controls
    boundary_mode: int,
    boundary_x0: float,
    boundary_y_min: float,
    boundary_y_max: float,
    front_only: float,
    front_only_bins: int,
    max_r_per_bin,
    max_merge_angle_rad: float,
):
    """
    Hot loop: iterate over current tips (node indices), grow/shrink using 'step_increments'
    (aligned with the same order), then branch. Returns:
      - next_tip_list: list of node indices (compartment ids) that are tips after this step
      - max_tip_id: updated max lineage id counter
      - next_contact_list: list[bool] aligned with next_tip_list
    """
    next_tip_list = List.empty_list(i64)
    next_contact_list = List.empty_list(b1)

    # TIP UPDATE (growth/shrink) 
    for k in range(len(tip_list)):
        node_idx = tip_list[k]
        if not alive[node_idx]:
            continue
        if parent[node_idx] < 0:
            continue  # ignore root
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
                next_tip_list.append(node_idx)
                next_contact_list.append(False)
                continue

        p_idx = parent[node_idx]
        vec = position[node_idx] - position[p_idx]
        L = seg_len(position[p_idx], position[node_idx])
        if L <= 0.0:
            u = 1.0 + 0.0j
        else:
            u = vec / (L + 1e-4)

        next_increment = step_increments[k]

        if next_increment > 0.0:
            # angular diffusion with persistence
            diff_angle = math.sqrt(max(2.0 * next_increment / max(persistence_length, 1e-4), 0.0)) * np.random.normal(0.0, 1.0)
            target_position = position[node_idx] + np.exp(1j * diff_angle) * u * next_increment
            target_position, boundary_contact = _clip_target_to_bounds(
                position[node_idx], target_position,
                boundary_x0, boundary_y_min, boundary_y_max,
                boundary_mode
            )

            exclusion = np.empty(2, dtype=np.int64)
            exclusion[0] = p_idx
            exclusion[1] = node_idx

            new_tip_idx, is_cont, _over = grow_tip(unit_size, box2elem, elem2box,
                                                   position, width, birth_time, tip_id, is_contact, is_protected_scaffold, parent, children, alive,
                                                   alive_nodes, alive_pos,
                                                   center, dist_to_center, angle_bin, max_dist_list, front_only_bins,
                                                   node_idx, target_position, exclusion,
                                                   simulation_time + dt, new_branch_width,
                                                   degree_exempt_collision,
                                                   proba_overlap, min_dist_node, 0.1)
            if alive[new_tip_idx] and is_branch_tip(children, new_tip_idx):
                next_tip_list.append(new_tip_idx)
                next_contact_list.append(is_cont or boundary_contact)
        else:
            # SHRINK
            new_idx, _left, tip_alive = shrink_tip(unit_size, box2elem, elem2box,
                                                   position, width, birth_time, tip_id, is_contact, is_protected_scaffold, parent, children, alive,
                                                   alive_nodes, alive_pos,
                                                   center, dist_to_center, angle_bin, max_dist_list, front_only_bins,
                                                   node_idx, -next_increment, min_dist_node, max_length_merge, max_merge_angle_rad,
                                                   boundary_mode, boundary_x0, boundary_y_min, boundary_y_max)
            if tip_alive and alive[new_idx] and is_branch_tip(children, new_idx):
                next_tip_list.append(new_idx)
                next_contact_list.append(False)  # no contact from pure shrink

    # BRANCHING LOOP (external decisions) 
    n_events = len(branch_parent_nodes)
    for ev in range(n_events):
        node_idx = branch_parent_nodes[ev]
        prop     = branch_prop_splits[ev]
        angle    = branch_angles[ev]
        new_L    = branch_lengths[ev]
        new_id   = branch_new_tip_ids[ev]
        if not alive[node_idx]:
            continue
        if boundary_mode != 0:
            p_idx = parent[node_idx]
            if p_idx < 0:
                continue
            middle_pos = (1.0 - prop) * position[p_idx] + prop * position[node_idx]
            u = unit_vec(position[p_idx], position[node_idx])
            side_pos = middle_pos + (1j * (1.0 if angle >= 0 else -1.0)) * u * width[node_idx]
            if not _is_inside_bounds(side_pos, boundary_x0, boundary_y_min, boundary_y_max, boundary_mode):
                continue
        new_tip_idx, is_cont, _ov = make_new_branch(unit_size, box2elem, elem2box,
                                                    position, width, birth_time, tip_id, is_contact, is_protected_scaffold, parent, children, alive,
                                                    alive_nodes, alive_pos,
                                                    center, dist_to_center, angle_bin, max_dist_list, front_only_bins,
                                                    node_idx, prop, angle, new_L,
                                                    simulation_time + dt,
                                                    new_id, new_branch_width,
                                                    degree_exempt_collision,
                                                    proba_overlap, min_dist_node,
                                                    boundary_mode, boundary_x0,
                                                    boundary_y_min, boundary_y_max)
        if alive[new_tip_idx] and is_branch_tip(children, new_tip_idx):
            next_tip_list.append(new_tip_idx)
            next_contact_list.append(is_cont)

    return next_tip_list, max_tip_id_inout, next_contact_list
