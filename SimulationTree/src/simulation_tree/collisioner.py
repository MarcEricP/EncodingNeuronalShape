import math
import numpy as np
from numba import njit, types
from numba.typed import Dict, List


# Tile-based collisioner for constant time look-up
_i64 = types.int64
_k2  = types.UniTuple(_i64, 2)                   # (row, colmuns)
_list_i64   = types.ListType(_i64)               # [int]
_list_k2    = types.ListType(_k2)                # [(row, columns)]
box2elem_sig = types.DictType(_k2, _list_i64)    # {(row,columns): [elem_ids]}
elem2box_sig = types.DictType(_i64, _list_k2)    # {elem_id: [(row,colmuns), ...]}


@njit(cache=True, fastmath=False)
def _cx(x, y):
    return x + 1j * y

@njit(cache=True, fastmath=False)
def _re(z):
    return z.real

@njit(cache=True, fastmath=False)
def _im(z):
    return z.imag


@njit(cache=True, fastmath=False)
def expand_segment(root_position, tip_position, eps=0.0002):
    """
    Extend segment by 'eps' at both ends along its direction.
    Inputs are complex numbers.
    """
    vec = tip_position - root_position
    l_vec = math.hypot(_re(vec), _im(vec))
    if l_vec > 0.0:
        ux = _re(vec) / l_vec
        uy = _im(vec) / l_vec
    else:
        ux = 0.0
        uy = 0.0
    root_pos = _cx(_re(root_position) - eps * ux, _im(root_position) - eps * uy)
    tip_pos  = _cx(_re(tip_position)  + eps * ux, _im(tip_position)  + eps * uy)
    return root_pos, tip_pos

@njit(cache=True, fastmath=False)
def get_polygon(root_coords_1, root_width, tip_coords_1, tip_width, expand_width=0.0002, expand_seg=0.0002):
    """
    Return vertices (complex) of the width-varying segment polygon (va,vb,vc,vd).
    """
    root_coords, tip_coords = expand_segment(root_coords_1, tip_coords_1, eps=expand_seg)
    vx = _re(tip_coords) - _re(root_coords)
    vy = _im(tip_coords) - _im(root_coords)
    # angle normal (perp) via rotation by +pi/2: (nx, ny) = (-vy, vx) normalized
    norm = math.hypot(vx, vy)
    if norm > 0.0:
        nx = -vy / norm
        ny =  vx / norm
    else:
        nx = 0.0
        ny = 0.0

    wr = 0.5 * (root_width + expand_width)
    wt = 0.5 * (tip_width  + expand_width)

    va = _cx(_re(root_coords) + nx * wr, _im(root_coords) + ny * wr)
    vd = _cx(_re(root_coords) - nx * wr, _im(root_coords) - ny * wr)
    vb = _cx(_re(tip_coords)  + nx * wt, _im(tip_coords)  + ny * wt)
    vc = _cx(_re(tip_coords)  - nx * wt, _im(tip_coords)  - ny * wt)

    return np.array([va, vb, vc, vd], dtype=np.complex128)

@njit(cache=True, fastmath=False)
def _segment_intersection(p1, p2, q1, q2):
    """
    Proper segment intersection between p1->p2 and q1->q2.
    Return (hit, x, y) where hit is boolean.
    """
    x1, y1 = _re(p1), _im(p1)
    x2, y2 = _re(p2), _im(p2)
    x3, y3 = _re(q1), _im(q1)
    x4, y4 = _re(q2), _im(q2)

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-15:
        return False, 0.0, 0.0

    pre  = (x1*y2 - y1*x2)
    post = (x3*y4 - y3*x4)
    x = (pre * (x3 - x4) - (x1 - x2) * post) / denom
    y = (pre * (y3 - y4) - (y1 - y2) * post) / denom

    # Check within both segments (bounding box test with small tol)
    tol = 1e-4
    if (min(x1, x2) - tol <= x <= max(x1, x2) + tol and
        min(y1, y2) - tol <= y <= max(y1, y2) + tol and
        min(x3, x4) - tol <= x <= max(x3, x4) + tol and
        min(y3, y4) - tol <= y <= max(y3, y4) + tol):
        return True, x, y
    return False, 0.0, 0.0

@njit(cache=True, fastmath=False)
def _point_in_convex_polygon(p, poly):
    """
    Return True if point p is inside (or on the boundary of) a convex polygon 'poly'.
    'poly' is an array of complex vertices in CCW order.
    Robust to near-collinearity via a small tolerance.
    """
    px, py = _re(p), _im(p)
    n = len(poly)
    tol = 1e-10

    # Determine the expected sign from the first non-degenerate edge
    sign_set = 0  # -1, 0, or +1
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        ax, ay = _re(a), _im(a)
        bx, by = _re(b), _im(b)
        cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
        if abs(cross) <= tol:
            # On or very close to the edge -> keep looking
            continue
        cur_sign = 1 if cross > 0.0 else -1
        if sign_set == 0:
            sign_set = cur_sign
        elif cur_sign != sign_set:
            return False  # different side than earlier -> outside
    # If we never contradicted the sign, we are inside or on boundary
    return True

@njit(cache=True, fastmath=False)
def fine_intersection_polygon(root1, tip1, wr1, wt1, root2, tip2, wr2, wt2):
    """
    Build polygon2
    Intersect line(root1->tip1) with polygon2 edges
    Return intersection point (complex) closest to root1, or None (encoded as np.nan+1j*np.nan)
    """
    # Build polygon2
    # poly1 = get_polygon(root1, wr1, tip1, wt1, expand_width=0.02, expand_seg=0.02)
    poly2 = get_polygon(root2, wr2, tip2, wt2, expand_width=0.0002, expand_seg=0.0002)
    # Edges: (v0->v1, v1->v2, v2->v3, v3->v0)
    edges = ((0,1),(1,2),(2,3),(3,0))
    # edges = ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))

    best_d2 = 1e300
    hitx = np.nan
    hity = np.nan

    for e0, e1 in edges:
        ok, x, y = _segment_intersection(root1, tip1, poly2[e0], poly2[e1])
        if ok:
            dx = x - _re(root1)
            dy = y - _im(root1)
            d2 = dx*dx + dy*dy
            if d2 < best_d2:
                best_d2 = d2
                hitx, hity = x, y

    if math.isfinite(best_d2):
        return _cx(hitx, hity)
        
    
    if _point_in_convex_polygon(root1, poly2) and _point_in_convex_polygon(tip1, poly2):
        return root1
    return np.nan + 1j*np.nan


@njit(cache=True, fastmath=False)
def box_coordinates_dirty(box_size, root_coords, tip_coords):
    """
    Approximate set of integer grid boxes crossed by the line on the (r,c) grid.
    We step uniformly with steps = max(|dr|,|dc|)+1 between integer boxes of endpoints.
    """
    r0 = int(math.floor(_re(root_coords) / box_size))
    c0 = int(math.floor(_im(root_coords) / box_size))
    r1 = int(math.floor(_re(tip_coords)  / box_size))
    c1 = int(math.floor(_im(tip_coords)  / box_size))

    dr = r1 - r0
    dc = c1 - c0
    steps = abs(dr) if abs(dr) > abs(dc) else abs(dc)
    steps += 1

    out = List.empty_list(_k2)
    if steps <= 1:
        out.append((_i64(r0), _i64(c0)))
        if r1 != r0 or c1 != c0:
            out.append((_i64(r1), _i64(c1)))
        return out

    for s in range(steps):
        t = s / (steps - 1.0)
        rr = int(round(r0 + t * dr))
        cc = int(round(c0 + t * dc))
        # avoid duplicates (cheap check: only push if different than previous)
        if len(out) == 0 or out[-1][0] != rr or out[-1][1] != cc:
            out.append((_i64(rr), _i64(cc)))
    return out


@njit(cache=True, fastmath=False)
def box_coordinates_dirty_polygon(box_size, root_coords, root_width, tip_coords, tip_width,
                                  expand_width=0.0002, expand_seg=0.0002):
    """
    Return all (r,c) integer boxes whose centers fall inside the segment polygon.
    """
    vertices = get_polygon(root_coords, root_width, tip_coords, tip_width,
                           expand_width=expand_width, expand_seg=expand_seg)
    # Convert vertices to grid coords (r,c) in integer box space (float first)
    vr = np.empty(4, dtype=np.float64)
    vc = np.empty(4, dtype=np.float64)
    for k in range(4):
        vr[k] = _re(vertices[k]) / box_size
        vc[k] = _im(vertices[k]) / box_size

    min_r = int(math.floor(vr.min()))
    max_r = int(math.ceil(vr.max()))
    min_c = int(math.floor(vc.min()))
    max_c = int(math.ceil(vc.max()))

    out = List.empty_list(_k2)
    for rr in range(min_r, max_r + 1):
        for cc in range(min_c, max_c + 1): 
            out.append((_i64(rr), _i64(cc)))
    return out

def initialize(unit_size_micron=10):
    """
    Create numba-typed dictionaries for collision structure.
    Return (unit_size, elem2box, box2elem)
    """
    box2elem = Dict.empty(key_type=_k2, value_type=_list_i64)
    elem2box = Dict.empty(key_type=_i64, value_type=_list_k2)
    return unit_size_micron, elem2box, box2elem



@njit(cache=True, fastmath=False)
def _set_elem_boxes(elem2box, elem_id, boxes_list):
    """
    Assign the list of boxes to elem2box[elem_id].
    """
    elem2box[elem_id] = boxes_list

@njit(cache=True, fastmath=False)
def _remove_id_from_list(lst, elem_id):
    """
    Return a NEW list without elem_id (numba Lists have limited remove()).
    """
    newlst = List.empty_list(_i64)
    for k in range(len(lst)):
        if lst[k] != elem_id:
            newlst.append(lst[k])
    return newlst

@njit(cache=True, fastmath=False)
def add_segment(unit_size, box2elem, elem2box,
                root_pos, tip_pos, root_width, tip_width, tip_id):
    """
    Insert a segment polygon:
    - updates box2elem[(r,c)] -> append tip_id
    - updates elem2box[tip_id] -> list of (r,c)
    """
    boxes_coords = box_coordinates_dirty_polygon(unit_size, root_pos, root_width,
                                                 tip_pos, tip_width,
                                                 expand_width=0.1, expand_seg=0.1)
    # Update box2elem
    for i in range(len(boxes_coords)):
        key = boxes_coords[i]
        if key not in box2elem:
            box2elem[key] = List.empty_list(_i64)
        lst = box2elem[key]
        lst.append(tip_id)

    # Store boxes list for this element
    _set_elem_boxes(elem2box, tip_id, boxes_coords)

@njit(cache=True, fastmath=False)
def delete_segment(box2elem, elem2box, segment_id):
    """
    Remove a segment:
    - from each box list in box2elem
    - delete elem2box[segment_id]
    """
    if segment_id not in elem2box:
        return
    boxes_coords = elem2box[segment_id]
    # For every box, rebuild its id list without the segment
    for i in range(len(boxes_coords)):
        key = boxes_coords[i]
        if key in box2elem:
            box2elem[key] = _remove_id_from_list(box2elem[key], segment_id)
            

    # Remove element entry
    del elem2box[segment_id]

@njit(cache=True, fastmath=False)
def get_potential_collisions(unit_size, box2elem,
                             root_position, tip_position,
                             root_width, tip_width, mode=0):
    """
    mode = 0 -> 'tip': use line boxes between expanded root->tip
    mode = 1 -> 'polygon': use polygon coverage
    Return a numba List of unique integer ids (potentially colliding).
    """
    if mode == 0:  # 'tip'
        root_pos, tip_pos = expand_segment(root_position, tip_position,eps=0.0002)
        boxes_coords = box_coordinates_dirty(unit_size, root_pos, tip_pos)
    else:  # 'polygon'
        boxes_coords = box_coordinates_dirty_polygon(unit_size,
                                                     root_position, root_width,
                                                     tip_position, tip_width,
                                                     expand_width=0.0002, expand_seg=0.0002)
    # Gather ids
    acc = List.empty_list(_i64)
    for i in range(len(boxes_coords)):
        key = boxes_coords[i]
        if key in box2elem:
            ids_here = box2elem[key]
            for k in range(len(ids_here)):
                acc.append(ids_here[k])

    # Unique via sort + dedup
    if len(acc) == 0:
        return acc

    # Convert to numpy array to sort 
    arr = np.empty(len(acc), dtype=np.int64)
    for i in range(len(acc)):
        arr[i] = acc[i]
    arr.sort()

    out = List.empty_list(_i64)
    last = arr[0]
    out.append(last)
    for i in range(1, arr.shape[0]):
        if arr[i] != last:
            last = arr[i]
            out.append(last)
    return out


