import numpy as np
import networkx as nx
from math import atan2

def tree_inertia_and_equivalent_ellipse(g: nx.Graph):
    """
    Compute geometric moments of a tree encoded as a graph with nodes at (row, col) = (y, x).
    Each edge is treated as a straight line segment with uniform linear density (mass per unit length = 1).

    Returns a dict with:
      - total_length: float, sum of segment lengths (mass)
      - centroid: np.array shape (2,), [y, x]
      - inertia_tensor: 2x2 np.array, central second moment ∫ (r - C)(r - C)^T ds  (units: length^3)
      - covariance: 2x2 np.array = inertia_tensor / total_length                    (units: length^2)
      - eigenvalues: np.array [λ_max, λ_min] of covariance (≥ 0 up to numerical noise)
      - eigenvectors: 2x2 np.array columns are unit eigenvectors for λ_max then λ_min (in [y, x] order)
      - semi_major: float, a = 2*sqrt(λ_max)
      - semi_minor: float, b = 2*sqrt(λ_min)
      - angle_rad: float, orientation of the semi-major axis, angle of the eigenvector wrt x-axis (columns).
                   Positive angles rotate toward +y (rows increasing). Uses atan2(v_y, v_x).
    """
    # Handle empty or trivial graphs
    if g is None or g.number_of_edges() == 0:
        return {
            "total_length": 0.0,
            "centroid": np.array([np.nan, np.nan], dtype=float),
            "inertia_tensor": np.full((2, 2), np.nan, dtype=float),
            "covariance": np.full((2, 2), np.nan, dtype=float),
            "eigenvalues": np.array([np.nan, np.nan], dtype=float),
            "eigenvectors": np.full((2, 2), np.nan, dtype=float),
            "semi_major": np.nan,
            "semi_minor": np.nan,
            "angle_rad": np.nan,
        }

    # Accumulators
    L_tot = 0.0                         # total length (mass)
    M1 = np.zeros(2, dtype=float)       # first moment: ∑ ∫ r ds  (vector)
    M2 = np.zeros((2, 2), dtype=float)  # second moment: ∑ ∫ r r^T ds  (matrix)

    for u, v, data in g.edges(data=True):
        # Node coordinates as [y, x]
        A = np.array(u, dtype=float)
        B = np.array(v, dtype=float)

        # Segment vector and length from node positions (NOT from pixel paths)
        U = B - A
        L = float(np.linalg.norm(U))
        if L == 0.0 or not np.isfinite(L):
            continue

        # First moment over the segment: ∫ r ds = L * (A + 0.5 U)
        M1_seg = L * (A + 0.5 * U)

        # Second moment over the segment:
        # ∫ r r^T ds = L * [ A A^T + 0.5 (A U^T + U A^T) + (1/3) U U^T ]
        AAT = np.outer(A, A)
        AUT = np.outer(A, U)
        UAT = np.outer(U, A)
        UUT = np.outer(U, U)
        M2_seg = L * (AAT + 0.5 * (AUT + UAT) + (1.0 / 3.0) * UUT)

        # Accumulate
        L_tot += L
        M1 += M1_seg
        M2 += M2_seg

    if L_tot == 0.0:
        return {
            "total_length": 0.0,
            "centroid": np.array([np.nan, np.nan], dtype=float),
            "inertia_tensor": np.full((2, 2), np.nan, dtype=float),
            "covariance": np.full((2, 2), np.nan, dtype=float),
            "eigenvalues": np.array([np.nan, np.nan], dtype=float),
            "eigenvectors": np.full((2, 2), np.nan, dtype=float),
            "semi_major": np.nan,
            "semi_minor": np.nan,
            "angle_rad": np.nan,
        }

    # Global centroid
    C = M1 / L_tot  # shape (2,)

    # Central inertia tensor: J = M2 - L_tot * C C^T
    CCt = np.outer(C, C)
    J = M2 - L_tot * CCt

    # Covariance-like matrix (second central moment per unit length)
    Sigma = J / L_tot

    # Numerical safety: symmetrize Sigma and clip tiny negatives
    Sigma = 0.5 * (Sigma + Sigma.T)
    # Eigen decomposition (largest first)
    evals, evecs = np.linalg.eigh(Sigma)
    # eigh returns ascending order; reverse to [λ_max, λ_min]
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Clip small negatives from numerical noise
    evals = np.clip(evals, a_min=0.0, a_max=None)

    # Semi-axes of equivalent (filled) ellipse: a=2*sqrt(λ_max), b=2*sqrt(λ_min)
    a = 2.0 * np.sqrt(evals[0]) if np.isfinite(evals[0]) else np.nan
    b = 2.0 * np.sqrt(evals[1]) if np.isfinite(evals[1]) else np.nan

    # Orientation of the major axis: eigenvector for λ_max.
    # Our coordinate order is [y, x]; angle wrt x-axis (columns) is atan2(v_y, v_x).
    v_major = evecs[:, 0]  # [v_y, v_x]
    angle_rad = atan2(float(v_major[0]), float(v_major[1]))
    

    return {
        "total_length": L_tot,
        "centroid": C,
        "inertia_tensor": J,
        "covariance": Sigma,
        "eigenvalues": evals,
        "eigenvectors": evecs,
        "semi_major": float(a),
        "semi_minor": float(b),
        "angle_rad": float(angle_rad),
    }
