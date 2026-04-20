"""
Python port of FSC.m and LSM.m from HOMO_Feature_ImgMatching.

Original MATLAB by Gao Chenzhong (gao-pingqi@qq.com).

Public API
----------
fsc(cor1_xy, cor2_xy, t_form, error, max_iter=800, ref=None)
    -> (solution, rmse, cor1_inliers, cor2_inliers)

    solution     : (3, 3) float array, or None when fewer than 20 inliers
    rmse         : float, or None
    cor1_inliers : (K, 2) inlier rows from cor1_xy, or empty (0, 2) array
    cor2_inliers : (K, 2) inlier rows from cor2_xy, or empty (0, 2) array
"""

import numpy as np


# =========================================================================== #
# LSM — Least-Squares spatial transform estimation                             #
# =========================================================================== #

def _lsm(cor1_xy: np.ndarray, cor2_xy: np.ndarray, t_form: str):
    """
    LSM.m: fit a spatial transform from point correspondences.

    Matches the QR-based least-squares solve in the original.  For the minimal
    sample case (e.g. 3 points for affine) the system is exactly determined;
    for the refit pass it is over-determined and lstsq gives the OLS solution.

    Parameters
    ----------
    cor1_xy : (N, 2) source points
    cor2_xy : (N, 2) target points
    t_form  : 'similarity' | 'affine' | 'projective'

    Returns
    -------
    H    : (8,) float vector whose entries fill the 3×3 solution matrix as
               [[H0, H1, H4],
                [H2, H3, H5],
                [H6, H7, 1.0]]
           For affine/similarity: H6 = H7 = 0.
    rmse : scalar reprojection error
    """
    N = len(cor1_xy)
    x = cor1_xy[:, 0]
    y = cor1_xy[:, 1]
    u = cor2_xy[:, 0]
    v = cor2_xy[:, 1]

    # Build right-hand side: interleave u,v per point — matches MATLAB b = t_match2_xy(:)
    b = np.empty(2 * N)
    b[0::2] = u
    b[1::2] = v

    if t_form == 'affine':
        # A = [x,y,0,0,1,0; 0,0,x,y,0,1; ...] — matches LSM.m affine block
        z = np.zeros(N); o = np.ones(N)
        A = np.empty((2 * N, 6))
        A[0::2] = np.column_stack([x, y, z, z, o, z])
        A[1::2] = np.column_stack([z, z, x, y, z, o])
        h, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        H = np.array([h[0], h[1], h[2], h[3], h[4], h[5], 0., 0.])
        pred_x = H[0]*x + H[1]*y + H[4]
        pred_y = H[2]*x + H[3]*y + H[5]

    elif t_form == 'similarity':
        # A = [x,y,1,0; y,-x,0,1; ...] — matches LSM.m similarity block
        A = np.empty((2 * N, 4))
        A[0::2] = np.column_stack([x,  y, np.ones(N), np.zeros(N)])
        A[1::2] = np.column_stack([y, -x, np.zeros(N), np.ones(N)])
        h, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        # MATLAB rearrangement: H(7:8)=0; H(5:6)=H(3:4); H(3)=-H(2); H(4)=H(1)
        a, bc, c, d = h
        H = np.array([a, bc, -bc, a, c, d, 0., 0.])
        pred_x = H[0]*x + H[1]*y + H[4]
        pred_y = H[2]*x + H[3]*y + H[5]

    elif t_form == 'projective':
        # DLT form — matches LSM.m projective block
        # Row 2i:   [x, y, 0, 0, 1, 0, -u*x, -u*y]
        # Row 2i+1: [0, 0, x, y, 0, 1, -v*x, -v*y]
        z = np.zeros(N); o = np.ones(N)
        A = np.empty((2 * N, 8))
        A[0::2] = np.column_stack([x, y, z, z, o, z, -u*x, -u*y])
        A[1::2] = np.column_stack([z, z, x, y, z, o, -v*x, -v*y])
        h, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        H = np.array(h)   # already 8 elements
        sol = np.array([[H[0],H[1],H[4]],[H[2],H[3],H[5]],[H[6],H[7],1.]])
        pts = np.vstack([x, y, np.ones(N)])
        proj = sol @ pts
        proj_xy = proj[:2] / proj[2:3]
        pred_x = proj_xy[0]
        pred_y = proj_xy[1]

    else:
        raise ValueError(f"Unknown t_form: {t_form!r}")

    rmse = float(np.sqrt(np.mean((pred_x - u)**2 + (pred_y - v)**2)))
    return H, rmse


def _build_solution(H: np.ndarray) -> np.ndarray:
    """Build the 3×3 solution matrix from the 8-element H vector (FSC.m layout)."""
    return np.array([
        [H[0], H[1], H[4]],
        [H[2], H[3], H[5]],
        [H[6], H[7], 1.0 ],
    ])


# =========================================================================== #
# FSC — Forward-backward Spatial Consistency (custom RANSAC)                   #
# =========================================================================== #

def fsc(
    cor1_xy: np.ndarray,
    cor2_xy: np.ndarray,
    t_form: str,
    error: float,
    max_iter: int = 800,
    ref=None,
):
    """
    FSC.m: RANSAC-based spatial transform estimation.

    Faithfully translates the MATLAB implementation, including:
      • reference-point anchoring (ref points improve the fit but are not
        counted as inliers — they are zeroed out via inlier_save[M0:] = False)
      • clamped iteration budget: min(combinatorial_upper_bound, max_iter)
      • final refit on inliers only via LSM

    Parameters
    ----------
    cor1_xy  : (M0, 2) source point coordinates
    cor2_xy  : (M0, 2) target point coordinates
    t_form   : 'similarity' | 'affine' | 'projective'
               (substrings 'project', 'perspect', 'homo' also accepted)
    error    : inlier distance threshold in pixels
    max_iter : RANSAC iteration cap (default 800, matches MATLAB)
    ref      : None, or (ref1, ref2) arrays of shape (R, ≥2).  Only the first
               two columns are used.  ref points are appended before RANSAC
               to anchor the fit but are excluded from the inlier count.

    Returns
    -------
    solution     : (3, 3) float array, or None when < 20 inliers found
    rmse         : float reprojection RMSE, or None
    cor1_inliers : (K, 2) inlier subset of cor1_xy, or empty (0, 2) array
    cor2_inliers : (K, 2) inlier subset of cor2_xy, or empty (0, 2) array
    """
    _EMPTY = np.empty((0, 2), dtype=float)

    cor1_xy = np.asarray(cor1_xy, dtype=float)
    cor2_xy = np.asarray(cor2_xy, dtype=float)
    M0 = len(cor1_xy)

    if M0 < 20:
        return None, None, _EMPTY, _EMPTY

    # Normalise transform name — matches MATLAB contains(t_form,{'project',...})
    t_norm = t_form.lower()
    if any(k in t_norm for k in ('project', 'perspect', 'homo')):
        t_norm = 'projective'

    # Append reference points (used for fitting only, never counted as inliers)
    cor1 = cor1_xy
    cor2 = cor2_xy
    has_ref = False
    if ref is not None:
        r1, r2 = ref[0], ref[1]
        if r1 is not None and len(r1) > 0:
            r1 = np.asarray(r1, dtype=float)[:, :2]
            r2 = np.asarray(r2, dtype=float)[:, :2]
            cor1 = np.vstack([cor1_xy, r1])
            cor2 = np.vstack([cor2_xy, r2])
            has_ref = True
    M = len(cor1)

    # Minimum samples and combinatorial iteration budget — matches FSC.m switch
    if t_norm == 'similarity':
        n = 2
        total_iter = M * (M - 1) // 2
        proj_flag = False
    elif t_norm == 'affine':
        n = 3
        total_iter = M * (M - 1) * (M - 2) // 6
        proj_flag = False
    elif t_norm == 'projective':
        n = 4
        total_iter = M * (M - 1) * (M - 2) // 6
        proj_flag = True
    else:
        raise ValueError(f"Unknown t_form: {t_form!r}")

    total_iter = min(int(total_iter), int(max_iter))

    # Pre-build homogeneous match arrays — matches MATLAB match1_xy / match2_xy setup
    match1_hom  = np.vstack([cor1.T, np.ones(M)])   # (3, M)
    match2_xy_T = cor2.T                              # (2, M) — for similarity/affine
    match2_test = cor2.T                              # (2, M) — for projective
    error_sq    = error ** 2

    # Initialise inlier_save over M slots.
    # MATLAB initialises as index array 1:M0, which effectively keeps all M0
    # original points when no model improves; we replicate with all-True for [:M0].
    inlier_save = np.zeros(M, dtype=bool)
    inlier_save[:M0] = True   # default fallback — matches MATLAB inlier_save=1:M0
    best_cnt = 0

    rng = np.random.default_rng()

    for _ in range(total_iter):
        idx = rng.choice(M, n, replace=False)
        try:
            H_vec, _ = _lsm(cor1[idx], cor2[idx], t_norm)
        except (np.linalg.LinAlgError, ValueError):
            continue
        if not np.all(np.isfinite(H_vec)):
            continue

        sol = _build_solution(H_vec)
        proj = sol @ match1_hom   # (3, M)

        if proj_flag:
            denom = proj[2:3]
            if np.any(np.abs(denom) < 1e-10):
                continue
            proj_xy = proj[:2] / denom
            diff_sq = np.sum((proj_xy - match2_test) ** 2, axis=0)
        else:
            diff_sq = np.sum((proj[:2] - match2_xy_T) ** 2, axis=0)

        inlier = diff_sq < error_sq   # (M,) bool
        cnt = int(inlier.sum())

        if cnt > best_cnt:
            best_cnt = cnt
            inlier_save = inlier.copy()

    # Zero out reference-point flags — matches MATLAB inlier_save((M0+1):end)=false
    inlier_save[M0:] = False

    if inlier_save.sum() < 4:
        return None, None, _EMPTY, _EMPTY

    # Final refit on inlier original points
    mask = inlier_save[:M0]
    cor1_in = cor1_xy[mask]
    cor2_in = cor2_xy[mask]
    H_vec, rmse = _lsm(cor1_in, cor2_in, t_norm)
    solution = _build_solution(H_vec)

    return solution, rmse, cor1_in, cor2_in


# =========================================================================== #
# Smoke test                                                                   #
# =========================================================================== #
if __name__ == '__main__':
    import sys

    rng = np.random.default_rng(0)

    # --------------------------------------------------------------------- #
    # Known affine transform                                                  #
    # --------------------------------------------------------------------- #
    A_true = np.array([[0.90,  0.10],
                       [-0.10, 0.95]])
    t_true = np.array([15.0, -8.0])

    N_in, N_out = 20, 5
    N = N_in + N_out

    pts1 = rng.uniform(20, 180, (N, 2))
    pts2 = (A_true @ pts1.T).T + t_true
    # Add small noise to inliers
    pts2[:N_in] += rng.normal(0, 0.5, (N_in, 2))
    # Replace outliers with random positions far from the model
    pts2[N_in:] = rng.uniform(0, 250, (N_out, 2))

    true_mask = np.array([True]*N_in + [False]*N_out)

    print(f"Input: {N} correspondences ({N_in} inliers, {N_out} outliers)")

    sol, rmse, c1, c2 = fsc(pts1, pts2, 'affine', error=5.0, max_iter=800)

    assert sol is not None, "FSC returned no solution — RANSAC failed"

    # Recovered inliers vs ground truth
    # Inlier mask derived from returned cor1_inliers (match by proximity to pts1)
    recovered = len(c1)
    correct = sum(
        np.any(np.all(np.abs(pts1[true_mask] - p) < 0.01, axis=1))
        for p in c1
    )

    print(f"Recovered inliers : {recovered}")
    print(f"Correct inliers   : {correct}/{N_in}")
    print(f"RMSE              : {rmse:.4f}")
    print(f"Solution:\n{sol}")

    assert recovered >= 15, f"Too few inliers recovered: {recovered}"
    assert correct >= 15,   f"Too few correct inliers: {correct}/{N_in}"
    assert rmse < 2.0,      f"RMSE too high: {rmse:.4f}"

    # --------------------------------------------------------------------- #
    # Edge case: fewer than 20 points → must return all None                 #
    # --------------------------------------------------------------------- #
    sol2, rmse2, c12, c22 = fsc(pts1[:10], pts2[:10], 'affine', error=5.0)
    assert sol2 is None and rmse2 is None, "Expected None for <20 points"
    assert c12.shape == (0, 2), f"Expected empty array, got {c12.shape}"

    # --------------------------------------------------------------------- #
    # Similarity mode                                                         #
    # --------------------------------------------------------------------- #
    angle = np.deg2rad(5)
    s = 1.02
    R = s * np.array([[np.cos(angle), -np.sin(angle)],
                       [np.sin(angle),  np.cos(angle)]])
    t_sim = np.array([5.0, -3.0])
    pts2_sim = (R @ pts1[:N_in].T).T + t_sim
    pts2_sim_all = np.vstack([pts2_sim + rng.normal(0, 0.3, (N_in, 2)),
                               rng.uniform(0, 250, (N_out, 2))])
    sol3, rmse3, c13, c23 = fsc(pts1, pts2_sim_all, 'similarity', error=5.0)
    assert sol3 is not None, "Similarity FSC returned no solution"
    assert len(c13) >= 15,   f"Similarity: too few inliers ({len(c13)})"
    print(f"\n[similarity] Recovered {len(c13)} inliers, RMSE={rmse3:.4f}")

    print("\nSmoke test passed.")
    sys.exit(0)
