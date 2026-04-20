"""
Python port of Base_Direction.m and GPolar_Descriptor.m from HOMO_Feature_ImgMatching.

Original MATLAB by Gao Chenzhong (gao-pingqi@qq.com).

Public API
----------
gpolar_descriptor(mom_m, mom_o, kps, patch_size, nba, nbo, rot_flag)
    -> (K_aug, ncols + 1 + (2+3*nba)*nbo) float array

The first columns mirror the (possibly augmented) keypoint array; the remaining
(2 + 3*nba)*nbo columns are the GPolar descriptor vector.
"""

import numpy as np


# =========================================================================== #
# Internal: Base_Direction                                                      #
# =========================================================================== #

def _base_direction(
    kps: np.ndarray,
    mag: np.ndarray,
    mom_o: np.ndarray,
    W: int,
    nbo: int,
) -> np.ndarray:
    """
    Base_Direction.m: assign dominant orientation(s) to each keypoint via a
    weighted orientation histogram with parabolic peak interpolation.

    Parameters
    ----------
    kps   : (K, ncols) float array — keypoint data; first two cols are x, y (0-indexed)
    mag   : (H, W_img) magnitude map (MOM_m)
    mom_o : (H, W_img) orientation map in [0, pi)
    W     : patch half-width (integer)
    nbo   : number of orientation bins

    Returns
    -------
    (K_aug, ncols+1) float array — original kps columns plus orientation in last col.
    One keypoint may produce 0, 1, or more rows (one per dominant orientation peak).
    """
    W = int(round(W))
    patch_size = 2 * W + 1
    X = np.arange(-W, W + 1, dtype=float)
    XX, YY = np.meshgrid(X, X)
    Wcircle = (XX ** 2 + YY ** 2) < (W + 1) ** 2

    rows, cols = mom_o.shape

    # Quantize MOM orientation [0,pi) → 1-indexed bins [1, nbo]
    angle_q = np.clip(
        np.floor(mom_o * nbo / np.pi).astype(int) + 1, 1, nbo
    )

    o_thr = 0.8
    out_rows = []

    for k in range(len(kps)):
        x0 = int(round(float(kps[k, 0])))
        y0 = int(round(float(kps[k, 1])))

        x1 = max(0, x0 - W);  x2 = min(x0 + W, cols - 1)
        y1 = max(0, y0 - W);  y2 = min(y0 + W, rows - 1)

        # angle_bin: 0 = not counted (outside patch / circle), [1,nbo] = valid
        angle_bin = np.zeros((patch_size, patch_size), dtype=int)
        oy = W + y1 - y0;  ox = W + x1 - x0
        angle_bin[oy:oy + (y2 - y1 + 1), ox:ox + (x2 - x1 + 1)] = (
            angle_q[y1:y2 + 1, x1:x2 + 1]
        )
        angle_bin = np.where(Wcircle, angle_bin, 0)

        weight_p = np.zeros((patch_size, patch_size))
        weight_p[oy:oy + (y2 - y1 + 1), ox:ox + (x2 - x1 + 1)] = (
            mag[y1:y2 + 1, x1:x2 + 1]
        )

        # Weighted orientation histogram (0-indexed: bins 0..nbo-1)
        o_hist = np.zeros(nbo)
        valid = angle_bin >= 1
        np.add.at(o_hist, angle_bin[valid] - 1, weight_p[valid])

        max_val = o_hist.max()
        if max_val == 0.0:
            continue
        max_thr = o_thr * max_val

        for o in range(nbo):
            aa = o_hist[(o - 1) % nbo]
            oo = o_hist[o]
            bb = o_hist[(o + 1) % nbo]
            if oo > aa and oo > bb and oo > max_thr:
                # Parabolic interpolation — faithful translation from 1-indexed MATLAB:
                #   o_bin_m = (o+1) + 0.5*(aa-bb)/(aa+bb-2*oo)
                #   o_bin_m = mod(o_bin_m-1, NBO) + 1
                #   orient  = o_bin_m * (pi/NBO)
                interp = 0.5 * (aa - bb) / (aa + bb - 2.0 * oo)
                o_bin_1idx = (o + interp) % nbo + 1.0  # [1, nbo]
                orient = o_bin_1idx * (np.pi / nbo)
                out_rows.append((*kps[k], orient))

    if not out_rows:
        return np.zeros((0, kps.shape[1] + 1))
    return np.array(out_rows, dtype=float)


# =========================================================================== #
# Public: gpolar_descriptor                                                     #
# =========================================================================== #

def gpolar_descriptor(
    mom_m: np.ndarray | None,
    mom_o: np.ndarray,
    kps: np.ndarray,
    patch_size: int,
    nba: int,
    nbo: int,
    rot_flag: bool,
) -> np.ndarray:
    """
    GPolar_Descriptor.m: compute GPolar descriptors at the given keypoints.

    Parameters
    ----------
    mom_m      : (H, W) magnitude map (MOM_m); pass None to use all-ones
    mom_o      : (H, W) orientation map in [0, pi)
    kps        : (K, ncols) float array; first two cols are x, y (0-indexed)
    patch_size : descriptor patch diameter (even, e.g. 72)
    nba        : number of angular bins for the polar grid (e.g. 12)
    nbo        : number of orientation bins (e.g. 12)
    rot_flag   : if True, compute dominant orientation via Base_Direction and
                 build a rotation-invariant descriptor

    Returns
    -------
    (K_aug, ncols + 1 + (2+3*nba)*nbo) float array
        Columns: [kps_aug | descriptor_vectors]
        kps_aug has the original keypoint columns plus a trailing orientation column.
        Descriptor length per keypoint: (2 + 3*nba) * nbo.
    """
    desc_len = (2 + 3 * nba) * nbo
    aug_cols = kps.shape[1] + 1 if kps.ndim == 2 else 3

    if len(kps) == 0:
        return np.zeros((0, aug_cols + desc_len))

    if mom_m is None:
        mom_m = np.ones_like(mom_o)

    # ------------------------------------------------------------------ #
    # Patch geometry                                                        #
    # ------------------------------------------------------------------ #
    W = int(np.floor(patch_size / 2))
    ps = 2 * W + 1          # effective patch side length (patch_size+1 in MATLAB for even input)

    X = np.arange(-W, W + 1, dtype=float)
    XX, YY = np.meshgrid(X, X)                      # (ps, ps)
    Wcircle = (XX ** 2 + YY ** 2) < (W + 1) ** 2

    # ------------------------------------------------------------------ #
    # Radial zone map: -1=outside circle, 0=center, 1=inner, 2=outer      #
    # rr1 = W^2/(2*NBA+1),  rr2 = rr1*(NBA+1)                             #
    # ------------------------------------------------------------------ #
    rr1 = W ** 2 / (2 * nba + 1)
    rr2 = rr1 * (nba + 1)
    rho_sq = XX ** 2 + YY ** 2
    Rho = np.where(rho_sq <= rr1, 1,
          np.where(rho_sq <= rr2, 2, 3))
    Rho = Rho * Wcircle - 1         # {-1, 0, 1, 2}
    Rho_flat = Rho.ravel()

    # ------------------------------------------------------------------ #
    # Angular grid (no-rotation case — constant over all keypoints)        #
    # ------------------------------------------------------------------ #
    Theta0 = np.arctan2(YY, XX) + np.pi             # [0, 2pi]
    Theta_norot = (
        np.floor(Theta0 * nba / (2.0 * np.pi)) % nba
    ).astype(int) + 1                               # [1, nba]

    # ------------------------------------------------------------------ #
    # Deeper-feature weights                                                #
    # w_ratio1 = sqrt(rr1/rr2),  w_ratio2 = sqrt(rr1)/W                   #
    # c_idx: circular-shift index for neighbor averaging over NBA axis     #
    # ------------------------------------------------------------------ #
    w_ratio1 = np.sqrt(rr1 / rr2)
    w_ratio2 = np.sqrt(rr1) / W
    c_idx = np.concatenate([np.arange(1, nba), [0]])   # [1,2,...,nba-1,0]

    # ------------------------------------------------------------------ #
    # Base direction                                                        #
    # ------------------------------------------------------------------ #
    if rot_flag:
        kps_aug = _base_direction(kps, mom_m, mom_o, W, nbo)
    else:
        kps_aug = np.column_stack([kps, np.zeros(len(kps))])

    rows, cols = mom_o.shape
    n_kps = len(kps_aug)
    descriptor = np.zeros((n_kps, desc_len))

    for k in range(n_kps):
        x0 = int(round(float(kps_aug[k, 0])))
        y0 = int(round(float(kps_aug[k, 1])))
        orient = float(kps_aug[k, -1])

        x1 = max(0, x0 - W);  x2 = min(x0 + W, cols - 1)
        y1 = max(0, y0 - W);  y2 = min(y0 + W, rows - 1)

        oy = W + y1 - y0;  ox = W + x1 - x0
        mom_o_patch = mom_o[y1:y2 + 1, x1:x2 + 1]

        # angle_bin: 0 = skip; [1, nbo] = orientation bin (1-indexed to match MATLAB)
        angle_bin = np.zeros((ps, ps), dtype=int)
        if rot_flag:
            angle_bin[oy:oy + (y2 - y1 + 1), ox:ox + (x2 - x1 + 1)] = (
                np.floor((mom_o_patch - orient) * nbo / np.pi) % nbo
            ).astype(int) + 1
            Theta = (
                np.floor((Theta0 - orient) * nba / (2.0 * np.pi)) % nba
            ).astype(int) + 1
        else:
            angle_bin[oy:oy + (y2 - y1 + 1), ox:ox + (x2 - x1 + 1)] = np.clip(
                np.floor(mom_o_patch * nbo / np.pi).astype(int) + 1, 1, nbo
            )
            Theta = Theta_norot

        weight_p = np.zeros((ps, ps))
        weight_p[oy:oy + (y2 - y1 + 1), ox:ox + (x2 - x1 + 1)] = (
            mom_m[y1:y2 + 1, x1:x2 + 1]
        )

        feat_center = np.zeros(nbo)
        feat_outer  = np.zeros((nbo, nba, 3))

        # Vectorised accumulation over the patch
        a_flat = angle_bin.ravel()    # [1, nbo] or 0 (invalid)
        t_flat = Theta.ravel()        # [1, nba]
        w_flat = weight_p.ravel()

        valid = (a_flat >= 1) & (Rho_flat >= 0) & (Rho_flat <= 2)

        # Center zone (Rho == 0)
        cm = valid & (Rho_flat == 0)
        np.add.at(feat_center, a_flat[cm] - 1, w_flat[cm])

        # Inner (Rho==1) and outer (Rho==2) rings → feat_outer[:, :, 0] and [:, :, 1]
        om = valid & (Rho_flat >= 1)
        ao = a_flat[om] - 1          # [0, nbo-1]
        to = t_flat[om] - 1          # [0, nba-1]
        ro = Rho_flat[om] - 1        # 0 or 1
        np.add.at(feat_outer, (ao, to, ro), w_flat[om])

        # ---------------------------------------------------------------- #
        # OCD: orientation-consistency discrimination (rot_flag only)       #
        # If the first NBA/2 angular bins have higher variance than the     #
        # second half, flip the ordering to correct descriptor reversal.    #
        # ---------------------------------------------------------------- #
        if rot_flag:
            half = nba // 2
            des_H1 = feat_outer[:, :half, :2]
            des_H2 = feat_outer[:, half:, :2]
            if np.var(des_H1) > np.var(des_H2):
                feat_outer[:, :, :2] = np.concatenate([des_H2, des_H1], axis=1)

        # ---------------------------------------------------------------- #
        # Deeper feature: circumferential neighbor averaging + center blend #
        # feat_outer[:, :, 2] = weighted combination of rings + center     #
        # ---------------------------------------------------------------- #
        deeper = (
            (feat_outer[:, :, 0] + feat_outer[:, c_idx, 0]) * w_ratio1 +
            (feat_outer[:, :, 1] + feat_outer[:, c_idx, 1]) * w_ratio2
        ) / 2.0
        feat_outer[:, :, 2] = (deeper + feat_center[:, np.newaxis]) / 3.0 / 2.0

        feat_all = feat_outer[:, :, 2].sum(axis=1) / nba   # (nbo,)

        # Descriptor vector: [feat_center | feat_outer(:) col-major | feat_all]
        # feat_outer.ravel('F') matches MATLAB's column-major feat_outer(:)
        descriptor[k] = np.concatenate([
            feat_center,
            feat_outer.ravel(order='F'),
            feat_all,
        ])

    return np.concatenate([kps_aug, descriptor], axis=1)


# =========================================================================== #
# Smoke test                                                                   #
# =========================================================================== #
if __name__ == '__main__':
    import sys

    rng = np.random.default_rng(42)

    H, W_img = 256, 256
    NBA, NBO = 12, 12
    PATCH = 72
    DESC_LEN = (2 + 3 * NBA) * NBO       # 456
    W_half = PATCH // 2                   # 36

    # Synthetic MOM maps: smooth sinusoidal orientation, uniform magnitude
    yy, xx = np.mgrid[0:H, 0:W_img]
    mom_o = (np.sin(xx / 30.0) * np.cos(yy / 30.0) * 0.5 + 0.5) * (np.pi - 1e-6)
    mom_m = np.ones((H, W_img))

    # 10 keypoints safely inside the image (away from boundaries by W_half+5)
    margin = W_half + 5
    kps = rng.integers(margin, min(H, W_img) - margin, size=(10, 2)).astype(float)

    print(f"MOM map: shape={mom_o.shape}  range=[{mom_o.min():.3f}, {mom_o.max():.3f}]")
    print(f"Keypoints: {len(kps)} points")
    print(f"Expected descriptor length per keypoint: {DESC_LEN}")

    for label, rot in (('rot_flag=False', False), ('rot_flag=True', True)):
        desc = gpolar_descriptor(mom_m, mom_o, kps, PATCH, NBA, NBO, rot_flag=rot)

        # Shape: (K_aug, 3 + DESC_LEN) — 3 = x,y,orient in first cols
        assert desc.ndim == 2, f"[{label}] expected 2-D output"
        assert desc.shape[0] >= 1, f"[{label}] no descriptors returned"
        assert desc.shape[1] == 3 + DESC_LEN, (
            f"[{label}] wrong width: {desc.shape[1]} != {3 + DESC_LEN}"
        )

        vectors = desc[:, 3:]   # drop kp columns
        assert np.all(np.isfinite(vectors)), f"[{label}] non-finite descriptor values"
        nonzero_rows = np.any(vectors != 0, axis=1).sum()
        assert nonzero_rows == len(desc), (
            f"[{label}] {len(desc) - nonzero_rows} all-zero descriptor(s)"
        )

        print(
            f"[{label}] OK — K_aug={desc.shape[0]}  "
            f"desc_cols={vectors.shape[1]}  "
            f"finite={np.isfinite(vectors).all()}  "
            f"non-zero rows={nonzero_rows}/{len(desc)}"
        )

    print("Smoke test passed.")
    sys.exit(0)
