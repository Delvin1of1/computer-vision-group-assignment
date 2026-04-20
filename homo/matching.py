"""
Python port of Match_Keypoint.m, Outlier_Removal.m, and the non-parallel
(version 2) of Multiscale_Strategy.m from HOMO_Feature_ImgMatching.

Original MATLAB by Gao Chenzhong (gao-pingqi@qq.com).

Public API
----------
multiscale_strategy(kps_1, kps_2, mom_pyr1, mom_pyr2,
                    patch_size, nba, nbo, g_resize,
                    error, K, t_form, rot_flag, scl_flag=False)
    -> (cor1, cor2)   both (M, 6) float arrays [x,y,xt,yt,orient,idx]
    len(cor1) is the RANSAC inlier count (NCM).

Column conventions (0-indexed Python)
--------------------------------------
After GPolar + reorder  descriptor columns:  [x, y, xt, yt, orient, idx,  des...]
                                               0  1   2   3     4    5   6…end
matches (12 cols):                            [cor1_6cols | cor2_6cols]
    idx of image-1 keypoint:  col  5
    idx of image-2 keypoint:  col 11
"""

import os
import sys

import numpy as np
from scipy.spatial import cKDTree

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from descriptor import gpolar_descriptor
from fsc import fsc


# =========================================================================== #
# sample_out                                                                   #
# =========================================================================== #

def _sample_out(
    samples: np.ndarray,
    id_list: np.ndarray,
    col_0idx: int,
) -> np.ndarray:
    """
    sample_out.m: delete rows whose value at col_0idx is in id_list.

    Parameters
    ----------
    samples   : (N, C) array
    id_list   : 1-D array of IDs to remove
    col_0idx  : 0-indexed column that holds the ID (MATLAB loc - 1)
    """
    if samples is None or len(samples) == 0:
        return samples
    if len(id_list) == 0:
        return samples
    if col_0idx >= samples.shape[1]:
        return samples
    mask = np.isin(samples[:, col_0idx], id_list)
    return samples[~mask]


# =========================================================================== #
# NN matching                                                                  #
# =========================================================================== #

def _nn_match(des1: np.ndarray, des2: np.ndarray):
    """
    matchFeatures equivalent — mutual (bidirectional) nearest-neighbour matching
    on L2-normalised descriptor vectors.

    MATLAB matchFeatures normalises descriptors by L2 norm before computing
    distances.  For cross-modal matching the un-normalised true-match descriptor
    distance can exceed the typical NN distance, causing incorrect ranking.
    Normalisation fixes this by converting the metric to cosine distance.

    Mutual NN (keep pair (i,j) only when des2[j] is also nearest to des1[i])
    gives fewer but higher-precision pairs, improving the RANSAC inlier fraction.

    Returns (idx1, idx2) — index arrays into des1 and des2.
    """
    if len(des1) == 0 or len(des2) == 0:
        return np.empty(0, int), np.empty(0, int)

    # L2-normalise to unit vectors before distance computation
    def _l2norm(d):
        n = np.linalg.norm(d, axis=1, keepdims=True)
        n = np.where(n == 0, 1.0, n)
        return d / n

    des1n = _l2norm(des1.astype(float))
    des2n = _l2norm(des2.astype(float))

    # Forward: des1[i] → nearest in des2
    tree2      = cKDTree(des2n)
    _, nn_fwd  = tree2.query(des1n, k=1, workers=-1)

    # Backward: des2[j] → nearest in des1
    tree1      = cKDTree(des1n)
    _, nn_bwd  = tree1.query(des2n, k=1, workers=-1)

    # Mutual NN only — let FSC filter outliers rather than pre-filtering with ratio
    n1     = len(des1)
    mutual = nn_bwd[nn_fwd] == np.arange(n1)
    idx1   = np.where(mutual)[0].astype(int)
    idx2   = nn_fwd[idx1].astype(int)
    return idx1, idx2


# =========================================================================== #
# Outlier_Removal                                                              #
# =========================================================================== #

def _outlier_removal(
    cor1: np.ndarray,
    cor2: np.ndarray,
    error: float,
    max_iter: int,
    t_form: str,
    ref=None,
):
    """
    Outlier_Removal.m: run FSC then reproject to build the final inlier mask.

    Parameters
    ----------
    cor1, cor2 : (N, ≥2) matched keypoint arrays; first 2 cols are x, y
    error      : inlier pixel threshold
    max_iter   : RANSAC iterations (passed to fsc)
    t_form     : 'similarity'|'affine'|'projective'
    ref        : None  or  (ref1, ref2) each (R, ≥2).
                 ref1[:,0:2] / ref2[:,0:2] are used inside FSC to anchor the fit.
                 After FSC the FULL ref arrays are appended to cor1/cor2 before
                 the reprojection check — matching MATLAB exactly.

    Returns
    -------
    (cor1_inliers, cor2_inliers, inlier_mask) or (None, None, None) on failure.
    inlier_mask is a bool array of length  len(cor1) [+ len(ref) when ref given].
    cor1_inliers / cor2_inliers are the filtered rows (original + surviving ref).
    """
    if len(cor1) < 20:
        return None, None, None

    # Prepare ref for FSC (only xy needed inside FSC)
    ref_clean = None
    if ref is not None:
        r1, r2 = ref[0], ref[1]
        if r1 is not None and np.asarray(r1).shape[0] > 0:
            ref_clean = (np.asarray(r1, dtype=float),
                         np.asarray(r2, dtype=float))

    H, _, cor1_fsc, _ = fsc(
        cor1[:, :2].astype(float),
        cor2[:, :2].astype(float),
        t_form, error, max_iter,
        ref=ref_clean,
    )

    if H is None or len(cor1_fsc) < 4:
        return None, None, None

    # Append ref AFTER FSC — matches MATLAB:
    #   if ~isempty(ref), cor1=[cor1;ref{1}]; cor2=[cor2;ref{2}]; end
    if ref_clean is not None:
        cor1_ext = np.vstack([cor1, ref_clean[0]])
        cor2_ext = np.vstack([cor2, ref_clean[1]])
    else:
        cor1_ext = cor1
        cor2_ext = cor2

    # Reproject cor1_ext through H and keep pairs within error pixels
    n_ext = len(cor1_ext)
    pts  = np.vstack([cor1_ext[:, :2].T, np.ones(n_ext)])   # (3, n_ext)
    proj = H @ pts                                            # (3, n_ext)
    with np.errstate(divide='ignore', invalid='ignore'):
        proj_xy = proj[:2] / proj[2:3]                      # perspective divide (safe for affine)
    E = np.sqrt(np.sum((proj_xy - cor2_ext[:, :2].T) ** 2, axis=0))
    inlier_mask = E < error

    cor1_in = cor1_ext[inlier_mask]
    cor2_in = cor2_ext[inlier_mask]
    return cor1_in, cor2_in, inlier_mask


# =========================================================================== #
# Match_Keypoint                                                               #
# =========================================================================== #

def _match_keypoint(
    desc1: np.ndarray,
    desc2: np.ndarray,
    error: float,
    K: int,
    t_form: str,
    ref=None,
):
    """
    Match_Keypoint.m: NN matching followed by K rounds of outlier removal.

    desc1 / desc2 : (N, 6+D) arrays with columns [x,y,xt,yt,orient,idx, des...]
    ref           : None or (ref1, ref2) each (R, 6) [x,y,xt,yt,orient,idx]

    Returns
    -------
    (matches, ncm)
    matches : (M, 12) = [cor1_6cols | cor2_6cols], or None
    ncm     : number of RANSAC inliers (0 on failure)
    """
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        return None, 0

    kps1 = desc1[:, :6]
    des1 = desc1[:, 6:]
    kps2 = desc2[:, :6]
    des2 = desc2[:, 6:]

    # NN match — equivalent to MATLAB matchFeatures(MaxRatio=1, MatchThreshold=100)
    idx1, idx2 = _nn_match(des1, des2)
    if len(idx1) < 20:
        return None, 0

    cor1 = kps1[idx1]   # (num_matches, 6)
    cor2 = kps2[idx2]   # (num_matches, 6)

    # K rounds of outlier removal; keep best
    # MATLAB: NCMs = zeros(K,1); for k=1:K ... NCMs(k) = sum(indexPairs{k}); end
    NCMs    = np.zeros(K, dtype=int)
    results = [None] * K

    for k in range(K):
        c1, c2, mask = _outlier_removal(cor1, cor2, error, 2000, t_form, ref)
        if mask is not None and c1 is not None:
            NCMs[k]    = int(mask.sum())
            results[k] = (c1, c2)

    best_k   = int(np.argmax(NCMs))
    num_keys = int(NCMs[best_k])

    # MATLAB: if(num_keys<4), num_keys=0; matches=[]; return; end
    if num_keys < 4 or results[best_k] is None:
        return None, 0

    cor1_out, cor2_out = results[best_k]
    matches = np.hstack([cor1_out, cor2_out])   # (M, 12)
    return matches, num_keys


# =========================================================================== #
# GPolar descriptor column reorder                                             #
# =========================================================================== #

def _reorder(d: np.ndarray) -> np.ndarray:
    """
    Reorder GPolar output from [xt,yt,x,y,idx,orient,des...]
    to                         [x,y,xt,yt,orient,idx,des...].

    MATLAB: descriptor(:,[3,4,1,2,6,5,7:end])  (1-indexed)
    Python: [:, [2,3,0,1,5,4] + list(range(6,total))]
    """
    if d is None or len(d) == 0:
        return d
    total     = d.shape[1]
    col_order = [2, 3, 0, 1, 5, 4] + list(range(6, total))
    return d[:, col_order]


# =========================================================================== #
# Multiscale_Strategy — version 2 (non-parallel, line-by-line translation)    #
# =========================================================================== #

def multiscale_strategy(
    kps_1: np.ndarray,
    kps_2: np.ndarray,
    mom_pyr1: list,
    mom_pyr2: list,
    patch_size: int,
    nba: int,
    nbo: int,
    g_resize: float,
    error: float,
    K: int,
    t_form: str,
    rot_flag: bool,
    scl_flag: bool = False,
):
    """
    Multiscale_Strategy.m — version 2 (non-parallel iterative).

    Parameters
    ----------
    kps_1, kps_2 : (N, 2) keypoint arrays [x, y] (0-indexed image coords)
    mom_pyr1/2   : list[nOctaves][nLayers] of (H,W) MOM orientation maps
    patch_size   : GPolar patch diameter (e.g. 72)
    nba          : angular grid bins (e.g. 12)
    nbo          : orientation bins (e.g. 12)
    g_resize     : pyramid downscale factor per octave (e.g. 1.2)
    error        : RANSAC / reprojection error threshold in pixels (e.g. 5)
    K            : RANSAC repetitions per Match_Keypoint call (e.g. 1)
    t_form       : 'similarity' | 'affine' | 'projective'
    rot_flag     : rotation-invariant descriptors
    scl_flag     : cross-scale matching (False = same octave only)

    Returns
    -------
    cor1, cor2 : (M, 6) float arrays  [x, y, xt, yt, orient, idx]
                 len(cor1) == RANSAC inlier count (NCM)
    """
    kps_1 = np.asarray(kps_1, dtype=float)
    kps_2 = np.asarray(kps_2, dtype=float)

    # ------------------------------------------------------------------ #
    # Initialization — MATLAB: kps_1 = [kps_1(:,1:2), (1:N)']           #
    # Give each keypoint a unique integer ID (0-indexed)                  #
    # ------------------------------------------------------------------ #
    n1    = len(kps_1)
    n2    = len(kps_2)
    kps_1 = np.column_stack([kps_1[:, :2], np.arange(n1, dtype=float)])  # [x,y,idx]
    kps_2 = np.column_stack([kps_2[:, :2], np.arange(n2, dtype=float)])  # [x,y,idx]

    n_octaves = len(mom_pyr1)
    n_layers  = len(mom_pyr1[0])

    # Build per-octave scaled keypoints
    # MATLAB: kps_1t = round(kps_1(:,1:2)./G_resize^(octave-1))
    #         keypoints_1{octave} = [kps_1t(idx_1,:), kps_1(idx_1,:)]
    # → result is [xt, yt, x, y, idx]  (5 cols)
    keypoints_1 = [None] * n_octaves
    keypoints_2 = [None] * n_octaves
    img_size1   = [None] * n_octaves
    img_size2   = [None] * n_octaves

    for oc in range(n_octaves):
        scale  = g_resize ** oc                                    # MATLAB: G_resize^(octave-1)
        kps1t  = np.round(kps_1[:, :2] / scale)
        kps2t  = np.round(kps_2[:, :2] / scale)
        # unique('rows') — keep one representative per unique scaled position
        _, u1  = np.unique(kps1t, axis=0, return_index=True)
        _, u2  = np.unique(kps2t, axis=0, return_index=True)
        keypoints_1[oc] = np.hstack([kps1t[u1], kps_1[u1]])       # (K1, 5) [xt,yt,x,y,idx]
        keypoints_2[oc] = np.hstack([kps2t[u2], kps_2[u2]])       # (K2, 5)
        img_size1[oc]   = mom_pyr1[oc][0].shape                    # (H, W)
        img_size2[oc]   = mom_pyr2[oc][0].shape

    # 4-D match storage — MATLAB: matches = cell(nOctaves,nLayers,nOctaves,nLayers)
    matches_store    = {}
    confidence_store = np.zeros((n_octaves, n_layers, n_octaves, n_layers))

    # ------------------------------------------------------------------ #
    # Version 2: iterative multiscale matching                            #
    # ------------------------------------------------------------------ #
    descriptors_1 = [[None] * n_layers for _ in range(n_octaves)]
    descriptors_2 = [[None] * n_layers for _ in range(n_octaves)]
    idx_1 = np.empty(0, dtype=float)
    idx_2 = np.empty(0, dtype=float)
    ref   = (None, None)

    # MATLAB: for octave2 = nOctaves:-1:1   (coarse → fine)
    for octave2 in range(n_octaves - 1, -1, -1):
        kps_2_cur = keypoints_2[octave2].copy()            # fresh per octave2
        mag_map2  = np.ones(img_size2[octave2])

        # MATLAB: for octave1=nOctaves:-1:1; if ~scl_flag, octave1=octave2; end ... break
        # Effect: for scl_flag=False only octave1=octave2 is processed
        octave1_range = range(n_octaves - 1, -1, -1) if scl_flag else [octave2]

        for octave1 in octave1_range:
            kps_1_cur = keypoints_1[octave1].copy()        # fresh per octave1
            mag_map1  = np.ones(img_size1[octave1])

            # MATLAB: for layer2=1:nLayers
            for layer2 in range(n_layers):
                desc2 = descriptors_2[octave2][layer2]

                if desc2 is None:
                    # First visit: compute, filtering already-matched kps first
                    # MATLAB: kps_2 = sample_out(kps_2, idx_2, 5)  [1-indexed col 5 = idx]
                    kps_2_cur = _sample_out(kps_2_cur, idx_2, col_0idx=4)
                    raw2 = gpolar_descriptor(
                        mag_map2,
                        mom_pyr2[octave2][layer2],
                        kps_2_cur,
                        patch_size, nba, nbo, rot_flag,
                    )                                           # [xt,yt,x,y,idx,orient,des...]
                    desc2 = _reorder(raw2)                      # [x,y,xt,yt,orient,idx,des...]
                else:
                    # Already computed: filter matched descriptors by idx
                    # MATLAB: descriptor2 = sample_out(descriptor2, idx_2, 6)  [col 6 = idx]
                    desc2 = _sample_out(desc2, idx_2, col_0idx=5)

                descriptors_2[octave2][layer2] = desc2

                # MATLAB: for layer1=1:nLayers
                for layer1 in range(n_layers):
                    desc1 = descriptors_1[octave1][layer1]

                    if desc1 is None:
                        kps_1_cur = _sample_out(kps_1_cur, idx_1, col_0idx=4)
                        # MATLAB: if size(kps_1,1)<3, continue; end
                        if len(kps_1_cur) < 3:
                            continue
                        raw1 = gpolar_descriptor(
                            mag_map1,
                            mom_pyr1[octave1][layer1],
                            kps_1_cur,
                            patch_size, nba, nbo, rot_flag,
                        )
                        desc1 = _reorder(raw1)
                    else:
                        desc1 = _sample_out(desc1, idx_1, col_0idx=5)

                    descriptors_1[octave1][layer1] = desc1

                    # Guard empty descriptors before matching
                    if desc1 is None or len(desc1) == 0:
                        continue
                    if desc2 is None or len(desc2) == 0:
                        continue

                    match, conf = _match_keypoint(
                        desc1, desc2, error, K, t_form, ref,
                    )
                    matches_store[(octave1, layer1, octave2, layer2)] = match
                    confidence_store[octave1, layer1, octave2, layer2] = conf

                    # Update spatial reference when enough inliers found
                    # MATLAB: if size(match,1)>50
                    if match is not None and len(match) > 50:
                        idx_1 = np.unique(match[:, 5])    # col 5 = idx of image-1 kp
                        idx_2 = np.unique(match[:, 11])   # col 11 = idx of image-2 kp
                        ref   = (match[:, :6], match[:, 6:12])

    # ------------------------------------------------------------------ #
    # Optimizing — aggregate matches per (octave1, octave2) pair          #
    # ------------------------------------------------------------------ #
    Confidence = np.zeros((n_octaves, n_octaves))
    Matches    = {}

    for o1 in range(n_octaves):
        for o2 in range(n_octaves):
            parts = []
            for l1 in range(n_layers):
                for l2 in range(n_layers):
                    m = matches_store.get((o1, l1, o2, l2))
                    if m is not None and len(m) > 0:
                        parts.append(m)

            if not parts:
                continue
            matches_t = np.vstack(parts)

            if len(matches_t) > 20:
                # Remove duplicate image-1 positions — unique on [x1,y1]
                # MATLAB: [~,idx,~] = unique(matches_t(:,1:2),'rows')
                _, uid1 = np.unique(matches_t[:, :2],  axis=0, return_index=True)
                matches_t = matches_t[uid1]
                # Remove duplicate image-2 positions — unique on [x2,y2]
                _, uid2 = np.unique(matches_t[:, 6:8], axis=0, return_index=True)
                matches_t = matches_t[uid2]

            if len(matches_t) > 20:
                Matches[(o1, o2)] = matches_t
                Confidence[o1, o2] = len(matches_t)

    if Confidence.max() == 0:
        return np.empty((0, 6), float), np.empty((0, 6), float)

    # Find octave pair with highest confidence
    # MATLAB: [max_O1,max_O2] = find(Confidence==max(max(Confidence)))
    rows_idx, cols_idx = np.where(Confidence == Confidence.max())
    # Use first element when multiple ties (MATLAB find returns column-major order)
    max_O1_m = int(rows_idx[0]) + 1   # convert to MATLAB 1-indexed
    max_O2_m = int(cols_idx[0]) + 1

    # Collect diagonal neighbours around the best octave pair
    # MATLAB: for i = 1-min(max_O1,max_O2) : min(nOctaves-max_O1, nOctaves-max_O2)
    #           Matches{max_O1+i, max_O2+i}
    min_o   = min(max_O1_m, max_O2_m)
    max_off = min(n_octaves - max_O1_m, n_octaves - max_O2_m)
    MMatches_parts = []
    for i in range(1 - min_o, max_off + 1):
        o1_py = (max_O1_m + i) - 1      # back to Python 0-indexed
        o2_py = (max_O2_m + i) - 1
        mt = Matches.get((o1_py, o2_py))
        if mt is not None and len(mt) > 3:
            MMatches_parts.append(mt)

    if not MMatches_parts:
        return np.empty((0, 6), float), np.empty((0, 6), float)

    MMatches = np.vstack(MMatches_parts)
    _, uid1  = np.unique(MMatches[:, :2],  axis=0, return_index=True)
    MMatches = MMatches[uid1]
    _, uid2  = np.unique(MMatches[:, 6:8], axis=0, return_index=True)
    MMatches = MMatches[uid2]

    # ------------------------------------------------------------------ #
    # One last outlier removal — K repetitions, keep best               #
    # MATLAB: [~,~,indexPairs{k}] = Outlier_Removal(MMatches(:,1:6),   #
    #                                MMatches(:,7:end),Error,800,...)   #
    # ------------------------------------------------------------------ #
    cor1_part = MMatches[:, :6]
    cor2_part = MMatches[:, 6:]

    NCMs_final  = np.zeros(K, dtype=int)
    masks_final = [None] * K

    for k in range(K):
        _, _, mask = _outlier_removal(
            cor1_part, cor2_part, error, 2000, t_form, ref=None,
        )
        if mask is not None:
            NCMs_final[k]  = int(mask.sum())
            masks_final[k] = mask

    best_k = int(np.argmax(NCMs_final))
    if NCMs_final[best_k] < 4:
        return np.empty((0, 6), float), np.empty((0, 6), float)

    best_mask = masks_final[best_k]
    MMatches  = MMatches[best_mask]
    cor1      = MMatches[:, :6]
    cor2      = MMatches[:, 6:]
    return cor1, cor2


# =========================================================================== #
# Smoke test — LLVIP visible + infrared pair                                   #
# =========================================================================== #
if __name__ == '__main__':
    import time
    import cv2

    from pyramid  import build_homo_pyramid
    from keypoints import detect_homo_keypoint

    _root     = os.path.join(_here, '..')
    ir_path   = os.path.join(_root, 'LLVIP', 'infrared', 'test', '190001.jpg')
    vis_path  = os.path.join(_root, 'LLVIP', 'visible',  'test', '190001.jpg')

    def _load_gray(path):
        bgr = cv2.imread(path)
        if bgr is None:
            raise FileNotFoundError(path)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(float) / 255.0

    print('Loading images…')
    I1 = _load_gray(ir_path)
    I2 = _load_gray(vis_path)
    print(f'  IR  shape: {I1.shape}')
    print(f'  Vis shape: {I2.shape}')

    # HOMO parameters from A_HOMO_demo.m
    N_OCT   = 3;  N_LAY   = 2
    G_RSZ   = 1.2; G_SIG  = 1.6
    PATCH   = 72;  NBA    = 12;  NBO = 12
    INT     = True; ROT   = True; KEY = 'PC-ShiTomasi'
    ERR     = 5;   K_REP  = 1
    TFORM   = 'affine'

    t0 = time.time()
    print('Building MOM pyramids…')
    mom1, dom1 = build_homo_pyramid(I1, N_OCT, N_LAY, G_RSZ, G_SIG, PATCH, NBA, INT, KEY)
    mom2, dom2 = build_homo_pyramid(I2, N_OCT, N_LAY, G_RSZ, G_SIG, PATCH, NBA, INT, KEY)
    print(f'  done in {time.time()-t0:.1f}s')

    t1 = time.time()
    print('Detecting keypoints…')
    kps1 = detect_homo_keypoint(I1, dom1, 6, 0, 1, 5000, G_RSZ, KEY)
    kps2 = detect_homo_keypoint(I2, dom2, 6, 0, 1, 5000, G_RSZ, KEY)
    print(f'  kps1={len(kps1)}  kps2={len(kps2)}  ({time.time()-t1:.1f}s)')

    t2 = time.time()
    print('Running multiscale_strategy…')
    cor1, cor2 = multiscale_strategy(
        kps1, kps2, mom1, mom2,
        PATCH, NBA, NBO, G_RSZ,
        ERR, K_REP, TFORM, ROT, scl_flag=False,
    )
    elapsed = time.time() - t2
    ncm = len(cor1)
    print(f'  NCM (RANSAC inliers): {ncm}  ({elapsed:.1f}s)')

    assert ncm > 0,    f'No inliers found — pipeline broken'
    assert ncm < 1000, f'NCM={ncm} > 1000 — RANSAC is not running'
    assert ncm >= 20,  f'NCM={ncm} too low — matching failed'

    print(f'\nSmoke test passed.  NCM={ncm}  (expected 50–300)')
    sys.exit(0)
