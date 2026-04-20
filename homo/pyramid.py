"""
Python port of Build_Homo_Pyramid.m from HOMO_Feature_ImgMatching.

Original MATLAB by Gao Chenzhong (gao-pingqi@qq.com).
Builds the multi-octave/multi-layer MOM (Major Orientation Map) pyramid
and the DoM (Difference of MOM) pyramid used for homogeneous-region weighting.
"""

import math

import cv2
import numpy as np
from scipy.ndimage import correlate

from mom import _fspecial_gaussian, major_orientation_map


# --------------------------------------------------------------------------- #
# Get_Gaussian_Scale: incremental sigmas for Gaussian scale space              #
# --------------------------------------------------------------------------- #

def _get_gaussian_scale(sigma: float, num_layers: int) -> list:
    """
    Matches Get_Gaussian_Scale.m.

    Returns incremental sigmas so that convolving each layer's image with
    its sigma yields the desired total scale:
      layer 0 : total sigma = sigma           (no extra blur)
      layer i : total sigma = k^i * sigma     (i > 0)
    where k = 2^(1/(num_layers-1)).

    sig[i] is the *incremental* sigma to apply to layer i's image.
    """
    sig = [0.0] * num_layers
    sig[0] = sigma
    if num_layers < 2:
        return sig
    k = 2.0 ** (1.0 / (num_layers - 1))
    for i in range(1, num_layers):
        sig_prev = k ** (i - 1) * sigma
        sig_curr = k * sig_prev           # = k^i * sigma
        sig[i] = math.sqrt(sig_curr ** 2 - sig_prev ** 2)
    return sig


# --------------------------------------------------------------------------- #
# Gaussian_Scaling: apply incremental Gaussian blur to the running image       #
# --------------------------------------------------------------------------- #

def _gaussian_scaling(I_t: np.ndarray, layer_idx: int, sig: float) -> np.ndarray:
    """
    Matches Gaussian_Scaling.m.
    layer_idx == 0 (MATLAB layer == 1): no smoothing, return as-is.
    layer_idx  > 0: apply Gaussian with sigma=sig,
                    window = 2*round(3*sig)+1  (same as MATLAB round).
    """
    if layer_idx == 0:
        return I_t
    w_half = int(round(3.0 * sig))
    w_size = 2 * w_half + 1
    kernel = _fspecial_gaussian(w_size, sig)
    return correlate(I_t, kernel, mode='nearest')


# --------------------------------------------------------------------------- #
# Build_Homo_Pyramid                                                            #
# --------------------------------------------------------------------------- #

def build_homo_pyramid(
    I: np.ndarray,
    n_octaves: int,
    n_layers: int,
    g_resize: float,
    g_sigma: float,
    patch_size: int,
    nba: int,
    int_flag: bool,
    key_type: str,
):
    """
    Build the HOMO MOM pyramid and DoM pyramid.

    Matches Build_Homo_Pyramid.m exactly.

    Parameters
    ----------
    I          : input 2-D grayscale image (float)
    n_octaves  : number of octaves
    n_layers   : number of layers per octave
    g_resize   : downscaling factor between octaves (>1, e.g. 1.2)
    g_sigma    : base Gaussian sigma for scale-space construction
    patch_size : descriptor patch size; sets ASLG window radius W = floor(patch_size/2)
    nba        : number of bins per axis; r = sqrt(W^2/(2*nba+1))
    int_flag   : passed directly to major_orientation_map
    key_type   : detector type string; if it contains 'free' (case-insensitive)
                 DoM is skipped

    Returns
    -------
    mom_pyr : list[n_octaves][n_layers] of orientation maps (values in [0, pi))
    dom_pyr : list[n_octaves][n_layers] of DoM maps (or None where not computed)
              DoM is only filled at index [octave][n_layers-2] (the second-to-last
              layer), matching the MATLAB post-loop assignment.
    """
    I = np.asarray(I, dtype=float)
    if I.ndim == 3:
        I = I[:, :, 0]
    rows, cols = I.shape

    dom_flag = 'free' not in key_type.lower()   # ~contains(lower(key_type),'free')

    # r = sqrt(W^2 / (2*NBA+1))  — ASLG outer radius passed to MOM
    W = int(math.floor(patch_size / 2))
    r = math.sqrt(W ** 2 / (2 * nba + 1))

    sig = _get_gaussian_scale(g_sigma, n_layers)

    # Initialise pyramids as lists of None
    mom_pyr = [[None] * n_layers for _ in range(n_octaves)]
    dom_pyr = [[None] * n_layers for _ in range(n_octaves)]

    for octave in range(n_octaves):
        # Downscale image for this octave  (matches imresize(I, 1/G_resize^(octave-1)))
        scale = 1.0 / (g_resize ** octave)
        new_w = max(1, int(round(cols * scale)))
        new_h = max(1, int(round(rows * scale)))
        I_t = cv2.resize(I, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        for layer_idx in range(n_layers):
            # Incremental Gaussian blur — I_t is modified in-place each layer
            I_t = _gaussian_scaling(I_t, layer_idx, sig[layer_idx])

            # Major Orientation Map — R1=1, R2=r, s=4
            _, orientation = major_orientation_map(I_t, R1=1.0, R2=r, s=4, int_flag=int_flag)
            mom_pyr[octave][layer_idx] = orientation

        # DoM: difference between the last two layers, stored at second-to-last slot.
        # MATLAB uses `layer` (= nLayers) after loop → DoM_pyr{octave, nLayers-1}.
        # Python: index n_layers-2, using layers n_layers-2 and n_layers-1.
        if dom_flag and n_layers > 1:
            last = n_layers - 1
            diff = np.abs(mom_pyr[octave][last - 1] - mom_pyr[octave][last])
            dom_pyr[octave][last - 1] = np.pi / 2.0 - np.abs(diff - np.pi / 2.0)

    return mom_pyr, dom_pyr


# ======================================================================== #
# Smoke test                                                                 #
# ======================================================================== #
if __name__ == "__main__":
    import sys

    # Build a 256x256 test image: Gaussian blob
    size = 256
    cx, cy = size // 2, size // 2
    gx = np.arange(size) - cx
    gy = np.arange(size) - cy
    xx, yy = np.meshgrid(gx, gy)
    img = np.exp(-(xx ** 2 + yy ** 2) / (2 * 40.0 ** 2))

    n_octaves  = 3
    n_layers   = 2
    g_resize   = 1.2
    g_sigma    = 1.6
    patch_size = 72
    nba        = 12
    int_flag   = True
    key_type   = 'PC-ShiTomasi'

    print(f"Input : shape={img.shape}")
    mom_pyr, dom_pyr = build_homo_pyramid(
        img, n_octaves, n_layers, g_resize, g_sigma,
        patch_size, nba, int_flag, key_type,
    )

    # --- shape check ---
    assert len(mom_pyr) == n_octaves, f"Wrong n_octaves: {len(mom_pyr)}"
    assert all(len(row) == n_layers for row in mom_pyr), "Wrong n_layers"
    print(f"Pyramid shape: ({len(mom_pyr)}, {len(mom_pyr[0])})  [expected (3, 2)]  OK")

    # --- orientation range check ---
    for o in range(n_octaves):
        for l in range(n_layers):
            ori = mom_pyr[o][l]
            assert ori is not None, f"mom_pyr[{o}][{l}] is None"
            assert ori.ndim == 2, f"mom_pyr[{o}][{l}] is not 2-D"
            lo, hi = float(ori.min()), float(ori.max())
            assert lo >= 0 and hi < math.pi, \
                f"mom_pyr[{o}][{l}] out of [0,pi): [{lo:.4f}, {hi:.4f}]"
            print(f"  mom_pyr[{o}][{l}]: shape={ori.shape}  "
                  f"range=[{lo:.4f}, {hi:.4f}]  OK")

    # --- DoM check (second-to-last layer slot) ---
    for o in range(n_octaves):
        dom = dom_pyr[o][0]           # n_layers-2 = 0 for n_layers=2
        assert dom is not None, f"dom_pyr[{o}][0] is None"
        lo, hi = float(dom.min()), float(dom.max())
        assert lo >= 0 and hi <= math.pi / 2 + 1e-9, \
            f"dom_pyr[{o}][0] out of [0, pi/2]: [{lo:.4f}, {hi:.4f}]"
        print(f"  dom_pyr[{o}][0]: shape={dom.shape}  range=[{lo:.4f}, {hi:.4f}]  OK")
    assert dom_pyr[0][1] is None, "dom_pyr[0][1] should be None"

    print("build_homo_pyramid smoke test passed.")
    sys.exit(0)
