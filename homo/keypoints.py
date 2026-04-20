"""
Python port of Detect_Homo_Keypoint.m and its helper sub-functions:
  - ShiTomasi.m / Harris.m  (structure tensor detectors)
  - D2gauss.m               (2D Gaussian kernel)
  - Mask / Image_Pat / Remove_Boundary_Points  (inline helpers)

Original MATLAB by Gao Chenzhong (gao-pingqi@qq.com).
"""

import math

import cv2
import numpy as np
from scipy.ndimage import correlate, maximum_filter, binary_erosion
from scipy.signal import convolve2d

from mom import _fspecial_gaussian
from phase_cong import phase_cong3


# =========================================================================== #
# Internal helpers                                                              #
# =========================================================================== #

def _image_pat(I: np.ndarray, s: int) -> np.ndarray:
    """
    Image_Pat.m: zero-pad I by s pixels on every side.
    Matches MATLAB Image_Pat exactly (zero fill, not reflection).
    """
    m, n = I.shape
    out = np.zeros((m + 2 * s, n + 2 * s), dtype=I.dtype)
    out[s:m + s, s:n + s] = I
    return out


def _d2gauss(n1: int, std1: float, n2: int, std2: float, theta: float) -> np.ndarray:
    """
    D2gauss.m: n2×n1 rotated 2D Gaussian, L2- then L1-normalised.
    Matches the double-normalisation in the original (sqrt(sum(h^2)) then sum(h)).
    """
    X, Y = np.meshgrid(np.arange(1, n1 + 1, dtype=float),
                       np.arange(1, n2 + 1, dtype=float))
    X -= (n1 + 1) / 2.0
    Y -= (n2 + 1) / 2.0

    R = np.array([[math.cos(theta), -math.sin(theta)],
                  [math.sin(theta),  math.cos(theta)]])
    coords = R @ np.vstack([X.ravel(), Y.ravel()])

    def _gauss(x, std):
        return np.exp(-x ** 2 / (2 * std ** 2)) / (std * math.sqrt(2 * math.pi))

    h = _gauss(coords[0], std1) * _gauss(coords[1], std2)
    h = h.reshape(n2, n1)
    h /= math.sqrt(float(np.sum(h ** 2)))   # L2 normalise
    h /= float(h.sum())                      # L1 normalise
    return h


def _mask(I_p: np.ndarray, th: float = -10.0) -> np.ndarray:
    """
    Mask.m: foreground mask for the padded image.
    th=-10 → all non-strictly-negative pixels included (effectively all-ones
    for a normal image); D2gauss convolution spreads valid regions.
    """
    mx = float(I_p.max())
    if mx > 0:
        I_norm = I_p / mx * 255.0
    else:
        I_norm = np.zeros_like(I_p)
    msk = (I_norm > th).astype(float)
    h = _d2gauss(7, 4, 7, 4, 0.0)
    conv = convolve2d(msk, h, mode='same', boundary='fill', fillvalue=0.0)
    return conv > 0.0


def _make_disk(radius: float) -> np.ndarray:
    """
    Circular binary structuring element matching MATLAB strel('disk', radius).
    All pixels with Euclidean distance <= radius from the centre are True.
    """
    r = int(round(radius))
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x ** 2 + y ** 2) <= r ** 2


def _remove_boundary_points(
    kps: np.ndarray,
    msk: np.ndarray,
    s: float,
) -> np.ndarray:
    """
    Remove_Boundary_Points.m: erode the mask by disk(s), keep kps inside.

    kps : (K, 3) array  — [x, y, response], 0-indexed padded-image coordinates
    msk : (H, W) bool array
    s   : disk radius for erosion
    """
    if len(kps) == 0:
        return kps
    disk = _make_disk(s)
    eroded = binary_erosion(msk, structure=disk)
    xs = kps[:, 0].astype(int)
    ys = kps[:, 1].astype(int)
    H, W = eroded.shape
    valid = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    valid[valid] = eroded[ys[valid], xs[valid]]
    return kps[valid]


# --------------------------------------------------------------------------- #
# Detectors                                                                    #
# --------------------------------------------------------------------------- #

_SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
_SOBEL_Y = _SOBEL_X.T


def _structure_tensor_kernel(scale: int):
    """
    Build the weighted Gaussian kernel used by ShiTomasi.m / Harris.m.
    Matches fspecial('gaussian', [scale+1, scale+1], scale/6) .* Wcircle.
    Works correctly for even scale (odd scale is not used in the HOMO demo).
    """
    ksize = scale + 1
    half = ksize // 2           # = W for even scale
    ax = np.arange(ksize) - half
    dx, dy = np.meshgrid(ax, ax)
    wcircle = (dx ** 2 + dy ** 2) < (half + 1) ** 2
    h = _fspecial_gaussian(ksize, scale / 6.0)
    return h * wcircle


def _shi_tomasi(img: np.ndarray, scale: int) -> np.ndarray:
    """
    ShiTomasi.m: minimum eigenvalue of the Sobel structure tensor.

    The MATLAB implementation loops per-pixel; here we vectorise using the
    closed-form minimum eigenvalue of the 2×2 symmetric tensor:
        λ_min = (Gxx + Gyy)/2  −  sqrt(((Gxx − Gyy)/2)² + Gxy²)
    """
    Gx = correlate(img, _SOBEL_X, mode='nearest')
    Gy = correlate(img, _SOBEL_Y, mode='nearest')
    h = _structure_tensor_kernel(scale)
    Gxx = correlate(Gx * Gx, h, mode='nearest')
    Gyy = correlate(Gy * Gy, h, mode='nearest')
    Gxy = correlate(Gx * Gy, h, mode='nearest')
    half_diff = (Gxx - Gyy) / 2.0
    return (Gxx + Gyy) / 2.0 - np.sqrt(half_diff ** 2 + Gxy ** 2)


def _harris(img: np.ndarray, scale: int) -> np.ndarray:
    """
    Harris.m: (det − trace²·k) response using Sobel gradients.
    Formula in .m: (Gxx*Gyy − Gxy²) / (Gxx + Gyy + eps)
    """
    Gx = correlate(img, _SOBEL_X, mode='nearest')
    Gy = correlate(img, _SOBEL_Y, mode='nearest')
    h = _structure_tensor_kernel(scale)
    Gxx = correlate(Gx * Gx, h, mode='nearest')
    Gyy = correlate(Gy * Gy, h, mode='nearest')
    Gxy = correlate(Gx * Gy, h, mode='nearest')
    eps = np.finfo(float).eps
    return (Gxx * Gyy - Gxy ** 2) / (Gxx + Gyy + eps)


# =========================================================================== #
# Main detection function                                                       #
# =========================================================================== #

def detect_homo_keypoint(
    I: np.ndarray,
    dom_pyr: list,
    scale: int,
    thresh: float,
    radius: int,
    N: int,
    g_resize: float,
    key_type: str,
) -> np.ndarray:
    """
    Detect HOMO keypoints.  Matches Detect_Homo_Keypoint.m exactly.

    Parameters
    ----------
    I        : 2-D float image (preprocessed, in [0, 1] or [0, 255])
    dom_pyr  : DoM pyramid from build_homo_pyramid  (list[nOctaves][nLayers])
    scale    : structure tensor window size parameter (e.g. 6)
    thresh   : detector response threshold (e.g. 0)
    radius   : LNMS radius in pixels  (e.g. 1 or 2)
    N        : maximum number of keypoints to return
    g_resize : downsampling ratio used to build the pyramid (e.g. 1.2)
    key_type : detector type string, e.g. 'PC-ShiTomasi'

    Returns
    -------
    kps : (K, 2) float array of [x, y] keypoint coordinates,
          0-indexed, in the original (unpadded) image coordinate system.
          Returns empty (0, 2) array when fewer than 10 keypoints are found.
    """
    I = np.asarray(I, dtype=float)
    if I.ndim == 3:
        I = I[:, :, 0]
    scale = int(scale)
    radius = int(radius)
    imagepat = 5

    # ------------------------------------------------------------------ #
    # Detector-free: regular grid sampling                                 #
    # ------------------------------------------------------------------ #
    if 'free' in key_type.lower():
        Im, In = I.shape
        step = max(math.sqrt(Im * In / N), radius)
        Nn = max(1, int(round(In / step)))
        Nm = max(1, int(round(Im / step)))
        # MATLAB round(In/Nn*((1:Nn)-0.5)) is 1-indexed; subtract 1 for Python
        ind_x = (np.round(In / Nn * (np.arange(Nn) + 0.5)) - 1).astype(int)
        ind_y = (np.round(Im / Nm * (np.arange(Nm) + 0.5)) - 1).astype(int)
        ind_x = np.clip(ind_x, 0, In - 1)
        ind_y = np.clip(ind_y, 0, Im - 1)
        XX, YY = np.meshgrid(ind_x, ind_y)
        kps = np.column_stack([XX.ravel(), YY.ravel()])
        values = I[kps[:, 1], kps[:, 0]]
        kps = kps[values > 0]
        return kps.astype(float)

    # ------------------------------------------------------------------ #
    # Homoness map: product of all non-None DoM layers upsampled to I size #
    # ------------------------------------------------------------------ #
    n_octaves = len(dom_pyr)
    n_layers  = len(dom_pyr[0]) if n_octaves > 0 else 0
    rows, cols = I.shape

    # Foreground mask: ceil(I / max(I)) — 1 where I > 0, 0 where I == 0
    I_max = float(I.max())
    mask = np.ceil(I / I_max) if I_max > 0 else np.zeros_like(I)

    homoness = np.ones((rows, cols))
    if n_layers > 0:
        for o in range(n_octaves):
            for l in range(n_layers):
                dom = dom_pyr[o][l]
                if dom is not None:
                    dom_up = cv2.resize(
                        dom.astype(np.float32),
                        (cols, rows),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    homoness *= dom_up.astype(float)
        exp = 6.0 / (n_octaves * max(n_layers - 1, 1))
        # 1/homoness^exp  (MATLAB allows Inf; we clamp denominator to avoid NaN)
        with np.errstate(divide='ignore', invalid='ignore'):
            homoness = np.where(homoness > 0,
                                1.0 / homoness ** exp,
                                0.0)

    # ------------------------------------------------------------------ #
    # Phase congruency detector (if 'PC' in type)                         #
    # ------------------------------------------------------------------ #
    if 'PC' in key_type:
        pc_M, pc_m, *_ = phase_cong3(
            I, nscale=4, norient=6, min_wave_length=3,
            mult=1.6, sigma_onf=0.75, g=3.0, k=1.0,
        )
        I = (pc_M + pc_m) * mask

    # ------------------------------------------------------------------ #
    # Normalise, pad image and homoness                                    #
    # ------------------------------------------------------------------ #
    I_max = float(I.max())
    if I_max > 0:
        I = I / I_max * 255.0
    I_p      = _image_pat(I,        imagepat)
    msk      = _mask(I_p,           -10.0)
    homoness_p = _image_pat(homoness, imagepat)

    # ------------------------------------------------------------------ #
    # Detector response on padded image                                    #
    # ------------------------------------------------------------------ #
    key_lo = key_type.lower()
    if 'harris' in key_lo:
        value = _harris(I_p, scale)
    elif 'shitomasi' in key_lo:
        value = _shi_tomasi(I_p, scale)
    else:
        value = np.zeros_like(I_p)

    # Zero out border region
    # MATLAB: border = imagepat + max(scale, radius)*2 + 1
    # Then rows 1:border+1 and end-border:end, same for cols (1-indexed).
    # Python: [:border+1] and [-(border+1):]
    border = int(imagepat + max(scale, radius) * 2 + 1)
    value[:border + 1, :]      = 0.0
    value[-(border + 1):, :]   = 0.0
    value[:, :border + 1]      = 0.0
    value[:, -(border + 1):]   = 0.0

    # Weight by homoness map
    value = value * homoness_p

    # ------------------------------------------------------------------ #
    # Local non-maximum suppression + threshold                            #
    # ordfilt2(value, sze^2, ones(sze))  →  maximum_filter(size=sze)      #
    # ------------------------------------------------------------------ #
    sze = 2 * radius + 1
    mx = maximum_filter(value, size=sze, mode='constant', cval=0.0)
    value_t = (value == mx) & (value > thresh)
    rows_t, cols_t = np.nonzero(value_t)
    if len(rows_t) == 0:
        return np.zeros((0, 2), dtype=float)
    vals_t = value[rows_t, cols_t]
    kps = np.column_stack([cols_t.astype(float),
                           rows_t.astype(float),
                           vals_t])          # [x, y, response]  0-indexed padded

    # ------------------------------------------------------------------ #
    # Remove boundary points (erode mask by disk)                          #
    # s = max(10, G_resize^(nOctaves-2))                                   #
    # ------------------------------------------------------------------ #
    s = max(10.0, g_resize ** (n_octaves - 2))
    kps = _remove_boundary_points(kps, msk, s)
    if len(kps) < 10:
        return np.zeros((0, 2), dtype=float)

    # Sort by response descending, keep top N
    kps = kps[np.argsort(-kps[:, 2])]
    kps = kps[:min(N, len(kps))]

    # Subtract padding to get 0-indexed original-image coordinates
    kps[:, :2] -= imagepat

    return kps[:, :2]


# =========================================================================== #
# Smoke test                                                                   #
# =========================================================================== #
if __name__ == '__main__':
    import os
    import sys

    from pyramid import build_homo_pyramid

    # ---- load a real LLVIP infrared image or fall back to synthetic ----
    llvip_path = os.path.join(
        os.path.dirname(__file__), '..', 'LLVIP', 'infrared', 'test', '190001.jpg'
    )
    if os.path.exists(llvip_path):
        bgr = cv2.imread(llvip_path)
        if bgr is not None:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(float) / 255.0
            print(f"Loaded LLVIP infrared: {llvip_path}  shape={gray.shape}")
        else:
            gray = None
    else:
        gray = None

    if gray is None:
        size = 512
        cx, cy = size // 2, size // 2
        gx = np.arange(size) - cx
        gy = np.arange(size) - cy
        xx, yy = np.meshgrid(gx, gy)
        # Textured synthetic: Gaussian blob + sinusoidal pattern
        gray = (np.exp(-(xx ** 2 + yy ** 2) / (2 * 100 ** 2))
                + 0.2 * np.sin(xx / 8) * np.cos(yy / 8))
        gray = (gray - gray.min()) / (gray.max() - gray.min())
        print(f"Using synthetic image  shape={gray.shape}")

    # ---- pyramid ----
    print("Building pyramid ...")
    dom_pyr, _ = build_homo_pyramid(   # note: returns (mom_pyr, dom_pyr)
        gray,
        n_octaves=3, n_layers=2, g_resize=1.2, g_sigma=1.6,
        patch_size=72, nba=12, int_flag=True, key_type='PC-ShiTomasi',
    )

    # ---- keypoints ----
    print("Detecting keypoints ...")
    kps = detect_homo_keypoint(
        gray, dom_pyr,
        scale=6, thresh=0.0, radius=1, N=5000,
        g_resize=1.2, key_type='PC-ShiTomasi',
    )

    print(f"Detected {len(kps)} keypoints")
    if len(kps) > 0:
        h, w = gray.shape
        xs, ys = kps[:, 0], kps[:, 1]
        print(f"  x range: [{xs.min():.1f}, {xs.max():.1f}]  (image width  {w})")
        print(f"  y range: [{ys.min():.1f}, {ys.max():.1f}]  (image height {h})")

        # Spatial distribution: split image into 3x3 grid, count per cell
        grid_r, grid_c = 3, 3
        print(f"  Keypoints per {grid_r}x{grid_c} grid cell:")
        for gi in range(grid_r):
            row_str = "  "
            for gj in range(grid_c):
                y_lo, y_hi = gi * h / grid_r, (gi + 1) * h / grid_r
                x_lo, x_hi = gj * w / grid_c, (gj + 1) * w / grid_c
                cnt = int(((xs >= x_lo) & (xs < x_hi) &
                           (ys >= y_lo) & (ys < y_hi)).sum())
                row_str += f"{cnt:5d}"
            print(row_str)

        # Sanity checks
        assert xs.min() >= 0 and xs.max() < w, "x out of image bounds"
        assert ys.min() >= 0 and ys.max() < h, "y out of image bounds"

        # Not all in one corner: at least 4 of 9 grid cells must be populated
        grid_counts = []
        for gi in range(grid_r):
            for gj in range(grid_c):
                y_lo, y_hi = gi * h / grid_r, (gi + 1) * h / grid_r
                x_lo, x_hi = gj * w / grid_c, (gj + 1) * w / grid_c
                cnt = int(((xs >= x_lo) & (xs < x_hi) &
                           (ys >= y_lo) & (ys < y_hi)).sum())
                grid_counts.append(cnt)
        n_populated = sum(c > 0 for c in grid_counts)
        assert n_populated >= 4, \
            f"Keypoints concentrated in too few cells ({n_populated}/9)"
        print(f"  Populated grid cells: {n_populated}/9  OK")

    print("Smoke test passed.")
    sys.exit(0)
