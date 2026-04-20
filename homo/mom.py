"""
Python port of Major_Orientation_Map.m from HOMO_Feature_ImgMatching.

Original MATLAB by Gao Chenzhong (gao-pingqi@qq.com).
Translates the MOM (Major Orientation Map) algorithm faithfully, including
odd/even Log-Gabor accumulation, ASLG window construction, and odd/even
coupling logic.
"""

import numpy as np
from scipy.ndimage import correlate

from log_gabor import log_gabor


# --------------------------------------------------------------------------- #
# Helpers matching MATLAB built-ins                                             #
# --------------------------------------------------------------------------- #

def _fspecial_gaussian(size: int, sigma: float) -> np.ndarray:
    """
    Matches MATLAB fspecial('gaussian', [size, size], sigma).
    Grid runs from -(size-1)/2 to +(size-1)/2, exponential then normalised.
    """
    half = (size - 1) / 2.0
    ax = np.arange(-half, half + 1)
    x, y = np.meshgrid(ax, ax)
    h = np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2))
    h /= h.sum()
    return h


def _imfilter_replicate(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Matches MATLAB imfilter(img, kernel, 'replicate').
    imfilter performs correlation; mode='nearest' replicates border pixels.
    For real inputs returns real; complex inputs are handled channel-wise.
    """
    if np.iscomplexobj(img):
        return (correlate(img.real, kernel, mode='nearest') +
                1j * correlate(img.imag, kernel, mode='nearest'))
    return correlate(img, kernel, mode='nearest')


# --------------------------------------------------------------------------- #
# Main function                                                                 #
# --------------------------------------------------------------------------- #

def major_orientation_map(
    I: np.ndarray,
    R1: float,
    R2: float,
    s: int,
    int_flag: bool,
):
    """
    MOM: Major Orientation Map.

    Matches Major_Orientation_Map.m exactly.

    Parameters
    ----------
    I        : 2-D grayscale image (float)
    R1       : inner radius for the ASLG Gaussian weighting
    R2       : outer radius; also sets the window half-width W = floor(R2)
    s        : number of Gaussian scale steps (1 → single kernel at R1/6)
    int_flag : if True, magnitude output is all-ones (uniform)

    Returns
    -------
    magnitude   : (M, N) float array
    orientation : (M, N) float array in [0, pi)
    """
    I = np.asarray(I, dtype=float)
    if I.ndim == 3:
        I = I[:, :, 0]

    # ------------------------------------------------------------------ #
    # Pre-smoothing: 5x5 Gaussian, sigma=0.5 (matches fspecial + imfilter)#
    # w = 2*round(3*0.5)+1 = 2*2+1 = 5                                    #
    # ------------------------------------------------------------------ #
    pre_sigma = 0.5
    pre_w = 2 * round(3 * pre_sigma) + 1   # = 5
    pre_kernel = _fspecial_gaussian(pre_w, pre_sigma)
    I = _imfilter_replicate(I, pre_kernel)

    # ------------------------------------------------------------------ #
    # Log-Gabor feature maps                                               #
    # ------------------------------------------------------------------ #
    Ns, No = 4, 6
    EO = log_gabor(I, nscale=Ns, norient=No, min_wave_length=3, mult=1.6, sigma_onf=0.75)
    M, N = I.shape

    # ------------------------------------------------------------------ #
    # Accumulate complex gradient fields Gx, Gy                           #
    # MATLAB angle(j) = pi*(j-1)/No  →  Python angle[o] = pi*o/No        #
    # Scale weight: MATLAB (Ns-i+1) for i=1..Ns  →  Python (Ns-s) for s=0..Ns-1
    # direct = (imag(EO) >= 0)*2 - 1  maps non-negative → +1, negative → -1
    # Gx/Gy are complex; odd part in imag, even (sign-aligned) in real     #
    # ------------------------------------------------------------------ #
    angles = np.pi * np.arange(No) / No
    angle_cos = np.cos(angles)
    angle_sin = np.sin(angles)

    Gx = np.zeros((M, N), dtype=complex)
    Gy = np.zeros((M, N), dtype=complex)

    for o in range(No):
        for sc in range(Ns):
            eo = EO[sc][o]
            direct = np.where(eo.imag >= 0, 1.0, -1.0)   # [-1, 1]
            weight = (Ns - sc)                             # Ns down to 1
            # term = imag(EO)*1i + real(EO).*direct  (complex)
            term = eo.imag * 1j + eo.real * direct
            Gx -= term * angle_cos[o] * weight
            Gy += term * angle_sin[o] * weight

    # ------------------------------------------------------------------ #
    # ASLG window: circular mask on (2W+1)x(2W+1) patch                   #
    # Wcircle: dx^2 + dy^2 < (W+1)^2                                      #
    # ------------------------------------------------------------------ #
    W = int(np.floor(R2))
    dx_vec = np.arange(-W, W + 1)
    dy_vec = np.arange(-W, W + 1)
    dx, dy = np.meshgrid(dx_vec, dy_vec)
    Wcircle = (dx ** 2 + dy ** 2) < (W + 1) ** 2
    patch_size = 2 * W + 1

    # ------------------------------------------------------------------ #
    # Build Gaussian weighting kernel h                                    #
    # s==1: single Gaussian at R1/6                                        #
    # s>1 : sum of s Gaussians linearly spaced from R1/6 to R2/6          #
    # ------------------------------------------------------------------ #
    if s == 1:
        h = _fspecial_gaussian(patch_size, R1 / 6.0)
    else:
        step = (R2 - R1) / (s - 1)
        h = np.zeros((patch_size, patch_size))
        for i in range(s):
            sigma_i = (R1 + step * i) / 6.0
            h += _fspecial_gaussian(patch_size, sigma_i)

    h = h * Wcircle   # apply circular mask

    # ------------------------------------------------------------------ #
    # Separate odd (imag) / even (real) channels                           #
    # ------------------------------------------------------------------ #
    Gx1 = Gx.imag    # odd-x
    Gx2 = Gx.real    # even-x
    Gy1 = Gy.imag    # odd-y
    Gy2 = Gy.real    # even-y

    # ASLG structure tensor components — odd channel
    Gxx1 = _imfilter_replicate(Gx1 * Gx1, h)
    Gyy1 = _imfilter_replicate(Gy1 * Gy1, h)
    Gxy1 = _imfilter_replicate(Gx1 * Gy1, h)
    Gsx1 = Gxx1 - Gyy1
    Gsy1 = 2.0 * Gxy1

    # ASLG structure tensor components — even channel
    Gxx2 = _imfilter_replicate(Gx2 * Gx2, h)
    Gyy2 = _imfilter_replicate(Gy2 * Gy2, h)
    Gxy2 = _imfilter_replicate(Gx2 * Gy2, h)
    Gsx2 = Gxx2 - Gyy2
    Gsy2 = 2.0 * Gxy2

    # ------------------------------------------------------------------ #
    # Dominant orientation and magnitude per channel                       #
    # atan2(Gsy, Gsx)/2 + pi/2 maps [-pi,pi] → [0,pi]                    #
    # magnitude = (Gsx^2+Gsy^2)^(1/4) = sqrt(sqrt(...))                  #
    # ------------------------------------------------------------------ #
    orientation1 = np.arctan2(Gsy1, Gsx1) / 2.0 + np.pi / 2.0
    magnitude1 = np.sqrt(np.sqrt(Gsx1 ** 2 + Gsy1 ** 2))

    orientation2 = np.arctan2(Gsy2, Gsx2) / 2.0 + np.pi / 2.0
    magnitude2 = np.sqrt(np.sqrt(Gsx2 ** 2 + Gsy2 ** 2))

    # ------------------------------------------------------------------ #
    # Odd/Even coupling: pick the stronger channel at each pixel           #
    # idx=1 → odd channel wins, idx=0 → even channel wins                 #
    # ceil((sign(m1-m2)+0.1)/2):  m1>m2 → 1,  m1<m2 → 0,  m1==m2 → 1   #
    # ------------------------------------------------------------------ #
    idx = np.ceil((np.sign(magnitude1 - magnitude2) + 0.1) / 2.0)

    orientation = idx * orientation1 + (1.0 - idx) * orientation2
    orientation = np.mod(orientation, np.pi)   # enforce [0, pi)

    if int_flag:
        magnitude = np.ones((M, N))
    else:
        magnitude = idx * magnitude1 + (1.0 - idx) * magnitude2

    return magnitude, orientation


# ======================================================================== #
# Smoke test                                                                 #
# ======================================================================== #
if __name__ == "__main__":
    import sys

    size = 128
    cx, cy = size // 2, size // 2
    gx = np.arange(size) - cx
    gy = np.arange(size) - cy
    xx, yy = np.meshgrid(gx, gy)
    img = np.exp(-(xx ** 2 + yy ** 2) / (2 * 20.0 ** 2))

    # Typical call parameters from the HOMO demo
    R1, R2 = 3.0, 12.0
    s = 3
    print(f"Input : shape={img.shape}, R1={R1}, R2={R2}, s={s}")

    # --- int_flag=1: magnitude must be all-ones ---
    mag1, ori1 = major_orientation_map(img, R1, R2, s, int_flag=True)
    assert mag1.shape == img.shape, f"magnitude shape mismatch: {mag1.shape}"
    assert ori1.shape == img.shape, f"orientation shape mismatch: {ori1.shape}"
    assert np.all(mag1 == 1.0), "int_flag=1 must yield magnitude=ones"
    assert np.all((ori1 >= 0) & (ori1 < np.pi)), \
        f"orientation out of [0, pi): min={ori1.min():.4f} max={ori1.max():.4f}"
    print(f"int_flag=1  -> magnitude all-ones OK  "
          f"orientation in [0,pi): [{ori1.min():.4f}, {ori1.max():.4f}] OK")

    # --- int_flag=0: magnitude values + orientation range check ---
    mag0, ori0 = major_orientation_map(img, R1, R2, s, int_flag=False)
    assert np.all((ori0 >= 0) & (ori0 < np.pi)), \
        f"orientation out of [0, pi): min={ori0.min():.4f} max={ori0.max():.4f}"
    print(f"int_flag=0  -> magnitude mean={mag0.mean():.6f} max={mag0.max():.6f}  "
          f"orientation in [0,pi): [{ori0.min():.4f}, {ori0.max():.4f}] OK")

    print("Smoke test passed.")
    sys.exit(0)
