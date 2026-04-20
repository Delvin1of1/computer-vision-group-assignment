"""
Python port of LogGabor.m and lowpassfilter.m from HOMO_Feature_ImgMatching.

Original MATLAB by Gao Chenzhong (gao-pingqi@qq.com).
Frequency-grid construction and filter arithmetic mirror the MATLAB source
exactly, including even/odd dimension handling and ifftshift conventions.
"""

import numpy as np


def lowpass_filter(rows: int, cols: int, cutoff: float = 0.45, n: int = 15) -> np.ndarray:
    """
    Butterworth lowpass filter in the frequency domain (DC at corners).

    Matches lowpassfilter.m:
        f = ifftshift( 1 / (1 + (radius/cutoff)^(2n)) )
    """
    if cols % 2:
        xrange = np.arange(-(cols - 1) / 2, (cols - 1) / 2 + 1) / (cols - 1)
    else:
        xrange = np.arange(-cols // 2, cols // 2) / cols

    if rows % 2:
        yrange = np.arange(-(rows - 1) / 2, (rows - 1) / 2 + 1) / (rows - 1)
    else:
        yrange = np.arange(-rows // 2, rows // 2) / rows

    x, y = np.meshgrid(xrange, yrange)          # same as MATLAB meshgrid(xrange, yrange)
    radius = np.sqrt(x ** 2 + y ** 2)
    f = 1.0 / (1.0 + (radius / cutoff) ** (2 * n))
    return np.fft.ifftshift(f)


def log_gabor(
    img: np.ndarray,
    nscale: int = 4,
    norient: int = 6,
    min_wave_length: int = 3,
    mult: float = 1.6,
    sigma_onf: float = 0.75,
) -> list:
    """
    Log-Gabor filter bank.

    Matches LogGabor.m exactly (parameters from Major_Orientation_Map.m call:
    minWaveLength=3, mult=1.6, sigmaOnf=0.75).

    Parameters
    ----------
    img : 2-D (or 3-D, first channel used) real-valued array
    nscale : number of wavelet scales
    norient : number of filter orientations
    min_wave_length : wavelength of the finest scale filter (pixels)
    mult : scaling factor between successive scales
    sigma_onf : bandwidth parameter (std / centre-freq in log-freq domain)

    Returns
    -------
    EO : list of lists, EO[s][o] — complex ndarray, shape (rows, cols)
         real part  = even (symmetric) Log-Gabor response
         imag part  = odd  (antisymmetric) Log-Gabor response
         Indexing: s in 0..nscale-1, o in 0..norient-1
    """
    img = np.asarray(img, dtype=float)
    if img.ndim == 3:
        img = img[:, :, 0]
    rows, cols = img.shape

    image_fft = np.fft.fft2(img)

    # ------------------------------------------------------------------ #
    # Frequency grid — mirrors MATLAB meshgrid(xrange, yrange) exactly.   #
    # Note: theta = atan2(-y, x) matches MATLAB (image y-axis is flipped) #
    # ------------------------------------------------------------------ #
    if cols % 2:
        xrange = np.arange(-(cols - 1) / 2, (cols - 1) / 2 + 1) / (cols - 1)
    else:
        xrange = np.arange(-cols // 2, cols // 2) / cols

    if rows % 2:
        yrange = np.arange(-(rows - 1) / 2, (rows - 1) / 2 + 1) / (rows - 1)
    else:
        yrange = np.arange(-rows // 2, rows // 2) / rows

    x, y = np.meshgrid(xrange, yrange)
    radius = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(-y, x)

    radius = np.fft.ifftshift(radius)
    theta = np.fft.ifftshift(theta)
    radius[0, 0] = 1.0          # avoid log(0) at DC

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    lp = lowpass_filter(rows, cols, cutoff=0.45, n=15)

    # ------------------------------------------------------------------ #
    # Radial Log-Gabor envelopes                                           #
    # exp( -(log(r/fo))^2 / (2 * log(sigma_onf)^2) )                     #
    # ------------------------------------------------------------------ #
    lg_filters = []
    log_sigma_sq = 2.0 * np.log(sigma_onf) ** 2
    for s in range(nscale):
        wavelength = min_wave_length * mult ** s
        fo = 1.0 / wavelength
        lg = np.exp(-(np.log(radius / fo)) ** 2 / log_sigma_sq)
        lg *= lp
        lg[0, 0] = 0.0          # zero DC component
        lg_filters.append(lg)

    # ------------------------------------------------------------------ #
    # Angular spread × radial envelope → full 2-D filter → ifft2          #
    # spread = (cos(clamp(|dtheta| * norient/2, pi)) + 1) / 2             #
    # ------------------------------------------------------------------ #
    EO = [[None] * norient for _ in range(nscale)]
    for o in range(norient):
        angle = o * np.pi / norient
        ds = sin_theta * np.cos(angle) - cos_theta * np.sin(angle)
        dc = cos_theta * np.cos(angle) + sin_theta * np.sin(angle)
        d_theta = np.abs(np.arctan2(ds, dc))
        d_theta = np.minimum(d_theta * norient / 2.0, np.pi)
        spread = (np.cos(d_theta) + 1.0) / 2.0

        for s in range(nscale):
            filt = lg_filters[s] * spread
            EO[s][o] = np.fft.ifft2(image_fft * filt)

    return EO


# ======================================================================== #
# Smoke test — run as: python log_gabor.py                                  #
# Generates a 64×64 Gaussian blob, runs the filter bank, prints mean        #
# magnitudes for EO[0][0] and EO[3][5] for later comparison against MATLAB. #
# ======================================================================== #
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Deterministic 64x64 Gaussian blob centred at (32, 32)
    size = 64
    cx, cy = size // 2, size // 2
    gx = np.arange(size) - cx
    gy = np.arange(size) - cy
    xx, yy = np.meshgrid(gx, gy)
    sigma_blob = 10.0
    blob = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma_blob ** 2))

    print(f"Input image  : shape={blob.shape}, min={blob.min():.4f}, max={blob.max():.4f}")

    EO = log_gabor(blob, nscale=4, norient=6, min_wave_length=3, mult=1.6, sigma_onf=0.75)

    mag_00 = np.mean(np.abs(EO[0][0]))
    mag_35 = np.mean(np.abs(EO[3][5]))

    print(f"mean |EO[0][0]| = {mag_00:.8f}")
    print(f"mean |EO[3][5]| = {mag_35:.8f}")

    # Extra: sanity-check shapes and that responses are non-trivial
    assert EO[0][0].shape == blob.shape, "Shape mismatch"
    assert mag_00 > 0 and mag_35 > 0, "Zero response — something is wrong"
    print("Smoke test passed.")
