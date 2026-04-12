"""
Python reimplementation of HOMO-Feature Image Matching Algorithm.

Original MATLAB code by Gao Chenzhong (gao-pingqi@qq.com)
Paper: "HOMO-Feature: Homogeneous-Feature-Based Cross-Modal Image Matching"

This port faithfully translates the MATLAB algorithm to Python/NumPy.
Coordinate convention: kps[:,0] = x (column), kps[:,1] = y (row).
"""

import numpy as np
import cv2
from scipy import ndimage
from scipy.spatial.distance import cdist
from skimage.morphology import disk as skdisk


# ─────────────────────────────────────────────────────────────────────────────
# LOW-LEVEL UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def fspecial_gaussian(shape, sigma):
    """MATLAB fspecial('gaussian', shape, sigma) – normalised Gaussian kernel."""
    m, n = [(s - 1) / 2.0 for s in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    h[h < np.finfo(float).eps * h.max()] = 0
    s = h.sum()
    return h / s if s != 0 else h


def imfilter_r(img, kernel):
    """MATLAB imfilter(img, kernel, 'replicate') — edge-replicate padding."""
    # ndimage.correlate matches MATLAB imfilter (correlation, not convolution)
    return ndimage.correlate(img.astype(np.float64),
                             kernel.astype(np.float64), mode='nearest')


def imresize_bicubic(img, scale_or_size):
    """MATLAB imresize(img, scale, 'bicubic')."""
    if isinstance(scale_or_size, (int, float)):
        new_h = max(1, int(round(img.shape[0] * scale_or_size)))
        new_w = max(1, int(round(img.shape[1] * scale_or_size)))
    else:
        new_h, new_w = int(scale_or_size[0]), int(scale_or_size[1])
    return cv2.resize(img.astype(np.float64), (new_w, new_h),
                      interpolation=cv2.INTER_CUBIC)


# ─────────────────────────────────────────────────────────────────────────────
# LOG-GABOR FILTER BANK
# ─────────────────────────────────────────────────────────────────────────────

def lowpass_filter(shape, cutoff, n):
    """Butterworth low-pass filter (MATLAB lowpassfilter.m)."""
    rows, cols = shape
    xrange = (np.arange(-cols // 2, cols - cols // 2) / cols
              if cols % 2 == 0
              else np.arange(-(cols - 1) // 2, (cols - 1) // 2 + 1) / (cols - 1))
    yrange = (np.arange(-rows // 2, rows - rows // 2) / rows
              if rows % 2 == 0
              else np.arange(-(rows - 1) // 2, (rows - 1) // 2 + 1) / (rows - 1))
    x, y = np.meshgrid(xrange, yrange)
    radius = np.sqrt(x ** 2 + y ** 2)
    lp = 1.0 / (1.0 + (radius / cutoff) ** (2 * n))
    return np.fft.ifftshift(lp)


def log_gabor(img, nscale, norient, min_wave_length=3, mult=1.6, sigma_f=0.75):
    """
    LogGabor filter bank (MATLAB LogGabor.m).
    Returns EO[scale][orient] – complex-valued convolution results.
    """
    if img.ndim == 3:
        img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2GRAY).astype(np.float64)
    rows, cols = img.shape
    image_fft = np.fft.fft2(img)

    if cols % 2:
        xrange = np.arange(-(cols - 1) // 2, (cols - 1) // 2 + 1) / (cols - 1)
    else:
        xrange = np.arange(-cols // 2, cols // 2) / cols
    if rows % 2:
        yrange = np.arange(-(rows - 1) // 2, (rows - 1) // 2 + 1) / (rows - 1)
    else:
        yrange = np.arange(-rows // 2, rows // 2) / rows

    x, y = np.meshgrid(xrange, yrange)
    radius = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(-y, x)

    radius = np.fft.ifftshift(radius)
    theta = np.fft.ifftshift(theta)
    radius[0, 0] = 1  # avoid log(0)

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    lp = lowpass_filter((rows, cols), 0.45, 15)

    lg_filters = []
    for s in range(nscale):
        wavelength = min_wave_length * mult ** s
        fo = 1.0 / wavelength
        lg = np.exp(-(np.log(radius / fo)) ** 2 / (2 * np.log(sigma_f) ** 2))
        lg = lg * lp
        lg[0, 0] = 0
        lg_filters.append(lg)

    EO = [[None] * norient for _ in range(nscale)]
    for o in range(norient):
        angle = o * np.pi / norient
        ds = sin_theta * np.cos(angle) - cos_theta * np.sin(angle)
        dc = cos_theta * np.cos(angle) + sin_theta * np.sin(angle)
        d_theta = np.abs(np.arctan2(ds, dc))
        d_theta = np.minimum(d_theta * norient / 2, np.pi)
        spread = (np.cos(d_theta) + 1) / 2
        for s in range(nscale):
            filt = lg_filters[s] * spread
            EO[s][o] = np.fft.ifft2(image_fft * filt)
    return EO


# ─────────────────────────────────────────────────────────────────────────────
# PHASE CONGRUENCY (phasecong3.m)
# ─────────────────────────────────────────────────────────────────────────────

def phase_congruency(img, nscale=4, norient=6,
                     min_wave_length=3, mult=1.6, sigma_f=0.75,
                     k=1.0, cut_off=0.5, g=3, noise_method=-1):
    """
    Phase congruency – returns (M, m) edge and corner maps.
    Translation of MATLAB phasecong3.m by Peter Kovesi.
    """
    if img.ndim == 3:
        img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2GRAY).astype(np.float64)
    rows, cols = img.shape
    epsilon = 1e-4
    image_fft = np.fft.fft2(img)

    if cols % 2:
        xrange = np.arange(-(cols - 1) // 2, (cols - 1) // 2 + 1) / (cols - 1)
    else:
        xrange = np.arange(-cols // 2, cols // 2) / cols
    if rows % 2:
        yrange = np.arange(-(rows - 1) // 2, (rows - 1) // 2 + 1) / (rows - 1)
    else:
        yrange = np.arange(-rows // 2, rows // 2) / rows

    x, y = np.meshgrid(xrange, yrange)
    radius = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(-y, x)
    radius = np.fft.ifftshift(radius)
    theta = np.fft.ifftshift(theta)
    radius[0, 0] = 1
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    lp = lowpass_filter((rows, cols), 0.45, 15)
    lg_filters = []
    for s in range(nscale):
        wavelength = min_wave_length * mult ** s
        fo = 1.0 / wavelength
        lg = np.exp(-(np.log(radius / fo)) ** 2 / (2 * np.log(sigma_f) ** 2))
        lg = lg * lp
        lg[0, 0] = 0
        lg_filters.append(lg)

    zero = np.zeros((rows, cols))
    covx2 = zero.copy(); covy2 = zero.copy(); covxy = zero.copy()
    EnergyV = np.zeros((rows, cols, 3))

    for o in range(norient):
        angl = o * np.pi / norient
        ds = sin_theta * np.cos(angl) - cos_theta * np.sin(angl)
        dc = cos_theta * np.cos(angl) + sin_theta * np.sin(angl)
        d_theta = np.abs(np.arctan2(ds, dc))
        d_theta = np.minimum(d_theta * norient / 2, np.pi)
        spread = (np.cos(d_theta) + 1) / 2

        sum_e = zero.copy()
        sum_o = zero.copy()
        sum_an = zero.copy()
        energy = zero.copy()
        tau = None
        max_an = None

        for s in range(nscale):
            filt = lg_filters[s] * spread
            eo = np.fft.ifft2(image_fft * filt)
            an = np.abs(eo)
            sum_an += an
            sum_e += np.real(eo)
            sum_o += np.imag(eo)
            if s == 0:
                if noise_method == -1:
                    tau = np.median(sum_an) / np.sqrt(np.log(4))
                elif noise_method == -2:
                    hist, edges = np.histogram(sum_an.ravel(), bins=50)
                    idx = np.argmax(hist)
                    tau = (edges[idx] + edges[idx + 1]) / 2
                max_an = an.copy()
            else:
                max_an = np.maximum(max_an, an)

        x_energy = np.sqrt(sum_e ** 2 + sum_o ** 2) + epsilon
        mean_e = sum_e / x_energy
        mean_o = sum_o / x_energy

        for s in range(nscale):
            filt = lg_filters[s] * spread
            eo = np.fft.ifft2(image_fft * filt)
            E = np.real(eo); O = np.imag(eo)
            energy += E * mean_e + O * mean_o - np.abs(E * mean_o - O * mean_e)

        if noise_method >= 0:
            T = noise_method
        else:
            total_tau = tau * (1 - (1 / mult) ** nscale) / (1 - (1 / mult))
            est_mean = total_tau * np.sqrt(np.pi / 2)
            est_sigma = total_tau * np.sqrt((4 - np.pi) / 2)
            T = est_mean + k * est_sigma

        energy = np.maximum(energy - T, 0)
        width = (sum_an / (max_an + epsilon) - 1) / (nscale - 1)
        weight = 1.0 / (1 + np.exp((cut_off - width) * g))
        pc = weight * energy / (sum_an + epsilon)

        covx = pc * np.cos(angl)
        covy = pc * np.sin(angl)
        covx2 += covx ** 2
        covy2 += covy ** 2
        covxy += covx * covy

        EnergyV[:, :, 0] += sum_e
        EnergyV[:, :, 1] += np.cos(angl) * sum_o
        EnergyV[:, :, 2] += np.sin(angl) * sum_o

    covx2 /= (norient / 2)
    covy2 /= (norient / 2)
    covxy = 4 * covxy / norient
    denom = np.sqrt(covxy ** 2 + (covx2 - covy2) ** 2) + epsilon
    M = (covy2 + covx2 + denom) / 2   # edge strength
    m = (covy2 + covx2 - denom) / 2   # corner strength
    return M, m


# ─────────────────────────────────────────────────────────────────────────────
# MAJOR ORIENTATION MAP (MOM)
# ─────────────────────────────────────────────────────────────────────────────

def major_orientation_map(I, R1, R2, s, int_flag):
    """
    MOM: Major Orientation Map.  Translation of Major_Orientation_Map.m.
    Returns (magnitude, orientation), orientation in [0, pi).
    """
    sigma = 0.5
    w_size = 2 * round(3 * sigma) + 1
    w_g = fspecial_gaussian((w_size, w_size), sigma)
    I = imfilter_r(I, w_g)

    Ns, No = 4, 6
    EO = log_gabor(I, Ns, No, 3, 1.6, 0.75)
    M_img, N_img = I.shape

    Gx = np.zeros((M_img, N_img), dtype=complex)
    Gy = np.zeros((M_img, N_img), dtype=complex)
    angles = np.pi * np.arange(No) / No

    for j in range(No):
        for i in range(Ns):
            eo = EO[i][j]
            direct = (np.imag(eo) >= 0).astype(np.float64) * 2 - 1
            Gx -= (np.imag(eo) * 1j + np.real(eo) * direct) * np.cos(angles[j]) * (Ns - i)
            Gy += (np.imag(eo) * 1j + np.real(eo) * direct) * np.sin(angles[j]) * (Ns - i)

    W = int(np.floor(R2))
    dx = np.arange(-W, W + 1)
    dy = np.arange(-W, W + 1)
    DX, DY = np.meshgrid(dx, dy)
    Wcircle = (DX ** 2 + DY ** 2) < (W + 1) ** 2
    psize = 2 * W + 1

    if s == 1:
        h = fspecial_gaussian((psize, psize), R1 / 6)
    else:
        step = (R2 - R1) / (s - 1)
        h = np.zeros((psize, psize))
        for i in range(s):
            sig_i = (R1 + step * i) / 6
            h += fspecial_gaussian((psize, psize), max(sig_i, 0.1))
    h = h * Wcircle

    Gx1 = np.imag(Gx); Gx2 = np.real(Gx)
    Gy1 = np.imag(Gy); Gy2 = np.real(Gy)

    Gxx1 = imfilter_r(Gx1 * Gx1, h)
    Gyy1 = imfilter_r(Gy1 * Gy1, h)
    Gxy1 = imfilter_r(Gx1 * Gy1, h)
    Gsx1 = Gxx1 - Gyy1
    Gsy1 = 2 * Gxy1

    Gxx2 = imfilter_r(Gx2 * Gx2, h)
    Gyy2 = imfilter_r(Gy2 * Gy2, h)
    Gxy2 = imfilter_r(Gx2 * Gy2, h)
    Gsx2 = Gxx2 - Gyy2
    Gsy2 = 2 * Gxy2

    orientation1 = np.arctan2(Gsy1, Gsx1) / 2 + np.pi / 2
    magnitude1 = np.sqrt(np.sqrt(Gsx1 ** 2 + Gsy1 ** 2))
    orientation2 = np.arctan2(Gsy2, Gsx2) / 2 + np.pi / 2
    magnitude2 = np.sqrt(np.sqrt(Gsx2 ** 2 + Gsy2 ** 2))

    idx = np.ceil((np.sign(magnitude1 - magnitude2) + 0.1) / 2)
    orientation = idx * orientation1 + (1 - idx) * orientation2
    orientation = np.mod(orientation, np.pi)

    if int_flag:
        magnitude = np.ones((M_img, N_img))
    else:
        magnitude = idx * magnitude1 + (1 - idx) * magnitude2

    return magnitude, orientation


# ─────────────────────────────────────────────────────────────────────────────
# GAUSSIAN PYRAMID HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_gaussian_scale(sigma, num_layers):
    """MATLAB Get_Gaussian_Scale – cumulative sigma sequence."""
    sig = np.zeros(num_layers)
    sig[0] = sigma
    if num_layers < 2:
        return sig
    k = 2 ** (1.0 / (num_layers - 1))
    for i in range(1, num_layers):
        sig_prev = k ** (i - 1) * sigma
        sig_curr = k * sig_prev
        sig[i] = np.sqrt(sig_curr ** 2 - sig_prev ** 2)
    return sig


def gaussian_scaling(I, layer, sig):
    """Apply Gaussian blur for pyramid layer > 0 (0-indexed)."""
    if layer == 0:
        return I
    window_size = round(3 * sig)
    window_size = 2 * window_size + 1
    w = fspecial_gaussian((window_size, window_size), sig)
    return imfilter_r(I, w)


# ─────────────────────────────────────────────────────────────────────────────
# BUILD HOMO PYRAMID
# ─────────────────────────────────────────────────────────────────────────────

def build_homo_pyramid(I, nOctaves, nLayers, G_resize, G_sigma,
                       patch_size, NBA, int_flag, key_type):
    """
    Build MOM and DoM pyramids.  Translation of Build_Homo_Pyramid.m.
    Returns (MOM_pyr, DoM_pyr) as 2-D lists [octave][layer].
    """
    do_dom = 'free' not in key_type.lower()
    sig = get_gaussian_scale(G_sigma, nLayers)
    W = int(np.floor(patch_size / 2))
    r = np.sqrt(W ** 2 / (2 * NBA + 1))

    MOM_pyr = [[None] * nLayers for _ in range(nOctaves)]
    DoM_pyr = [[None] * nLayers for _ in range(nOctaves)]

    for octave in range(nOctaves):
        scale = 1.0 / G_resize ** octave
        I_t = imresize_bicubic(I, scale)
        for layer in range(nLayers):
            I_t = gaussian_scaling(I_t, layer, sig[layer])
            _, MOM_pyr[octave][layer] = major_orientation_map(I_t, 1, r, 4, int_flag)

        if do_dom and nLayers > 1:
            # DoM between layer 0 and 1  (nLayers=2 case)
            last = nLayers - 1
            temp = np.abs(MOM_pyr[octave][last - 1] - MOM_pyr[octave][last])
            temp = np.pi / 2 - np.abs(temp - np.pi / 2)
            DoM_pyr[octave][last - 1] = temp

    return MOM_pyr, DoM_pyr


# ─────────────────────────────────────────────────────────────────────────────
# KEYPOINT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def shi_tomasi(img, scale):
    """
    Vectorised Shi-Tomasi corner response.  Translation of ShiTomasi.m.
    Uses minimum eigenvalue of the structure tensor.
    """
    hx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    hy = hx.T
    Gx = imfilter_r(img, hx)
    Gy = imfilter_r(img, hy)

    W_r = int(np.floor(scale / 2))
    dx = np.arange(-W_r, W_r + 1)
    DX, DY = np.meshgrid(dx, dx)
    Wcircle = (DX ** 2 + DY ** 2) < (W_r + 1) ** 2
    h = fspecial_gaussian((scale + 1, scale + 1), scale / 6) * Wcircle

    Gxx = imfilter_r(Gx * Gx, h)
    Gyy = imfilter_r(Gy * Gy, h)
    Gxy = imfilter_r(Gx * Gy, h)

    # Minimum eigenvalue of [[Gxx, Gxy],[Gxy, Gyy]] — vectorised
    trace = Gxx + Gyy
    det_diff = np.sqrt(np.maximum((Gxx - Gyy) ** 2 + 4 * Gxy ** 2, 0))
    value = (trace - det_diff) / 2
    return np.maximum(value, 0)


def detect_homo_keypoints(I, DoM_pyr, scale, thresh, radius, N,
                          G_resize, key_type):
    """
    Detect HOMO keypoints.  Translation of Detect_Homo_Keypoint.m.
    Returns kps array of shape (K, 2) with (x, y) coordinates.
    """
    rows, cols = I.shape

    # ── Detector-free grid ──────────────────────────────────────────────────
    if 'free' in key_type.lower():
        step = max(np.sqrt(rows * cols / N), radius)
        Nn = max(1, round(cols / step))
        Nm = max(1, round(rows / step))
        ind_x = np.round(cols / Nn * (np.arange(Nn) + 0.5)).astype(int)
        ind_y = np.round(rows / Nm * (np.arange(Nm) + 0.5)).astype(int)
        XX, YY = np.meshgrid(ind_x, ind_y)
        kps_xy = np.column_stack([XX.ravel(), YY.ravel()])
        mask = I[kps_xy[:, 1].clip(0, rows - 1),
                  kps_xy[:, 0].clip(0, cols - 1)] > 0
        kps_xy = kps_xy[mask]
        return kps_xy

    # ── HOMO feature weighting (homoness map) ───────────────────────────────
    nOctaves = len(DoM_pyr)
    nLayers = len(DoM_pyr[0])
    homoness = np.ones((rows, cols))
    for octave in range(nOctaves):
        for layer in range(nLayers):
            if DoM_pyr[octave][layer] is not None:
                resized = imresize_bicubic(DoM_pyr[octave][layer], (rows, cols))
                homoness *= resized
    exp = 6.0 / (nOctaves * max(nLayers - 1, 1))
    with np.errstate(divide='ignore', invalid='ignore'):
        homoness = 1.0 / (homoness ** exp + 1e-10)

    # ── Phase Congruency weighting ──────────────────────────────────────────
    if 'pc' in key_type.lower():
        pc_M, pc_m = phase_congruency(I, 4, 6, 3, mult=1.6, sigma_f=0.75,
                                      k=1.0, g=3, noise_method=-1)
        mask_img = np.ceil(I / (I.max() + 1e-10))
        I_det = (pc_M + pc_m) * mask_img
    else:
        I_det = I.copy()

    I_max = I_det.max()
    if I_max > 0:
        I_det = I_det / I_max * 255

    # ── Image pad (border protection) ───────────────────────────────────────
    pad = 5
    rows_p, cols_p = rows + 2 * pad, cols + 2 * pad
    I_p = np.zeros((rows_p, cols_p))
    I_p[pad:pad + rows, pad:pad + cols] = I_det

    hom_p = np.zeros((rows_p, cols_p))
    hom_p[pad:pad + rows, pad:pad + cols] = homoness

    # ── Corner detector ─────────────────────────────────────────────────────
    if 'harris' in key_type.lower():
        val = _harris(I_p, scale)
    else:  # ShiTomasi (default)
        val = shi_tomasi(I_p, scale)

    border = pad + max(scale, radius) * 2 + 1
    b = int(border)
    val[:b + 1, :] = 0; val[-(b + 1):, :] = 0
    val[:, :b + 1] = 0; val[:, -(b + 1):] = 0

    val = val * hom_p

    # ── Non-maximal suppression ──────────────────────────────────────────────
    sze = 2 * radius + 1
    mx = ndimage.maximum_filter(val, size=int(sze))
    value_t = (val == mx) & (val > thresh)
    ys, xs = np.where(value_t)
    values = val[ys, xs]

    kps = np.column_stack([xs, ys, values]).astype(np.float64)

    # ── Remove boundary points (erode mask) ─────────────────────────────────
    mask_img_p = np.zeros((rows_p, cols_p), dtype=bool)
    mask_img_p[pad:pad + rows, pad:pad + cols] = True
    se_size = max(10, int(G_resize ** (nOctaves - 2)))
    try:
        se = skdisk(se_size)
        mask_eroded = ndimage.binary_erosion(mask_img_p, structure=se)
    except Exception:
        mask_eroded = mask_img_p

    keep = mask_eroded[kps[:, 1].astype(int).clip(0, rows_p - 1),
                        kps[:, 0].astype(int).clip(0, cols_p - 1)]
    kps = kps[keep]

    if len(kps) < 10:
        return np.empty((0, 2))

    # ── Sort, limit, strip pad ───────────────────────────────────────────────
    order = np.argsort(-kps[:, 2])
    kps = kps[order[:min(N, len(kps))]]
    kps = kps[:, :2] - pad  # remove pad offset
    return kps


def _harris(img, scale):
    """Harris corner response (simplified, same structure as ShiTomasi)."""
    hx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    hy = hx.T
    Gx = imfilter_r(img, hx)
    Gy = imfilter_r(img, hy)
    W_r = int(np.floor(scale / 2))
    dx = np.arange(-W_r, W_r + 1)
    DX, DY = np.meshgrid(dx, dx)
    Wcircle = (DX ** 2 + DY ** 2) < (W_r + 1) ** 2
    h = fspecial_gaussian((scale + 1, scale + 1), scale / 6) * Wcircle
    Gxx = imfilter_r(Gx * Gx, h)
    Gyy = imfilter_r(Gy * Gy, h)
    Gxy = imfilter_r(Gx * Gy, h)
    return (Gxx * Gyy - Gxy ** 2) / (Gxx + Gyy + 1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# BASE DIRECTION (orientation assignment)
# ─────────────────────────────────────────────────────────────────────────────

def base_direction(kps, mag, angle_map, W, NBO):
    """
    Assign dominant orientation(s) to each keypoint.
    Translation of Base_Direction.m.
    Returns kps_new with extra column for orientation.
    Vectorised: np.bincount replaces the inner pixel loop.
    """
    W = round(W)
    patch_size = W * 2 + 1           # = 73 for W=36
    X = np.arange(-W, W + 1)
    XX, YY = np.meshgrid(X, X)       # both (patch_size × patch_size)
    Wcircle = (XX ** 2 + YY ** 2) < (W + 1) ** 2

    rows_img, cols_img = angle_map.shape
    # Quantise full orientation map to bins [1, NBO]
    angle_q_full = (np.floor(angle_map * NBO / np.pi) + 1).astype(int)
    angle_q_full = np.clip(angle_q_full, 1, NBO)

    o_thr = 0.8
    kps_out = []

    for k in range(len(kps)):
        x, y = int(round(kps[k, 0])), int(round(kps[k, 1]))
        x1, x2 = max(0, x - W), min(x + W + 1, cols_img)
        y1, y2 = max(0, y - W), min(y + W + 1, rows_img)
        wx1 = W + x1 - x; wx2 = W + x2 - x
        wy1 = W + y1 - y; wy2 = W + y2 - y

        # Extract angle-bin patch (zero outside circle)
        angle_bin = np.zeros((patch_size, patch_size), dtype=int)
        angle_bin[wy1:wy2, wx1:wx2] = angle_q_full[y1:y2, x1:x2]
        angle_bin = angle_bin * Wcircle

        # Extract magnitude patch
        weight = np.zeros((patch_size, patch_size))
        weight[wy1:wy2, wx1:wx2] = mag[y1:y2, x1:x2]

        # Vectorised histogram (replaces double for-loop)
        valid = angle_bin > 0
        if not valid.any():
            continue
        o_hist = np.bincount(angle_bin[valid] - 1,
                             weights=weight[valid],
                             minlength=NBO).astype(np.float64)

        max_thr = o_thr * o_hist.max()
        if max_thr <= 0:
            continue

        for o in range(NBO):
            aa = o_hist[(o - 1) % NBO]
            oo = o_hist[o]
            bb = o_hist[(o + 1) % NBO]
            if oo > aa and oo > bb and oo > max_thr:
                o_bin = o + 0.5 * (aa - bb) / (aa + bb - 2 * oo + 1e-10)
                o_bin = o_bin % NBO
                orient = (o_bin + 1) * (np.pi / NBO)
                kps_out.append(list(kps[k]) + [orient])

    if len(kps_out) == 0:
        return np.empty((0, kps.shape[1] + 1))
    return np.array(kps_out)


# ─────────────────────────────────────────────────────────────────────────────
# GPOLAR DESCRIPTOR
# ─────────────────────────────────────────────────────────────────────────────

def gpolar_descriptor(MOM_m, MOM_o, kps, patch_size, NBA, NBO, rot_flag):
    """
    GPolar descriptor computation.  Translation of GPolar_Descriptor.m.
    Input kps: (K, 2+) array with columns [x, y, ...original coords..., idx].
    Returns (K, 6 + desc_len) array: [x, y, xt, yt, orient, idx, ...desc...].
    """
    if len(kps) == 0:
        return np.empty((0, 6 + (2 + 3 * NBA) * NBO))

    if MOM_m is None:
        MOM_m = np.ones_like(MOM_o)

    W = int(patch_size // 2)
    X = np.arange(-W, W + 1)
    XX, YY = np.meshgrid(X, X)
    Wcircle = (XX ** 2 + YY ** 2) < (W + 1) ** 2

    rr1 = W ** 2 / (2 * NBA + 1)
    rr2 = rr1 * (NBA + 1)
    Rho_sq = XX ** 2 + YY ** 2
    Rho = np.zeros_like(Rho_sq, dtype=int)
    Rho[Rho_sq <= rr1] = 1
    Rho[(Rho_sq > rr1) & (Rho_sq <= rr2)] = 2
    Rho[Rho_sq > rr2] = 3
    Rho = (Rho * Wcircle) - 1  # -1: outside circle, 0: centre, 1/2: rings

    Theta0 = np.arctan2(YY, XX) + np.pi  # [0, 2*pi]
    if not rot_flag:
        Theta_base = (np.floor(Theta0 * NBA / (np.pi * 2)) % NBA).astype(int) + 1

    w_ratio1 = np.sqrt(rr1 / rr2)
    w_ratio2 = np.sqrt(rr1) / W
    c_idx = list(range(1, NBA)) + [0]  # circular shift left

    if rot_flag:
        kps = base_direction(kps, MOM_m, MOM_o, W, NBO)
    else:
        kps = np.hstack([kps, np.zeros((len(kps), 1))])

    if len(kps) == 0:
        return np.empty((0, 6 + (2 + 3 * NBA) * NBO))

    rows_img, cols_img = MOM_o.shape
    desc_len = (2 + 3 * NBA) * NBO
    descriptors = np.zeros((len(kps), desc_len))

    for k in range(len(kps)):
        x, y = int(round(kps[k, 0])), int(round(kps[k, 1]))
        x1, x2 = max(0, x - W), min(x + W + 1, cols_img)
        y1, y2 = max(0, y - W), min(y + W + 1, rows_img)
        wx1 = W + x1 - x; wx2 = W + x2 - x
        wy1 = W + y1 - y; wy2 = W + y2 - y

        weight = np.zeros((patch_size + 1, patch_size + 1))
        weight[wy1:wy2, wx1:wx2] = MOM_m[y1:y2, x1:x2]

        orient = kps[k, -1]

        if rot_flag:
            orient_patch = MOM_o[y1:y2, x1:x2]
            rel = orient_patch - orient
            bins = (np.floor(rel * NBO / np.pi) % NBO).astype(int) + 1
            angle_bin = np.zeros((patch_size + 1, patch_size + 1), dtype=int)
            angle_bin[wy1:wy2, wx1:wx2] = bins
            Theta = (np.floor((Theta0 - orient) * NBA / (np.pi * 2)) % NBA).astype(int) + 1
        else:
            orient_patch = MOM_o[y1:y2, x1:x2]
            bins = np.clip(
                (np.floor(orient_patch * NBO / np.pi)).astype(int) + 1, 1, NBO)
            angle_bin = np.zeros((patch_size + 1, patch_size + 1), dtype=int)
            angle_bin[wy1:wy2, wx1:wx2] = bins
            Theta = Theta_base

        feat_center = np.zeros(NBO)
        feat_outer = np.zeros((NBO, NBA, 3))

        valid = (angle_bin >= 1) & (Rho >= 0) & (Rho <= 2)

        # Centre region
        ctr_mask = valid & (Rho == 0)
        if ctr_mask.any():
            ab = angle_bin[ctr_mask] - 1
            np.add.at(feat_center, ab, weight[ctr_mask])

        # Ring 1 and 2
        for rho_val in (1, 2):
            ring = valid & (Rho == rho_val)
            if ring.any():
                ab = angle_bin[ring] - 1          # [0, NBO-1]
                th = Theta[ring] - 1               # [0, NBA-1]
                w_ = weight[ring]
                # Use tuple indexing so np.add.at writes into the view, not a copy
                np.add.at(feat_outer[:, :, rho_val - 1], (ab, th), w_)

        # Inversion invariance
        if rot_flag and NBA >= 2:
            half = NBA // 2
            H1 = feat_outer[:, :half, :2].copy()
            H2 = feat_outer[:, half:, :2].copy()
            if np.var(H1) > np.var(H2):
                feat_outer[:, :, :2] = np.concatenate([H2, H1], axis=1)

        # Deeper feature (3rd ring = cross-ring combination)
        deep = ((feat_outer[:, :, 0] + feat_outer[:, c_idx, 0]) * w_ratio1 +
                (feat_outer[:, :, 1] + feat_outer[:, c_idx, 1]) * w_ratio2) / 2 \
               + np.tile(feat_center[:, np.newaxis], (1, NBA))
        feat_outer[:, :, 2] = deep / 3 / 2

        feat_all = feat_outer[:, :, 2].sum(axis=1) / NBA

        # Concatenate: centre + outer (flattened Fortran order to match MATLAB) + feat_all
        descriptors[k] = np.concatenate([
            feat_center,
            feat_outer.ravel(order='F'),   # column-major to match MATLAB(:)
            feat_all
        ])

    return np.hstack([kps, descriptors])


# ─────────────────────────────────────────────────────────────────────────────
# FSC (Fast Sample Consensus / RANSAC)
# ─────────────────────────────────────────────────────────────────────────────

def _lsm_affine(cor1, cor2):
    """Affine transform via least squares. Returns 3×3 homogeneous matrix."""
    n = len(cor1)
    A = np.zeros((2 * n, 6))
    b = np.zeros(2 * n)
    for i in range(n):
        x, y = cor1[i, 0], cor1[i, 1]
        u, v = cor2[i, 0], cor2[i, 1]
        A[2 * i]     = [x, y, 0, 0, 1, 0]
        A[2 * i + 1] = [0, 0, x, y, 0, 1]
        b[2 * i]     = u
        b[2 * i + 1] = v
    H, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    M = np.array([[H[0], H[1], H[4]],
                  [H[2], H[3], H[5]],
                  [0,    0,    1   ]])
    return M


def _lsm_projective(cor1, cor2):
    """Projective (homography) transform via least squares."""
    n = len(cor1)
    A = np.zeros((2 * n, 8))
    b = np.zeros(2 * n)
    for i in range(n):
        x, y = cor1[i, 0], cor1[i, 1]
        u, v = cor2[i, 0], cor2[i, 1]
        A[2 * i]     = [x, y, 0, 0, 1, 0, -u * x, -u * y]
        A[2 * i + 1] = [0, 0, x, y, 0, 1, -v * x, -v * y]
        b[2 * i]     = u
        b[2 * i + 1] = v
    H, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    M = np.array([[H[0], H[1], H[4]],
                  [H[2], H[3], H[5]],
                  [H[6], H[7], 1   ]])
    return M


def fsc(cor1, cor2, t_form='affine', error=5, max_iter=800, ref=None):
    """
    Fast Sample Consensus.  Translation of FSC.m.
    Returns (H, inlier_mask).
    """
    if len(cor1) < 20:
        return None, np.zeros(len(cor1), dtype=bool)

    t_form = t_form.lower()
    if any(k in t_form for k in ('project', 'perspect', 'homo')):
        t_form = 'projective'

    n_sample = {'similarity': 2, 'affine': 3, 'projective': 4}.get(t_form, 3)
    proj_flag = (t_form == 'projective')

    cor1_orig = cor1.copy()
    cor2_orig = cor2.copy()

    if ref is not None and ref[0] is not None and len(ref[0]) > 0:
        cor1 = np.vstack([cor1, ref[0][:, :2]])
        cor2 = np.vstack([cor2, ref[1][:, :2]])

    M = len(cor1)
    M0 = len(cor1_orig)
    import math as _math
    total_iter = min(max_iter,
                     int(_math.comb(M, n_sample)) if M >= n_sample else 0)
    if total_iter == 0:
        return None, np.zeros(M0, dtype=bool)

    err2 = error ** 2
    best_n = 0
    best_inlier = np.zeros(M, dtype=bool)
    best_inlier[:M0] = True

    c1_xy = cor1[:, :2]
    c2_xy = cor2[:, :2]
    c1_h = np.hstack([c1_xy, np.ones((M, 1))]).T  # 3×M

    rng = np.random.default_rng(42)
    for _ in range(total_iter):
        idx = rng.choice(M, n_sample, replace=False)
        try:
            if t_form == 'affine':
                H = _lsm_affine(c1_xy[idx], c2_xy[idx])
            elif t_form == 'projective':
                H = _lsm_projective(c1_xy[idx], c2_xy[idx])
            else:  # similarity — approximate as affine with 2 pts
                H = _lsm_affine(c1_xy[idx[:2]], c2_xy[idx[:2]])
        except Exception:
            continue

        proj = H @ c1_h  # 3×M
        if proj_flag:
            proj = proj[:2] / (proj[2:3] + 1e-10)
        else:
            proj = proj[:2]
        diff = proj - c2_xy.T
        sq_dist = (diff ** 2).sum(axis=0)
        inlier = sq_dist < err2
        n_in = inlier.sum()
        if n_in > best_n:
            best_n = n_in
            best_inlier = inlier

    best_inlier[M0:] = False  # never mark reference pts as inliers

    if best_inlier[:M0].sum() < 4:
        return None, np.zeros(M0, dtype=bool)

    # Final fit on inliers
    in_c1 = c1_xy[best_inlier]
    in_c2 = c2_xy[best_inlier]
    try:
        if t_form == 'affine':
            H_final = _lsm_affine(in_c1, in_c2)
        elif t_form == 'projective':
            H_final = _lsm_projective(in_c1, in_c2)
        else:
            H_final = _lsm_affine(in_c1, in_c2)
    except Exception:
        H_final = None

    return H_final, best_inlier[:M0]


def outlier_removal(cor1, cor2, error, n_iter, trans_form, ref=None):
    """
    Translation of Outlier_Removal.m.
    Returns (cor1_in, cor2_in, inlier_mask).
    inlier_mask covers all M0 original points (ref points never included here).
    """
    M0 = len(cor1)
    if M0 < 20:
        return np.empty((0, cor1.shape[1])), np.empty((0, cor2.shape[1])), \
               np.zeros(M0, dtype=bool)

    if ref is None:
        ref = [None, None]

    H, _ = fsc(cor1[:, :2], cor2[:, :2], trans_form, error, n_iter, ref)
    if H is None:
        return np.empty((0, cor1.shape[1])), np.empty((0, cor2.shape[1])), \
               np.zeros(M0, dtype=bool)

    # Second pass: project all M0 points and measure reprojection error
    c1_h = np.hstack([cor1[:, :2], np.ones((M0, 1))]).T
    proj = H @ c1_h
    proj = proj[:2] / (proj[2:3] + 1e-10)
    err_v = np.sqrt(((proj - cor2[:, :2].T) ** 2).sum(axis=0))
    final_inlier = err_v < error

    if final_inlier.sum() < 4:
        return np.empty((0, cor1.shape[1])), np.empty((0, cor2.shape[1])), \
               np.zeros(M0, dtype=bool)

    return cor1[final_inlier], cor2[final_inlier], final_inlier


# ─────────────────────────────────────────────────────────────────────────────
# KEYPOINT MATCHING
# ─────────────────────────────────────────────────────────────────────────────

def match_keypoints(desc1, desc2, error, K, trans_form, ref=None):
    """
    Match two sets of HOMO descriptors.  Translation of Match_Keypoint.m.
    Returns (matches, num_matches) where matches is (N, 12) array.
    """
    if len(desc1) == 0 or len(desc2) == 0:
        return np.empty((0, 12)), 0

    kps1 = desc1[:, :6]; des1 = desc1[:, 6:]
    kps2 = desc2[:, :6]; des2 = desc2[:, 6:]

    # L2-normalise descriptors — makes histogram magnitudes scale-invariant,
    # which helps cross-modal pairs with different intensity distributions.
    des1 = des1 / (np.linalg.norm(des1, axis=1, keepdims=True) + 1e-10)
    des2 = des2 / (np.linalg.norm(des2, axis=1, keepdims=True) + 1e-10)

    # Nearest-neighbour matching — no ratio test (matches MATLAB matchFeatures
    # called with MaxRatio=1, MatchThreshold=100, which accepts all matches).
    D = cdist(des1.astype(np.float32), des2.astype(np.float32), metric='euclidean')
    nn_idx = np.argmin(D, axis=1)

    # Unique matches: keep only the closest match per desc2 index
    # (equivalent to MATLAB's [~,uniqueIdx] = unique(indexPairs(:,2)))
    seen = {}
    for i, j in enumerate(nn_idx):
        d = D[i, j]
        if j not in seen or d < seen[j][0]:
            seen[j] = (d, i)
    pairs = [(v[1], k) for k, v in seen.items()]

    if len(pairs) < 20:
        return np.empty((0, 12)), 0

    idx1 = np.array([p[0] for p in pairs])
    idx2 = np.array([p[1] for p in pairs])
    cor1 = kps1[idx1]
    cor2 = kps2[idx2]

    if ref is None:
        ref = [None, None]

    # Outlier removal (K repetitions)
    best_ncm = 0
    best_inlier = None
    for _ in range(K):
        _, _, inlier = outlier_removal(cor1, cor2, error, 800, trans_form, ref)
        ncm = int(inlier.sum())
        if ncm > best_ncm:
            best_ncm = ncm
            best_inlier = inlier

    if best_ncm < 4:
        return np.empty((0, 12)), 0

    matched_cor1 = cor1[best_inlier]
    matched_cor2 = cor2[best_inlier]

    # Include reference anchor points in final output (MATLAB Match_Keypoint behaviour)
    if ref[0] is not None and len(ref[0]) > 0:
        matched_cor1 = np.vstack([matched_cor1, ref[0]])
        matched_cor2 = np.vstack([matched_cor2, ref[1]])

    matches = np.hstack([matched_cor1, matched_cor2])
    return matches, len(matched_cor1)


# ─────────────────────────────────────────────────────────────────────────────
# MULTISCALE STRATEGY
# ─────────────────────────────────────────────────────────────────────────────

def multiscale_strategy(kps_1, kps_2, MOM_pyr1, MOM_pyr2,
                        patch_size, NBA, NBO, G_resize,
                        error, K, trans_form, rot_flag, scl_flag=False):
    """
    Multiscale descriptor extraction and matching.
    Translation of Multiscale_Strategy.m (non-parallel path).
    Returns (cor1, cor2) matched coordinates, each (N, 6) array.
    """
    # Augment keypoints with original-coordinate tracking
    kps_1 = np.hstack([kps_1[:, :2], np.arange(len(kps_1))[:, None]])  # [x,y,idx]
    kps_2 = np.hstack([kps_2[:, :2], np.arange(len(kps_2))[:, None]])

    nOctaves = len(MOM_pyr1)
    nLayers = len(MOM_pyr1[0])

    # Scale keypoints for each octave
    keypoints_1 = []
    keypoints_2 = []
    img_size1 = []
    img_size2 = []
    for octave in range(nOctaves):
        sc = G_resize ** octave
        kps_1t = np.round(kps_1[:, :2] / sc).astype(int)
        kps_2t = np.round(kps_2[:, :2] / sc).astype(int)
        # Unique rows
        _, u1 = np.unique(kps_1t, axis=0, return_index=True)
        _, u2 = np.unique(kps_2t, axis=0, return_index=True)
        # Store: [xt, yt, x, y, idx]
        kp1 = np.hstack([kps_1t[u1], kps_1[u1]])
        kp2 = np.hstack([kps_2t[u2], kps_2[u2]])
        keypoints_1.append(kp1)
        keypoints_2.append(kp2)
        img_size1.append(MOM_pyr1[octave][0].shape)
        img_size2.append(MOM_pyr2[octave][0].shape)

    matches = {}
    confidence = {}

    descriptors_1 = [[None] * nLayers for _ in range(nOctaves)]
    descriptors_2 = [[None] * nLayers for _ in range(nOctaves)]
    idx_1 = []
    idx_2 = []
    ref = [None, None]

    # Iterate from coarsest to finest octave
    for octave2 in range(nOctaves - 1, -1, -1):
        kp2 = keypoints_2[octave2]
        mag2 = np.ones(img_size2[octave2])

        octave1 = octave2  # scl_flag=False: only same-octave matching
        kp1 = keypoints_1[octave1]
        mag1 = np.ones(img_size1[octave1])

        for layer2 in range(nLayers):
            # Lazy descriptor computation for image 2
            if descriptors_2[octave2][layer2] is None:
                kp2_filtered = _sample_out(kp2, idx_2, 4)
                desc2 = gpolar_descriptor(
                    mag2, MOM_pyr2[octave2][layer2], kp2_filtered,
                    patch_size, NBA, NBO, rot_flag)
                # Reorder: [x,y,xt,yt,orient,idx,...des...]
                if len(desc2) > 0:
                    desc2 = np.hstack([desc2[:, 2:4], desc2[:, :2],
                                       desc2[:, 5:6], desc2[:, 4:5],
                                       desc2[:, 6:]])
                descriptors_2[octave2][layer2] = desc2
            else:
                desc2 = _sample_out(descriptors_2[octave2][layer2], idx_2, 5)
                descriptors_2[octave2][layer2] = desc2

            for layer1 in range(nLayers):
                # Lazy descriptor computation for image 1
                if descriptors_1[octave1][layer1] is None:
                    kp1_filtered = _sample_out(kp1, idx_1, 4)
                    if len(kp1_filtered) < 3:
                        continue
                    desc1 = gpolar_descriptor(
                        mag1, MOM_pyr1[octave1][layer1], kp1_filtered,
                        patch_size, NBA, NBO, rot_flag)
                    if len(desc1) > 0:
                        desc1 = np.hstack([desc1[:, 2:4], desc1[:, :2],
                                           desc1[:, 5:6], desc1[:, 4:5],
                                           desc1[:, 6:]])
                    descriptors_1[octave1][layer1] = desc1
                else:
                    desc1 = _sample_out(descriptors_1[octave1][layer1], idx_1, 5)
                    descriptors_1[octave1][layer1] = desc1

                if desc2 is None or len(desc2) == 0 or \
                   desc1 is None or len(desc1) == 0:
                    continue

                match, ncm = match_keypoints(desc1, desc2, error, K,
                                             trans_form, ref)
                key = (octave1, layer1, octave2, layer2)
                matches[key] = match
                confidence[key] = ncm

                if ncm > 50:
                    idx_1 = match[:, 5].astype(int).tolist()
                    idx_2 = match[:, 11].astype(int).tolist()
                    ref[0] = match[:, :6]
                    ref[1] = match[:, 6:12]

    # ── Optimise: aggregate by octave pair ──────────────────────────────────
    Confidence = np.zeros((nOctaves, nOctaves))
    Matches_by_oct = {}

    for oct1 in range(nOctaves):
        for oct2 in range(nOctaves):
            acc = []
            for l1 in range(nLayers):
                for l2 in range(nLayers):
                    m = matches.get((oct1, l1, oct2, l2), np.empty((0, 12)))
                    if len(m) > 0:
                        acc.append(m)
            if acc:
                acc = np.vstack(acc)
                if len(acc) > 20:
                    _, u = np.unique(acc[:, :2], axis=0, return_index=True)
                    acc = acc[u]
                    _, u = np.unique(acc[:, 6:8], axis=0, return_index=True)
                    acc = acc[u]
                if len(acc) > 20:
                    Matches_by_oct[(oct1, oct2)] = acc
                    Confidence[oct1, oct2] = len(acc)

    if Confidence.max() == 0:
        return np.empty((0, 6)), np.empty((0, 6))

    best = np.unravel_index(np.argmax(Confidence), Confidence.shape)
    best_o1, best_o2 = best

    # Combine matches from diagonal band around best octave pair
    MMatches = []
    lo = -min(best_o1, best_o2)
    hi = min(nOctaves - 1 - best_o1, nOctaves - 1 - best_o2) + 1
    for di in range(lo, hi):
        m = Matches_by_oct.get((best_o1 + di, best_o2 + di))
        if m is not None and len(m) > 3:
            MMatches.append(m)

    if not MMatches:
        return np.empty((0, 6)), np.empty((0, 6))

    MMatches = np.vstack(MMatches)
    _, u = np.unique(MMatches[:, :2], axis=0, return_index=True)
    MMatches = MMatches[u]
    _, u = np.unique(MMatches[:, 6:8], axis=0, return_index=True)
    MMatches = MMatches[u]

    # ── Final outlier removal ────────────────────────────────────────────────
    best_ncm = 0
    best_inlier = None
    for _ in range(K):
        _, _, inlier = outlier_removal(
            MMatches[:, :6], MMatches[:, 6:], error, 800, trans_form, [None, None])
        if inlier.sum() > best_ncm:
            best_ncm = inlier.sum()
            best_inlier = inlier

    if best_inlier is None or best_inlier.sum() == 0:
        return np.empty((0, 6)), np.empty((0, 6))

    MMatches = MMatches[best_inlier]
    cor1 = MMatches[:, :6]
    cor2 = MMatches[:, 6:]
    return cor1, cor2


def _sample_out(samples, id_list, loc_col):
    """Remove rows from samples where column loc_col is in id_list."""
    if samples is None or len(samples) == 0 or len(id_list) == 0:
        return samples
    if loc_col >= samples.shape[1]:
        return samples
    mask = np.isin(samples[:, loc_col], id_list)
    return samples[~mask]


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_image(img_path, min_size=64, max_size=512):
    """
    Load image, convert to float64 greyscale, resize to fit [min_size, max_size],
    normalise to [0, 1].  Returns (img_float, resample_factor).
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    else:
        gray = img.astype(np.float64)

    h, w = gray.shape
    resample = 1.0
    if max(h, w) > max_size:
        resample = max_size / max(h, w)
    elif min(h, w) < min_size and min(h, w) > 0:
        resample = min_size / min(h, w)

    if resample != 1.0:
        nh = max(1, int(round(h * resample)))
        nw = max(1, int(round(w * resample)))
        gray = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_CUBIC)

    mn, mx = gray.min(), gray.max()
    if mx > mn:
        gray = (gray - mn) / (mx - mn)
    return gray, resample


# ─────────────────────────────────────────────────────────────────────────────
# TOP-LEVEL HOMO MATCH FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def homo_match(img1_path, img2_path,
               int_flag=1, rot_flag=True, scl_flag=False,
               nOctaves=3, nLayers=2, G_resize=1.2, G_sigma=1.6,
               key_type='PC-ShiTomasi', thresh=0, radius=1, Npoint=5000,
               patch_size=72, NBA=12, NBO=12, error=5, K=1,
               trans_form='affine', verbose=True):
    """
    Full HOMO-Feature image matching pipeline.

    Returns dict with keys:
      ncm       – Number of Correct Matches (inlier count after RANSAC)
      cor1      – Matched coords in image 1 (N×2, original-image scale)
      cor2      – Matched coords in image 2 (N×2, original-image scale)
      time_total – Wall-clock seconds for the full pipeline
    """
    import time as _time
    t0 = _time.time()

    def log(msg):
        if verbose:
            print(msg)

    # ── Load & preprocess ────────────────────────────────────────────────────
    log(f"\n** Image matching starts: {img1_path} vs {img2_path}")
    I1, resample1 = preprocess_image(img1_path)
    I2, resample2 = preprocess_image(img2_path)
    log(f"   Image 1: {I1.shape}  resample={resample1:.3f}")
    log(f"   Image 2: {I2.shape}  resample={resample2:.3f}")

    # ── Build HOMO pyramids ──────────────────────────────────────────────────
    t1 = _time.time()
    MOM_pyr1, DoM_pyr1 = build_homo_pyramid(
        I1, nOctaves, nLayers, G_resize, G_sigma, patch_size, NBA, int_flag, key_type)
    log(f" Done: HOMO pyramid image 1, time={_time.time()-t1:.1f}s")

    t1 = _time.time()
    MOM_pyr2, DoM_pyr2 = build_homo_pyramid(
        I2, nOctaves, nLayers, G_resize, G_sigma, patch_size, NBA, int_flag, key_type)
    log(f" Done: HOMO pyramid image 2, time={_time.time()-t1:.1f}s")

    # ── Detect keypoints ─────────────────────────────────────────────────────
    h1, w1 = I1.shape; h2, w2 = I2.shape
    ratio = np.sqrt((h1 * w1) / (h2 * w2))
    r1 = round(radius * ratio) if ratio >= 1 else radius
    r2 = round(radius / ratio) if ratio < 1 else radius
    r1 = max(1, int(r1)); r2 = max(1, int(r2))

    t1 = _time.time()
    kps1 = detect_homo_keypoints(I1, DoM_pyr1, 6, thresh, r1, Npoint, G_resize, key_type)
    log(f" Done: Keypoints image 1: {len(kps1)} pts, time={_time.time()-t1:.1f}s")

    t1 = _time.time()
    kps2 = detect_homo_keypoints(I2, DoM_pyr2, 6, thresh, r2, Npoint, G_resize, key_type)
    log(f" Done: Keypoints image 2: {len(kps2)} pts, time={_time.time()-t1:.1f}s")

    if len(kps1) < 4 or len(kps2) < 4:
        log("** Too few keypoints detected!")
        return {'ncm': 0, 'cor1': np.empty((0, 2)), 'cor2': np.empty((0, 2)),
                'time_total': _time.time() - t0}

    # ── Multiscale matching ───────────────────────────────────────────────────
    t1 = _time.time()
    cor1, cor2 = multiscale_strategy(
        kps1, kps2, MOM_pyr1, MOM_pyr2,
        patch_size, NBA, NBO, G_resize,
        error, K, trans_form, rot_flag, scl_flag)
    log(f" Done: Multiscale matching, time={_time.time()-t1:.1f}s")

    ncm = len(cor1)
    total_time = _time.time() - t0
    log(f"* Done! NCM={ncm}, total time={total_time:.1f}s")

    # Convert coordinates back to original image scale
    cor1_orig = cor1[:, :2] / resample1 if ncm > 0 else np.empty((0, 2))
    cor2_orig = cor2[:, :2] / resample2 if ncm > 0 else np.empty((0, 2))

    return {
        'ncm': ncm,
        'cor1': cor1_orig,
        'cor2': cor2_orig,
        'time_total': total_time,
    }
