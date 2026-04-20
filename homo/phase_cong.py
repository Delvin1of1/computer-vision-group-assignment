"""
Python port of phasecong3.m (Kovesi phase congruency).

Original MATLAB by Peter Kovesi, ported here for use in the HOMO pipeline.
Only M (maximum moment / edge strength) and m (minimum moment / corner
strength) are computed fully; the remaining six return slots are None.

Reference: Kovesi, P. "Image Features From Phase Congruency." Videre 1(3), 1999.
"""

import numpy as np

from log_gabor import lowpass_filter


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #

def _rayleigh_mode(data: np.ndarray, nbins: int = 50) -> float:
    """
    Matches MATLAB rayleighmode: histogram mode of a Rayleigh-distributed sample.
    Edges are uniformly spaced 0..max over nbins+1 points; bin centres are returned.
    """
    mx = float(data.max())
    if mx == 0:
        return 0.0
    edges = np.linspace(0.0, mx, nbins + 1)
    counts, _ = np.histogram(data.ravel(), bins=edges)
    ind = int(np.argmax(counts))
    return (edges[ind] + edges[ind + 1]) / 2.0


# --------------------------------------------------------------------------- #
# Main function                                                                 #
# --------------------------------------------------------------------------- #

def phase_cong3(
    im: np.ndarray,
    nscale: int = 4,
    norient: int = 6,
    min_wave_length: int = 3,
    mult: float = 1.6,
    sigma_onf: float = 0.75,
    k: float = 1.0,
    cut_off: float = 0.5,
    g: float = 3.0,
    noise_method: float = -1,
):
    """
    Phase congruency (PC_2 measure).  Faithful port of phasecong3.m.

    Parameters mirror the HOMO call:
        phasecong3(I, 4, 6, 3, 'mult', 1.6, 'sigmaOnf', 0.75, 'g', 3, 'k', 1)

    Returns
    -------
    M : maximum moment of the PC covariance (edge strength)
    m : minimum moment of the PC covariance (corner strength)
    six None placeholders for the remaining MATLAB outputs
      (or, featType, PC, EO, T, pcSum)
    """
    im = np.asarray(im, dtype=float)
    if im.ndim == 3:
        im = im[:, :, 0]

    epsilon = 1e-4
    rows, cols = im.shape
    image_fft = np.fft.fft2(im)

    zero = np.zeros((rows, cols))

    # ------------------------------------------------------------------ #
    # Frequency grid — identical to log_gabor.py / phasecong3.m           #
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
    radius[0, 0] = 1.0

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    lp = lowpass_filter(rows, cols, cutoff=0.45, n=15)

    # ------------------------------------------------------------------ #
    # Radial log-Gabor envelopes                                           #
    # ------------------------------------------------------------------ #
    log_sigma_sq = 2.0 * np.log(sigma_onf) ** 2
    lg_filters = []
    for s in range(nscale):
        wavelength = min_wave_length * mult ** s
        fo = 1.0 / wavelength
        lg = np.exp(-(np.log(radius / fo)) ** 2 / log_sigma_sq)
        lg = lg * lp
        lg[0, 0] = 0.0
        lg_filters.append(lg)

    # ------------------------------------------------------------------ #
    # Covariance accumulators                                              #
    # ------------------------------------------------------------------ #
    covx2 = zero.copy()
    covy2 = zero.copy()
    covxy = zero.copy()

    # ------------------------------------------------------------------ #
    # Main loop over orientations                                          #
    # ------------------------------------------------------------------ #
    for o in range(norient):
        angl = o * np.pi / norient

        # Angular spread
        ds = sin_theta * np.cos(angl) - cos_theta * np.sin(angl)
        dc = cos_theta * np.cos(angl) + sin_theta * np.sin(angl)
        d_theta = np.abs(np.arctan2(ds, dc))
        d_theta = np.minimum(d_theta * norient / 2.0, np.pi)
        spread = (np.cos(d_theta) + 1.0) / 2.0

        sum_e = zero.copy()
        sum_o = zero.copy()
        sum_an = zero.copy()
        max_an = None
        tau = 0.0

        # Store real/imag parts for second pass
        e_per_scale = []
        o_per_scale = []

        for s in range(nscale):
            filt = lg_filters[s] * spread
            eo = np.fft.ifft2(image_fft * filt)
            E = eo.real
            O = eo.imag
            An = np.abs(eo)

            e_per_scale.append(E)
            o_per_scale.append(O)

            sum_an += An
            sum_e += E
            sum_o += O

            if s == 0:
                if noise_method == -1:
                    tau = float(np.median(An)) / np.sqrt(np.log(4))
                elif noise_method == -2:
                    tau = _rayleigh_mode(An)
                max_an = An.copy()
            else:
                max_an = np.maximum(max_an, An)

        # ------------------------------------------------------------------ #
        # Noise threshold                                                      #
        # ------------------------------------------------------------------ #
        if noise_method >= 0:
            T = float(noise_method)
        else:
            total_tau = tau * (1.0 - (1.0 / mult) ** nscale) / (1.0 - 1.0 / mult)
            est_mean  = total_tau * np.sqrt(np.pi / 2.0)
            est_sigma = total_tau * np.sqrt((4.0 - np.pi) / 2.0)
            T = est_mean + k * est_sigma

        # ------------------------------------------------------------------ #
        # Weighted mean phase response                                         #
        # ------------------------------------------------------------------ #
        x_energy = np.sqrt(sum_e ** 2 + sum_o ** 2) + epsilon
        mean_e = sum_e / x_energy
        mean_o = sum_o / x_energy

        # ------------------------------------------------------------------ #
        # Energy: sum over scales of phase deviation measure                  #
        # ------------------------------------------------------------------ #
        energy = zero.copy()
        for s in range(nscale):
            E = e_per_scale[s]
            O = o_per_scale[s]
            energy += E * mean_e + O * mean_o - np.abs(E * mean_o - O * mean_e)

        energy = np.maximum(energy - T, 0.0)

        # ------------------------------------------------------------------ #
        # Frequency spread weighting (sigmoid)                                #
        # ------------------------------------------------------------------ #
        width = (sum_an / (max_an + epsilon) - 1.0) / (nscale - 1)
        weight = 1.0 / (1.0 + np.exp((cut_off - width) * g))

        # Phase congruency for this orientation
        pc_o = weight * energy / (sum_an + epsilon)

        # ------------------------------------------------------------------ #
        # Accumulate covariance data                                           #
        # ------------------------------------------------------------------ #
        covx = pc_o * np.cos(angl)
        covy = pc_o * np.sin(angl)
        covx2 += covx ** 2
        covy2 += covy ** 2
        covxy += covx * covy

    # ------------------------------------------------------------------ #
    # Maximum and minimum moments of PC covariance                         #
    # covx2 /= norient/2;  covy2 /= norient/2;  covxy = 4*covxy/norient  #
    # ------------------------------------------------------------------ #
    covx2 /= (norient / 2.0)
    covy2 /= (norient / 2.0)
    covxy = 4.0 * covxy / norient

    denom = np.sqrt(covxy ** 2 + (covx2 - covy2) ** 2) + epsilon
    M = (covy2 + covx2 + denom) / 2.0
    m = (covy2 + covx2 - denom) / 2.0

    return M, m, None, None, None, None, None, None


# ======================================================================== #
# Smoke test                                                                 #
# ======================================================================== #
if __name__ == "__main__":
    size = 128
    cx, cy = size // 2, size // 2
    gx = np.arange(size) - cx
    gy = np.arange(size) - cy
    xx, yy = np.meshgrid(gx, gy)
    img = np.exp(-(xx ** 2 + yy ** 2) / (2 * 20.0 ** 2))

    M, m, *_ = phase_cong3(img)

    assert M.shape == img.shape
    assert m.shape == img.shape
    assert float(M.max()) > 0, "M is all zeros"
    assert float(m.max()) > 0, "m is all zeros"
    print(f"M: min={M.min():.6f}  max={M.max():.6f}  mean={M.mean():.6f}")
    print(f"m: min={m.min():.6f}  max={m.max():.6f}  mean={m.mean():.6f}")
    print("phase_cong3 smoke test passed.")
