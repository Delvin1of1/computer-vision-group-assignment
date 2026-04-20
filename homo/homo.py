"""
Top-level HOMO pipeline.

Public API
----------
run_homo(img1_path, img2_path, params=None) -> dict
    Runs the full HOMO feature matching pipeline on two images.

    Returns
    -------
    {
        'ncm'       : int           — RANSAC inlier count (len(cor1))
        'cor1'      : (M, 6) array  — inlier coords from image 1 [x,y,xt,yt,orient,idx]
        'cor2'      : (M, 6) array  — inlier coords from image 2
        'match_img' : (H, 2W, 3) uint8 array — side-by-side visualisation with yellow match lines
    }
"""

import os
import sys
import time

import cv2
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from pyramid  import build_homo_pyramid
from keypoints import detect_homo_keypoint
from matching  import multiscale_strategy


# =========================================================================== #
# Default parameters — mirror A_HOMO_demo.m                                    #
# =========================================================================== #

_DEFAULTS = dict(
    n_octaves  = 3,
    n_layers   = 4,
    g_resize   = 2,
    g_sigma    = 1.6,
    patch_size = 72,
    nba        = 12,
    nbo        = 12,
    error      = 5,
    K          = 1,
    int_flag   = True,
    rot_flag   = True,
    scl_flag   = False,
    key_type   = 'PC-ShiTomasi',
    n_points   = 5000,
    trans_form = 'affine',
)


def _load_gray(path: str) -> np.ndarray:
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(float) / 255.0


def _draw_matches(img1: np.ndarray, img2: np.ndarray,
                  cor1: np.ndarray, cor2: np.ndarray) -> np.ndarray:
    """
    Side-by-side match visualisation with yellow lines.

    img1, img2 : float [0,1] grayscale
    cor1, cor2 : (M, ≥2) arrays; first two cols are x (col) and y (row)
    """
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    H = max(h1, h2)

    def to_u8(I):
        return (np.clip(I, 0, 1) * 255).astype(np.uint8)

    # Convert to BGR panels
    p1 = cv2.cvtColor(to_u8(img1), cv2.COLOR_GRAY2BGR)
    p2 = cv2.cvtColor(to_u8(img2), cv2.COLOR_GRAY2BGR)

    # Pad shorter panel to equal height
    if h1 < H:
        p1 = np.vstack([p1, np.zeros((H - h1, w1, 3), dtype=np.uint8)])
    if h2 < H:
        p2 = np.vstack([p2, np.zeros((H - h2, w2, 3), dtype=np.uint8)])

    canvas = np.hstack([p1, p2])
    yellow = (0, 255, 255)   # BGR

    if len(cor1) == 0:
        return canvas

    for i in range(len(cor1)):
        x1, y1 = int(round(float(cor1[i, 0]))), int(round(float(cor1[i, 1])))
        x2, y2 = int(round(float(cor2[i, 0]))) + w1, int(round(float(cor2[i, 1])))
        cv2.line(canvas, (x1, y1), (x2, y2), yellow, 1, cv2.LINE_AA)

    return canvas


# =========================================================================== #
# Public API                                                                    #
# =========================================================================== #

def run_homo(img1_path: str, img2_path: str, params: dict = None) -> dict:
    """
    Run the full HOMO matching pipeline.

    Parameters
    ----------
    img1_path : path to image 1 (infrared / reference)
    img2_path : path to image 2 (visible / query)
    params    : optional dict to override any key in _DEFAULTS

    Returns
    -------
    dict with keys: 'ncm', 'cor1', 'cor2', 'match_img'
    """
    p = dict(_DEFAULTS)
    if params:
        p.update(params)

    I1 = _load_gray(img1_path)
    I2 = _load_gray(img2_path)

    mom1, dom1 = build_homo_pyramid(
        I1, p['n_octaves'], p['n_layers'], p['g_resize'], p['g_sigma'],
        p['patch_size'], p['nba'], p['int_flag'], p['key_type'],
    )
    mom2, dom2 = build_homo_pyramid(
        I2, p['n_octaves'], p['n_layers'], p['g_resize'], p['g_sigma'],
        p['patch_size'], p['nba'], p['int_flag'], p['key_type'],
    )

    kps1 = detect_homo_keypoint(
        I1, dom1, 6, 0, 1, p['n_points'], p['g_resize'], p['key_type'],
    )
    kps2 = detect_homo_keypoint(
        I2, dom2, 6, 0, 1, p['n_points'], p['g_resize'], p['key_type'],
    )

    cor1, cor2 = multiscale_strategy(
        kps1, kps2, mom1, mom2,
        p['patch_size'], p['nba'], p['nbo'], p['g_resize'],
        p['error'], p['K'], p['trans_form'], p['rot_flag'], p['scl_flag'],
    )

    ncm       = len(cor1)
    match_img = _draw_matches(I1, I2, cor1, cor2)

    return {
        'ncm'       : ncm,
        'cor1'      : cor1,
        'cor2'      : cor2,
        'match_img' : match_img,
    }
