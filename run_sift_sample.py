"""
SIFT baseline on the same 50 LLVIP pairs from llvip_sample.csv.
Uses OpenCV SIFT + BFMatcher ratio test (0.8) + RANSAC (error=5).
Saves results/sift_sample.csv.
"""

import sys
import csv
import time
from pathlib import Path

import numpy as np
import cv2

WORKSPACE   = Path(__file__).parent
LLVIP_VIS   = WORKSPACE / "LLVIP" / "visible"
LLVIP_IR    = WORKSPACE / "LLVIP" / "infrared"
SAMPLE_CSV  = WORKSPACE / "results" / "llvip_sample.csv"
OUT_CSV     = WORKSPACE / "results" / "sift_sample.csv"
RESULTS_DIR = WORKSPACE / "results" / "sift_sample"

CSV_HEADER = ["pair_name", "split", "NCM", "time_s", "status"]
MAX_DIM    = 512        # match HOMO preprocessing
RANSAC_ERR = 5          # same threshold as HOMO (pixels)
RATIO_TEST = 0.8        # Lowe's ratio threshold


def load_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape
    s = min(1.0, MAX_DIM / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def sift_ncm(vis_path, ir_path):
    """Return (ncm, elapsed)."""
    t0 = time.time()
    img1 = load_gray(vis_path)
    img2 = load_gray(ir_path)
    if img1 is None or img2 is None:
        return 0, time.time() - t0

    sift = cv2.SIFT_create(nfeatures=5000)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return 0, time.time() - t0

    # BFMatcher + ratio test
    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in raw if m.distance < RATIO_TEST * n.distance]

    if len(good) < 4:
        return 0, time.time() - t0

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    _, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, RANSAC_ERR)
    ncm = int(mask.sum()) if mask is not None else 0
    return ncm, time.time() - t0


def compute_summary(records):
    total     = len(records)
    passed    = [r for r in records if r['status'] == 'PASS']
    failed    = [r for r in records if r['status'] == 'FAIL']
    all_ncms  = [r['NCM'] for r in records]
    pass_ncms = [r['NCM'] for r in passed]
    times     = [r['time_s'] for r in records]
    lines = [
        "=" * 60,
        "SIFT BASELINE — LLVIP 50-PAIR SAMPLE SUMMARY (seed=42)",
        "=" * 60,
        f"Total pairs tested    : {total}",
        f"PASS (NCM >= 50)      : {len(passed)}",
        f"FAIL (NCM <  50)      : {len(failed)}",
        f"Success rate          : {100*len(passed)/total:.1f}%",
        "-" * 60,
        (f"Mean NCM (PASS only)  : {sum(pass_ncms)/len(pass_ncms):.1f}"
         if pass_ncms else "Mean NCM (PASS only)  : N/A"),
        f"Mean NCM (all pairs)  : {sum(all_ncms)/total:.1f}",
        f"Min NCM               : {min(all_ncms)}",
        f"Max NCM               : {max(all_ncms)}",
        f"Mean time per pair    : {sum(times)/total:.1f}s",
        f"Total wall time       : {sum(times)/60:.1f}min",
        "-" * 60,
    ]
    if failed:
        lines.append(f"FAILED pairs ({len(failed)}):")
        for r in failed:
            lines.append(f"  {r['pair_name']}  NCM={r['NCM']}")
    else:
        lines.append("All sampled pairs PASSED.")
    lines.append("=" * 60)
    return "\n".join(lines)


def main():
    # Load the 50 pairs in the exact order from llvip_sample.csv
    sample_rows = list(csv.DictReader(open(SAMPLE_CSV, encoding='utf-8')))
    pairs = []
    for row in sample_rows:
        name  = row['pair_name']
        split = row['split']
        vp = LLVIP_VIS / split / f"{name}.jpg"
        ip = LLVIP_IR  / split / f"{name}.jpg"
        if vp.exists() and ip.exists():
            pairs.append((name, split, vp, ip))
        else:
            print(f"[WARN] missing: {name}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"SIFT BASELINE on {len(pairs)} LLVIP pairs (same as HOMO sample)")
    print(f"  Detector : OpenCV SIFT (nfeatures=5000)")
    print(f"  Matching : BFMatcher L2 + ratio test {RATIO_TEST}")
    print(f"  RANSAC   : findHomography, error={RANSAC_ERR}px")
    print("=" * 60)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        csv.DictWriter(f, fieldnames=CSV_HEADER).writeheader()

    records = []
    for i, (name, split, vp, ip) in enumerate(pairs, 1):
        print(f"  [{i:>2}/{len(pairs)}] {name} ...", end='', flush=True)
        ncm, elapsed = sift_ncm(vp, ip)
        status = 'PASS' if ncm >= 50 else 'FAIL'
        print(f" NCM={ncm:>5}  {elapsed:>5.1f}s  {status}")

        row = {'pair_name': name, 'split': split,
               'NCM': ncm, 'time_s': round(elapsed, 2), 'status': status}
        records.append(row)
        with open(OUT_CSV, 'a', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=CSV_HEADER).writerow(row)

        if i % 5 == 0 or i == len(pairs):
            pass_n = sum(1 for r in records if r['status'] == 'PASS')
            print(f"\n  --- {i}/{len(pairs)} | PASS {pass_n}/{i} ---\n")

    summary = compute_summary(records)
    print("\n" + summary)
    (WORKSPACE / "results" / "sift_sample_summary.txt").write_text(
        summary + "\n", encoding="utf-8")
    print(f"\nCSV saved: {OUT_CSV}")


if __name__ == "__main__":
    main()
