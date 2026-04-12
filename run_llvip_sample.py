"""
LLVIP 50-pair random sample runner for HOMO-Feature matching.
Fixed seed=42 ensures reproducibility.

Usage:
    python run_llvip_sample.py
    python run_llvip_sample.py --n 50 --seed 42   # explicit defaults
    python run_llvip_sample.py --split train        # sample from train split
"""

import sys
import csv
import time
import argparse
from pathlib import Path

import numpy as np
import cv2

# ── Paths ─────────────────────────────────────────────────────────────────────
WORKSPACE    = Path(__file__).parent
LLVIP_VIS    = WORKSPACE / "LLVIP" / "visible"
LLVIP_IR     = WORKSPACE / "LLVIP" / "infrared"
RESULTS_DIR  = WORKSPACE / "results" / "llvip_sample"
CSV_PATH     = WORKSPACE / "results" / "llvip_sample.csv"
SUMMARY_PATH = WORKSPACE / "results" / "llvip_sample_summary.txt"

sys.path.insert(0, str(WORKSPACE))
from homo_feature import homo_match

CSV_HEADER = ["pair_name", "split", "NCM", "time_s", "status"]


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def _load_bgr(path, max_dim=512):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    s = min(1.0, max_dim / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def save_overlap(vis_path, ir_path, cor1_xy, cor2_xy, out_path):
    """Side-by-side with up to 200 random match lines."""
    img1 = _load_bgr(vis_path)
    img2 = _load_bgr(ir_path)
    if img1 is None or img2 is None:
        return
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = max(h1, h2)
    canvas = np.zeros((h, w1 + 4 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1 + 4:] = img2
    if len(cor1_xy) > 0:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(cor1_xy), min(200, len(cor1_xy)), replace=False)
        for i in idx:
            pt1 = (int(cor1_xy[i, 0]), int(cor1_xy[i, 1]))
            pt2 = (int(cor2_xy[i, 0]) + w1 + 4, int(cor2_xy[i, 1]))
            c = tuple(int(x) for x in rng.integers(100, 255, 3))
            cv2.line(canvas, pt1, pt2, c, 1, cv2.LINE_AA)
            cv2.circle(canvas, pt1, 3, c, -1)
            cv2.circle(canvas, pt2, 3, c, -1)
    cv2.imwrite(str(out_path), canvas)


def save_mosaic(vis_path, ir_path, cor1_xy, cor2_xy, out_path):
    """Warp IR onto visible, alpha-blend for mosaic overlay."""
    img1 = _load_bgr(vis_path)
    img2 = _load_bgr(ir_path)
    if img1 is None or img2 is None or len(cor1_xy) < 4:
        return
    h, w = img1.shape[:2]
    try:
        M, _ = cv2.estimateAffinePartial2D(
            cor2_xy.astype(np.float32), cor1_xy.astype(np.float32),
            method=cv2.RANSAC, ransacReprojThreshold=10.0)
    except Exception:
        M = None
    if M is None:
        return
    warped = cv2.warpAffine(img2, M, (w, h))
    ir_col = cv2.applyColorMap(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY),
                               cv2.COLORMAP_INFERNO)
    mosaic = cv2.addWeighted(img1, 0.6, ir_col, 0.4, 0)
    cv2.imwrite(str(out_path), mosaic)


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def compute_summary(records):
    total     = len(records)
    passed    = [r for r in records if r['status'] == 'PASS']
    failed    = [r for r in records if r['status'] == 'FAIL']
    all_ncms  = [r['NCM'] for r in records]
    pass_ncms = [r['NCM'] for r in passed]
    times     = [r['time_s'] for r in records]

    lines = [
        "=" * 60,
        "LLVIP 50-PAIR SAMPLE SUMMARY (seed=42)",
        "=" * 60,
        f"Total pairs sampled   : {total}",
        f"PASS (NCM >= 50)      : {len(passed)}",
        f"FAIL (NCM <  50)      : {len(failed)}",
        f"Success rate          : {100*len(passed)/total:.1f}%",
        "-" * 60,
        f"Mean NCM (PASS only)  : {sum(pass_ncms)/len(pass_ncms):.1f}" if pass_ncms else "Mean NCM (PASS only)  : N/A",
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


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",     type=int, default=50,   help="Number of pairs to sample")
    parser.add_argument("--seed",  type=int, default=42,   help="Random seed")
    parser.add_argument("--split", default='test', choices=['test', 'train', 'all'])
    args = parser.parse_args()

    # Collect all available pairs
    vis_dirs = []
    ir_dirs  = []
    if args.split in ('test', 'all'):
        vis_dirs.append(LLVIP_VIS / 'test')
        ir_dirs.append(LLVIP_IR  / 'test')
    if args.split in ('train', 'all'):
        vis_dirs.append(LLVIP_VIS / 'train')
        ir_dirs.append(LLVIP_IR  / 'train')

    all_pairs = []
    for vd, id_ in zip(vis_dirs, ir_dirs):
        if not vd.exists():
            print(f"[WARN] {vd} not found")
            continue
        for vp in sorted(vd.glob("*.jpg")):
            ip = id_ / vp.name
            if ip.exists():
                all_pairs.append((vp, ip))

    if len(all_pairs) == 0:
        print("No pairs found. Check LLVIP directory.")
        sys.exit(1)

    # Reproducible sample
    rng    = np.random.default_rng(args.seed)
    chosen = rng.choice(len(all_pairs), size=min(args.n, len(all_pairs)), replace=False)
    chosen = sorted(chosen)                          # stable file order
    sample = [all_pairs[i] for i in chosen]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"LLVIP SAMPLE  n={args.n}  seed={args.seed}  split={args.split}")
    print("=" * 60)
    print(f"  Pool size     : {len(all_pairs)}")
    print(f"  Sampled pairs : {len(sample)}")
    print(f"  Est. time     : {len(sample)*125/60:.0f}min (~125s/pair)")
    print(f"  CSV           : {CSV_PATH}")
    print(f"  Results dir   : {RESULTS_DIR}")
    print("=" * 60)

    # Write CSV header (overwrite any previous run)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        csv.DictWriter(f, fieldnames=CSV_HEADER).writeheader()

    records = []
    batch_t0 = time.time()

    for i, (vis_path, ir_path) in enumerate(sample, 1):
        name     = vis_path.stem
        pair_dir = RESULTS_DIR / name
        pair_dir.mkdir(parents=True, exist_ok=True)

        print(f"  [{i:>2}/{len(sample)}] {name} ...", end='', flush=True)
        t0 = time.time()

        try:
            result = homo_match(
                str(vis_path), str(ir_path),
                int_flag=1, rot_flag=True, scl_flag=False,
                nOctaves=3, nLayers=2, G_resize=1.2, G_sigma=1.6,
                key_type='PC-ShiTomasi', thresh=0, radius=1, Npoint=5000,
                patch_size=72, NBA=12, NBO=12, error=10, K=5,
                trans_form='affine', verbose=False
            )
            ncm     = result['ncm']
            cor1_xy = result['cor1'][:, :2] if len(result['cor1']) > 0 else np.empty((0, 2))
            cor2_xy = result['cor2'][:, :2] if len(result['cor2']) > 0 else np.empty((0, 2))
        except Exception as e:
            print(f" ERROR: {e}")
            ncm, cor1_xy, cor2_xy = 0, np.empty((0, 2)), np.empty((0, 2))

        elapsed = time.time() - t0
        status  = 'PASS' if ncm >= 50 else 'FAIL'
        print(f" NCM={ncm:>5}  {elapsed:>6.1f}s  {status}")

        # Visualisations
        try:
            save_overlap(vis_path, ir_path, cor1_xy, cor2_xy, pair_dir / "overlap.png")
            save_mosaic (vis_path, ir_path, cor1_xy, cor2_xy, pair_dir / "mosaic.png")
        except Exception as e:
            print(f"    [WARN] vis failed: {e}")

        # result.txt
        (pair_dir / "result.txt").write_text(
            f"Pair:   {name}\n"
            f"Vis:    {vis_path}\n"
            f"IR:     {ir_path}\n"
            f"NCM:    {ncm}\n"
            f"Time:   {elapsed:.2f}s\n"
            f"PASS:   {'YES' if ncm >= 50 else 'NO'}\n",
            encoding="utf-8"
        )

        row = {'pair_name': name, 'split': args.split,
               'NCM': ncm, 'time_s': round(elapsed, 2), 'status': status}
        records.append(row)
        with open(CSV_PATH, 'a', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=CSV_HEADER).writerow(row)

        # Progress every 5 pairs
        if i % 5 == 0 or i == len(sample):
            elapsed_total = time.time() - batch_t0
            mean_t  = elapsed_total / i
            eta_min = (len(sample) - i) * mean_t / 60
            pass_n  = sum(1 for r in records if r['status'] == 'PASS')
            print(f"\n  --- {i}/{len(sample)} done | PASS {pass_n}/{i} "
                  f"| mean {mean_t:.0f}s/pair | ETA {eta_min:.1f}min ---\n")

    # Final summary
    summary = compute_summary(records)
    print("\n" + summary)
    SUMMARY_PATH.write_text(summary + "\n", encoding="utf-8")
    print(f"\nSummary saved: {SUMMARY_PATH}")
    print(f"CSV saved    : {CSV_PATH}")


if __name__ == "__main__":
    main()
