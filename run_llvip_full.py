"""
Full LLVIP batch runner for HOMO-Feature matching.

Usage:
    python run_llvip_full.py                     # test split only (3 463 pairs)
    python run_llvip_full.py --split train        # train split only (12 025 pairs)
    python run_llvip_full.py --split all          # both splits (~15 488 pairs)
    python run_llvip_full.py --resume             # skip pairs that already have result.txt
    python run_llvip_full.py --limit 100          # cap at first N pairs (for quick checks)

Features:
  - Resumable: --resume skips already-completed pairs
  - CSV written after every pair  (results/llvip_results.csv)
  - Per-pair result.txt, overlap.png, mosaic.png saved under results/llvip/<name>/
  - Progress printed every 10 pairs with live ETA
  - Full summary printed and saved to results/llvip_summary.txt on completion
"""

import sys
import os
import csv
import time
import argparse
from pathlib import Path

import numpy as np
import cv2

# ── Paths ────────────────────────────────────────────────────────────────────
WORKSPACE   = Path(__file__).parent
LLVIP_VIS   = WORKSPACE / "LLVIP" / "visible"
LLVIP_IR    = WORKSPACE / "LLVIP" / "infrared"
RESULTS_DIR = WORKSPACE / "results" / "llvip"
CSV_PATH    = WORKSPACE / "results" / "llvip_results.csv"
SUMMARY_PATH = WORKSPACE / "results" / "llvip_summary.txt"

# ── Import core algorithm ─────────────────────────────────────────────────────
sys.path.insert(0, str(WORKSPACE))
from homo_feature import homo_match, preprocess_image

CSV_HEADER = ["pair_name", "split", "NCM", "time_s", "status"]


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _load_bgr(path):
    """Load image as uint8 BGR for visualisation (keep original size, max 512)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = min(1.0, 512 / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def save_overlap(vis_path, ir_path, cor1_xy, cor2_xy, out_path):
    """
    Side-by-side image with up to 200 random match lines drawn between them.
    cor1_xy / cor2_xy are (N, 2) arrays in downscaled (max-512) coords.
    """
    img1 = _load_bgr(vis_path)
    img2 = _load_bgr(ir_path)
    if img1 is None or img2 is None:
        return

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = max(h1, h2)
    canvas = np.zeros((h, w1 + w2 + 4, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1 + 4:] = img2

    if len(cor1_xy) > 0:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(cor1_xy), min(200, len(cor1_xy)), replace=False)
        for i in idx:
            pt1 = (int(cor1_xy[i, 0]), int(cor1_xy[i, 1]))
            pt2 = (int(cor2_xy[i, 0]) + w1 + 4, int(cor2_xy[i, 1]))
            color = (rng.integers(100, 255).item(),
                     rng.integers(100, 255).item(),
                     rng.integers(100, 255).item())
            cv2.line(canvas, pt1, pt2, color, 1, cv2.LINE_AA)
            cv2.circle(canvas, pt1, 3, color, -1)
            cv2.circle(canvas, pt2, 3, color, -1)

    cv2.imwrite(str(out_path), canvas)


def save_mosaic(vis_path, ir_path, cor1_xy, cor2_xy, out_path):
    """
    Warp IR image onto visible using the affine transform from matched points,
    then alpha-blend the two for a mosaic overlay.
    cor1_xy / cor2_xy: (N, 2) in downscaled coords.
    """
    img1 = _load_bgr(vis_path)
    img2 = _load_bgr(ir_path)
    if img1 is None or img2 is None or len(cor1_xy) < 4:
        return

    h, w = img1.shape[:2]
    try:
        M, _ = cv2.estimateAffinePartial2D(
            cor2_xy.astype(np.float32),
            cor1_xy.astype(np.float32),
            method=cv2.RANSAC,
            ransacReprojThreshold=10.0
        )
    except Exception:
        M = None

    if M is None:
        return

    warped = cv2.warpAffine(img2, M, (w, h))

    # Alpha blend: 60% visible, 40% warped IR (converted to false colour)
    ir_colored = cv2.applyColorMap(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_INFERNO)
    mosaic = cv2.addWeighted(img1, 0.6, ir_colored, 0.4, 0)
    cv2.imwrite(str(out_path), mosaic)


# ─────────────────────────────────────────────────────────────────────────────
# PAIR RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_pair(vis_path, ir_path, pair_dir):
    """Run homo_match on one pair. Returns (ncm, elapsed, cor1_xy, cor2_xy)."""
    t0 = time.time()
    result = homo_match(
        str(vis_path), str(ir_path),
        int_flag=1, rot_flag=True, scl_flag=False,
        nOctaves=3, nLayers=2, G_resize=1.2, G_sigma=1.6,
        key_type='PC-ShiTomasi', thresh=0, radius=1, Npoint=5000,
        patch_size=72, NBA=12, NBO=12, error=10, K=5,
        trans_form='affine', verbose=False
    )
    elapsed = time.time() - t0
    ncm  = result['ncm']
    cor1 = result['cor1']   # (N, 6) in internal (downscaled) coords — col 0,1 = x,y
    cor2 = result['cor2']

    cor1_xy = cor1[:, :2] if len(cor1) > 0 else np.empty((0, 2))
    cor2_xy = cor2[:, :2] if len(cor2) > 0 else np.empty((0, 2))

    # per-pair result.txt
    passed = ncm >= 50
    pair_dir.mkdir(parents=True, exist_ok=True)
    (pair_dir / "result.txt").write_text(
        f"Pair:   {vis_path.stem}\n"
        f"Vis:    {vis_path}\n"
        f"IR:     {ir_path}\n"
        f"NCM:    {ncm}\n"
        f"Time:   {elapsed:.2f}s\n"
        f"PASS:   {'YES' if passed else 'NO'}\n",
        encoding="utf-8"
    )

    return ncm, elapsed, cor1_xy, cor2_xy


# ─────────────────────────────────────────────────────────────────────────────
# COLLECT PAIRS
# ─────────────────────────────────────────────────────────────────────────────

def collect_pairs(split):
    """Return list of (vis_path, ir_path, split_name) tuples."""
    splits = []
    if split in ('test', 'all'):
        splits.append(('test',
                        LLVIP_VIS / 'test',
                        LLVIP_IR  / 'test'))
    if split in ('train', 'all'):
        splits.append(('train',
                        LLVIP_VIS / 'train',
                        LLVIP_IR  / 'train'))

    pairs = []
    for split_name, vis_dir, ir_dir in splits:
        if not vis_dir.exists():
            print(f"[WARN] Visible dir not found: {vis_dir}")
            continue
        for vis_path in sorted(vis_dir.glob("*.jpg")):
            ir_path = ir_dir / vis_path.name
            if ir_path.exists():
                pairs.append((vis_path, ir_path, split_name))
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def compute_and_print_summary(records, out_file=None):
    if not records:
        print("No records to summarise.")
        return

    total      = len(records)
    passed     = [r for r in records if r['status'] == 'PASS']
    failed     = [r for r in records if r['status'] == 'FAIL']
    all_ncms   = [r['NCM'] for r in records]
    pass_ncms  = [r['NCM'] for r in passed]
    times      = [r['time_s'] for r in records]

    lines = [
        "=" * 60,
        "LLVIP FULL-BATCH SUMMARY",
        "=" * 60,
        f"Total pairs attempted : {total}",
        f"PASS (NCM >= 50)      : {len(passed)}",
        f"FAIL (NCM <  50)      : {len(failed)}",
        f"Success rate          : {100*len(passed)/total:.1f}%",
        "-" * 60,
        f"Mean NCM (PASS only)  : {sum(pass_ncms)/len(pass_ncms):.1f}" if pass_ncms else "Mean NCM (PASS only)  : N/A",
        f"Mean NCM (ALL pairs)  : {sum(all_ncms)/total:.1f}",
        f"Min NCM               : {min(all_ncms)}",
        f"Max NCM               : {max(all_ncms)}",
        f"Mean time per pair    : {sum(times)/total:.1f}s",
        f"Total wall time       : {sum(times)/3600:.2f}h",
        "-" * 60,
    ]

    if failed:
        lines.append(f"FAILED pairs ({len(failed)}):")
        for r in failed:
            lines.append(f"  {r['pair_name']} ({r['split']})  NCM={r['NCM']}")
    else:
        lines.append("All pairs PASSED.")

    lines.append("=" * 60)

    text = "\n".join(lines)
    print("\n" + text)

    if out_file:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(text + "\n", encoding="utf-8")
        print(f"\nSummary saved: {out_file}")


# ─────────────────────────────────────────────────────────────────────────────
# CSV HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_existing_csv(csv_path):
    """Return set of already-done pair names and list of existing records."""
    done = set()
    records = []
    if not csv_path.exists():
        return done, records
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                records.append({
                    'pair_name': row['pair_name'],
                    'split':     row.get('split', 'test'),
                    'NCM':       int(row['NCM']),
                    'time_s':    float(row['time_s']),
                    'status':    row['status'],
                })
                done.add(row['pair_name'])
            except (KeyError, ValueError):
                pass
    return done, records


def append_csv(csv_path, row_dict):
    """Append one row to the CSV, writing header if file is new."""
    is_new = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if is_new:
            writer.writeheader()
        writer.writerow(row_dict)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Full LLVIP HOMO-Feature batch runner")
    parser.add_argument("--split",  default='test', choices=['test', 'train', 'all'],
                        help="Which LLVIP split to process (default: test)")
    parser.add_argument("--resume", action='store_true',
                        help="Skip pairs that already have result.txt in results/llvip/")
    parser.add_argument("--limit",  type=int, default=0,
                        help="Process at most this many pairs (0 = no limit)")
    args = parser.parse_args()

    pairs = collect_pairs(args.split)
    if not pairs:
        print("No pairs found. Check LLVIP directory structure.")
        sys.exit(1)

    # Load already-completed records when resuming
    done_names, records = load_existing_csv(CSV_PATH)

    # Filter pairs
    todo = []
    for vis_path, ir_path, split_name in pairs:
        name = vis_path.stem
        if args.resume and name in done_names:
            continue
        todo.append((vis_path, ir_path, split_name))
        if args.limit and len(todo) >= args.limit:
            break

    total_pairs = len(pairs)
    skipped     = len(done_names) if args.resume else 0
    todo_count  = len(todo)

    print("=" * 60)
    print(f"LLVIP FULL BATCH  (split={args.split})")
    print("=" * 60)
    print(f"  Total pairs in split : {total_pairs}")
    print(f"  Already done (skip)  : {skipped}")
    print(f"  To process now       : {todo_count}")
    if args.limit:
        print(f"  Limit applied        : {args.limit}")

    # Rough ETA based on previous run (~125s/pair)
    eta_h = todo_count * 125 / 3600
    print(f"  ETA (est. ~125s/pair): {eta_h:.1f}h")
    print(f"  CSV log              : {CSV_PATH}")
    print(f"  Results dir          : {RESULTS_DIR}")
    print("=" * 60)

    if todo_count == 0:
        print("Nothing to do. Use --resume=False or delete the CSV to rerun.")
        compute_and_print_summary(records, SUMMARY_PATH)
        return

    batch_start = time.time()
    batch_times = []

    for i, (vis_path, ir_path, split_name) in enumerate(todo, 1):
        name = vis_path.stem
        pair_dir = RESULTS_DIR / name

        print(f"  [{i:>4}/{todo_count}] {name} ...", end='', flush=True)
        t_pair = time.time()

        try:
            ncm, elapsed, cor1_xy, cor2_xy = run_pair(vis_path, ir_path, pair_dir)
        except Exception as e:
            print(f" ERROR: {e}")
            ncm, elapsed, cor1_xy, cor2_xy = 0, time.time() - t_pair, np.empty((0,2)), np.empty((0,2))

        passed = ncm >= 50
        status = 'PASS' if passed else 'FAIL'
        batch_times.append(elapsed)

        print(f" NCM={ncm:>5}  {elapsed:>6.1f}s  {status}")

        # Save visualisations
        try:
            save_overlap(vis_path, ir_path, cor1_xy, cor2_xy,
                         pair_dir / "overlap.png")
            save_mosaic(vis_path, ir_path, cor1_xy, cor2_xy,
                        pair_dir / "mosaic.png")
        except Exception as e:
            print(f"    [WARN] visualisation failed: {e}")

        # Record and append to CSV
        row = {
            'pair_name': name,
            'split':     split_name,
            'NCM':       ncm,
            'time_s':    round(elapsed, 2),
            'status':    status,
        }
        records.append(row)
        append_csv(CSV_PATH, row)

        # Progress report every 10 pairs
        if i % 10 == 0 or i == todo_count:
            elapsed_total = time.time() - batch_start
            mean_t = elapsed_total / i
            remaining = (todo_count - i) * mean_t
            pass_so_far = sum(1 for r in records if r['status'] == 'PASS')
            print(f"\n  --- Progress {i}/{todo_count} "
                  f"| PASS {pass_so_far}/{len(records)} "
                  f"| mean {mean_t:.0f}s/pair "
                  f"| ETA {remaining/3600:.2f}h ---\n")

    # Final summary
    compute_and_print_summary(records, SUMMARY_PATH)


if __name__ == "__main__":
    main()
