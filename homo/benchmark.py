"""
HOMO benchmark runner on the 50-pair LLVIP sample.

Usage
-----
    python homo/benchmark.py

Reads   : results/llvip_sample.csv  (seed-42 pair list; regenerated if missing)
Writes  : results/llvip_python_fixed/<pair_name>_match.jpg  (visualisations)
          results/llvip_python_fixed/benchmark_results.csv   (per-pair NCM / time)
Prints  : per-pair NCM, then summary (mean / min / max / pass-rate)
"""

import csv
import os
import sys
import time

import cv2
import numpy as np

# Make homo/ importable when run from the repo root or from homo/
_here    = os.path.dirname(os.path.abspath(__file__))
_root    = os.path.dirname(_here)
_results = os.path.join(_root, 'results')

if _here not in sys.path:
    sys.path.insert(0, _here)

from homo import run_homo


# =========================================================================== #
# Locate / regenerate the 50-pair sample list                                  #
# =========================================================================== #

_SAMPLE_CSV = os.path.join(_results, 'llvip_sample.csv')
_IR_DIR     = os.path.join(_root, 'LLVIP', 'infrared', 'test')
_VIS_DIR    = os.path.join(_root, 'LLVIP', 'visible',  'test')
_OUT_DIR    = os.path.join(_results, 'llvip_python_fixed')
_OUT_CSV    = os.path.join(_OUT_DIR, 'benchmark_results.csv')


def _load_sample_pairs() -> list[dict]:
    """Return list of {pair_name, split} dicts from the seed-42 sample CSV."""
    if os.path.exists(_SAMPLE_CSV):
        pairs = []
        with open(_SAMPLE_CSV, newline='') as f:
            for row in csv.DictReader(f):
                pairs.append({'pair_name': row['pair_name'],
                               'split':     row.get('split', 'test')})
        return pairs

    # Regenerate with seed=42
    print(f"[benchmark] {_SAMPLE_CSV} not found — regenerating with seed=42")
    all_files = sorted(f for f in os.listdir(_IR_DIR) if f.endswith('.jpg'))
    rng       = np.random.default_rng(42)
    chosen    = rng.choice(all_files, size=min(50, len(all_files)), replace=False)
    pairs     = [{'pair_name': f.replace('.jpg', ''), 'split': 'test'}
                 for f in sorted(chosen)]

    os.makedirs(_results, exist_ok=True)
    with open(_SAMPLE_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['pair_name', 'split'])
        w.writeheader()
        w.writerows(pairs)
    print(f"[benchmark] Saved {len(pairs)} pairs to {_SAMPLE_CSV}")
    return pairs


# =========================================================================== #
# Benchmark runner                                                              #
# =========================================================================== #

def run_benchmark(pass_threshold: int = 50) -> None:
    os.makedirs(_OUT_DIR, exist_ok=True)

    pairs    = _load_sample_pairs()
    n_pairs  = len(pairs)
    results  = []

    print(f"\n{'='*60}")
    print(f"  HOMO Python benchmark — {n_pairs} pairs")
    print(f"  Output dir : {_OUT_DIR}")
    print(f"{'='*60}")
    print(f"  {'Pair':>10}  {'NCM':>6}  {'Time(s)':>8}  Status")
    print(f"  {'-'*10}  {'-'*6}  {'-'*8}  ------")

    total_t0 = time.time()

    for i, p in enumerate(pairs):
        name  = p['pair_name']
        split = p['split']
        ir_path  = os.path.join(_IR_DIR,  f"{name}.jpg")
        vis_path = os.path.join(_VIS_DIR, f"{name}.jpg")

        if not os.path.exists(ir_path) or not os.path.exists(vis_path):
            print(f"  {name:>10}  {'N/A':>6}  {'N/A':>8}  SKIP (file missing)")
            results.append({'pair_name': name, 'split': split,
                            'NCM': 0, 'time_s': 0.0, 'status': 'SKIP'})
            continue

        t0 = time.time()
        try:
            out    = run_homo(ir_path, vis_path)
            ncm    = out['ncm']
            elapsed = round(time.time() - t0, 2)
            status  = 'PASS' if ncm >= pass_threshold else 'FAIL'

            # Save visualisation
            vis_path_out = os.path.join(_OUT_DIR, f"{name}_match.jpg")
            cv2.imwrite(vis_path_out, out['match_img'])

        except Exception as exc:
            elapsed = round(time.time() - t0, 2)
            ncm     = 0
            status  = f'ERROR: {exc}'

        results.append({'pair_name': name, 'split': split,
                        'NCM': ncm, 'time_s': elapsed, 'status': status})
        print(f"  {name:>10}  {ncm:>6}  {elapsed:>8.2f}  {status}")
        sys.stdout.flush()

    total_elapsed = time.time() - total_t0

    # ------------------------------------------------------------------ #
    # Summary                                                              #
    # ------------------------------------------------------------------ #
    valid   = [r for r in results if r['status'] in ('PASS', 'FAIL')]
    ncm_arr = np.array([r['NCM'] for r in valid], dtype=float)
    passes  = sum(r['status'] == 'PASS' for r in valid)

    print(f"\n{'='*60}")
    print(f"  Summary  ({n_pairs} pairs, {total_elapsed/60:.1f} min total)")
    print(f"  Pass rate  : {passes}/{len(valid)} = {100*passes/max(len(valid),1):.1f}%  (NCM >= {pass_threshold})")
    if len(ncm_arr) > 0:
        print(f"  Mean NCM   : {ncm_arr.mean():.1f}")
        print(f"  Median NCM : {np.median(ncm_arr):.1f}")
        print(f"  Min NCM    : {ncm_arr.min():.0f}")
        print(f"  Max NCM    : {ncm_arr.max():.0f}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------ #
    # Save per-pair CSV                                                    #
    # ------------------------------------------------------------------ #
    with open(_OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['pair_name', 'split', 'NCM', 'time_s', 'status'])
        w.writeheader()
        w.writerows(results)
    print(f"  Results saved to {_OUT_CSV}")


if __name__ == '__main__':
    run_benchmark()
