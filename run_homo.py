"""
HOMO-Feature runner script.
Runs the smoke test (display1 vs display2) and LLVIP batch (first 5 pairs).

Usage:
    python run_homo.py                    # smoke test only
    python run_homo.py --llvip            # smoke + LLVIP
    python run_homo.py --llvip --n 5      # choose number of LLVIP pairs (default 5)
"""

import sys
import os
import time
import argparse
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).parent
HOMO_MISC  = WORKSPACE / "HOMO_Feature_ImgMatching" / "misc"
LLVIP_VIS  = WORKSPACE / "LLVIP" / "visible" / "test"
LLVIP_IR   = WORKSPACE / "LLVIP" / "infrared" / "test"
LOG_FILE   = WORKSPACE / "smoke_test.log"
RESULTS_DIR = WORKSPACE / "results" / "llvip"

# ── Import core algorithm ────────────────────────────────────────────────────
sys.path.insert(0, str(WORKSPACE))
from homo_feature import homo_match


# ─────────────────────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

def run_smoke_test():
    """Run HOMO on the two built-in display images and log results."""
    img1 = HOMO_MISC / "display1.jpg"
    img2 = HOMO_MISC / "display2.jpg"

    print("=" * 60)
    print("SMOKE TEST: display1.jpg vs display2.jpg")
    print("=" * 60)

    log_lines = []
    log_lines.append("HOMO-Feature Smoke Test")
    log_lines.append(f"Image 1: {img1}")
    log_lines.append(f"Image 2: {img2}")
    log_lines.append("-" * 60)

    class Tee:
        """Mirrors stdout to a list for logging."""
        def __init__(self, lines):
            self.lines = lines
        def write(self, msg):
            sys.__stdout__.write(msg)
            if msg.strip():
                self.lines.append(msg.rstrip())
        def flush(self):
            sys.__stdout__.flush()

    tee = Tee(log_lines)
    old_stdout = sys.stdout
    sys.stdout = tee

    result = homo_match(
        str(img1), str(img2),
        int_flag=1, rot_flag=True, scl_flag=False,
        nOctaves=3, nLayers=2, G_resize=1.2, G_sigma=1.6,
        key_type='PC-ShiTomasi', thresh=0, radius=1, Npoint=5000,
        patch_size=72, NBA=12, NBO=12, error=10, K=5,
        trans_form='affine', verbose=True
    )

    sys.stdout = old_stdout

    ncm   = result['ncm']
    t_tot = result['time_total']
    passed = ncm >= 50

    summary = [
        "-" * 60,
        f"NCM (Number of Correct Matches): {ncm}",
        f"Total time:                       {t_tot:.2f}s",
        f"Smoke test PASS (NCM >= 50):      {'YES' if passed else 'NO'}",
    ]
    for line in summary:
        print(line)
        log_lines.append(line)

    LOG_FILE.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"\nFull log saved: {LOG_FILE}")

    return result, passed


# ─────────────────────────────────────────────────────────────────────────────
# LLVIP BATCH
# ─────────────────────────────────────────────────────────────────────────────

def run_llvip_batch(n_pairs=5):
    """Run HOMO on the first n_pairs visible/infrared pairs from LLVIP."""
    if not LLVIP_VIS.exists() or not LLVIP_IR.exists():
        print(f"\nLLVIP not found at {WORKSPACE / 'LLVIP'}. Skipping LLVIP batch.")
        return []

    vis_imgs = sorted(LLVIP_VIS.glob("*.jpg"))[:n_pairs]
    if not vis_imgs:
        print("No LLVIP images found.")
        return []

    print("\n" + "=" * 60)
    print(f"LLVIP BATCH: first {len(vis_imgs)} visible/infrared pairs")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    for vis_path in vis_imgs:
        name = vis_path.stem
        ir_path = LLVIP_IR / vis_path.name
        if not ir_path.exists():
            print(f"  [SKIP] infrared counterpart not found: {ir_path}")
            continue

        pair_dir = RESULTS_DIR / name
        pair_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  Pair: {name}")
        t0 = time.time()
        result = homo_match(
            str(vis_path), str(ir_path),
            int_flag=1, rot_flag=True, scl_flag=False,
            nOctaves=3, nLayers=2, G_resize=1.2, G_sigma=1.6,
            key_type='PC-ShiTomasi', thresh=0, radius=1, Npoint=5000,
            patch_size=72, NBA=12, NBO=12, error=10, K=5,
            trans_form='affine', verbose=True
        )
        elapsed = time.time() - t0
        ncm    = result['ncm']
        passed = ncm >= 50

        # Save per-pair log
        pair_log = pair_dir / "result.txt"
        pair_log.write_text(
            f"Pair:   {name}\n"
            f"Vis:    {vis_path}\n"
            f"IR:     {ir_path}\n"
            f"NCM:    {ncm}\n"
            f"Time:   {elapsed:.2f}s\n"
            f"PASS:   {'YES' if passed else 'NO'}\n",
            encoding="utf-8"
        )

        records.append({
            'name':   name,
            'ncm':    ncm,
            'time':   elapsed,
            'passed': passed,
        })

    return records


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(smoke_result, smoke_passed, llvip_records):
    """Print the final summary table for all pairs run."""
    print("\n" + "=" * 60)
    print("FINAL SUMMARY TABLE")
    print("=" * 60)
    header = f"{'Filename':<22} {'NCM':>6} {'Time(s)':>9} {'Result':>8}"
    print(header)
    print("-" * 60)

    # Smoke test row
    ncm  = smoke_result['ncm']
    t    = smoke_result['time_total']
    stat = "PASS" if smoke_passed else "FAIL"
    print(f"{'display1+2 (smoke)':<22} {ncm:>6} {t:>9.2f} {stat:>8}")

    # LLVIP rows
    for rec in llvip_records:
        stat = "PASS" if rec['passed'] else "FAIL"
        print(f"{rec['name']:<22} {rec['ncm']:>6} {rec['time']:>9.2f} {stat:>8}")

    print("-" * 60)
    all_ncms = [smoke_result['ncm']] + [r['ncm'] for r in llvip_records]
    all_pass = [smoke_passed]       + [r['passed'] for r in llvip_records]
    print(f"{'Total pairs run:':<22} {len(all_ncms):>6}")
    print(f"{'Passed (NCM>=50):':<22} {sum(all_pass):>6}")
    print(f"{'Mean NCM:':<22} {sum(all_ncms)/len(all_ncms):>6.1f}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HOMO-Feature matching runner")
    parser.add_argument("--llvip", action="store_true",
                        help="Also run the LLVIP batch after the smoke test")
    parser.add_argument("--n", type=int, default=5,
                        help="Number of LLVIP pairs to run (default: 5)")
    args = parser.parse_args()

    # Always run smoke test
    smoke_result, smoke_passed = run_smoke_test()

    llvip_records = []
    if args.llvip:
        llvip_records = run_llvip_batch(n_pairs=args.n)

    print_summary(smoke_result, smoke_passed, llvip_records)

    sys.exit(0 if smoke_passed else 1)


if __name__ == "__main__":
    main()
