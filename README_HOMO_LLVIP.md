# HOMO-Feature: VIS-IR Matching on LLVIP — Team Reproduction Guide

---

## What This Is

**HOMO-Feature** is a classical (non-deep-learning) keypoint matching algorithm designed for
cross-modal image pairs — for example, a visible-light photo and an infrared thermal image of
the same scene. Unlike SIFT or ORB, it does not rely on raw pixel intensity, so it works even
when the two images look completely different in brightness and texture. It was published at
ICCV 2025 by Gao et al. We ran it to evaluate how well it matches visible and infrared images
across a range of real outdoor scenes.

**LLVIP** (Low-Light Visible-Infrared Paired dataset) is a benchmark of 15,488 co-registered
visible/infrared image pairs captured at night from a fixed dual-camera rig on a street. It is
a demanding test for cross-modal matching because the visible channel is dim and noisy while
the infrared channel shows thermal emission — two very different representations of the same
scene. The precise hardware alignment means we can treat every pixel correspondence as
ground-truth for evaluation.

You can follow this guide to reproduce our 50-pair sample experiment from scratch — clone the
code, download the data, run the matching, and get the same summary statistics we report.

---

## What You Need Before Starting

| Requirement | Detail |
| --- | --- |
| **Python** | 3.11 or newer (we used 3.13.5) |
| **MATLAB** | Not required — we use the Python reimplementation provided in this repo |
| **Git** | Any recent version, for cloning the HOMO repo |
| **Disk space** | HOMO repo ~43 MB · LLVIP full dataset ~3.8 GB · 50-pair results ~45 MB |
| **RAM** | 4 GB minimum; 8 GB recommended |
| **Time** | ~105 minutes for the 50-pair HOMO sample on a 4-core laptop CPU |

> **Python packages needed** (includes matplotlib for plots):
>
> ```bash
> pip install numpy scipy scikit-image opencv-python matplotlib
> ```

---

## Folder Structure

After following all setup steps, your workspace should look like this:

```text
GroupAssignment/                       <- workspace root (this folder)
  HOMO_Feature_ImgMatching/            <- cloned HOMO repo
    HOMO_image_matching_demo/
    misc/
      display1.jpg                     <- smoke-test image pair
      display2.jpg
    README.md
  LLVIP/
    visible/
      test/                            <- 3 463 visible test images  (19xxxx–26xxxx)
      train/                           <- 12 025 visible train images (01xxxx)
    infrared/
      test/                            <- matching infrared test images
      train/
  results/
    llvip_sample/                      <- per-pair HOMO output folders
      190222/
        result.txt
        overlap.png
        mosaic.png
      ...
    sift_sample/                       <- per-pair SIFT output (baseline)
    llvip_sample.csv                   <- HOMO results table (50 pairs)
    sift_sample.csv                    <- SIFT baseline results (same 50 pairs)
    comparison.csv                     <- HOMO vs SIFT side-by-side
    llvip_sample_summary.txt           <- HOMO summary statistics
    ncm_distribution.png               <- NCM histogram: HOMO vs SIFT
    ncm_per_pair.png                   <- per-pair bar chart
    HOMO_LLVIP_report.md               <- full analysis report
  homo_feature.py                      <- Python reimplementation of HOMO
  run_homo.py                          <- smoke test runner
  run_llvip_sample.py                  <- 50-pair HOMO sample runner
  run_sift_sample.py                   <- 50-pair SIFT baseline runner
  make_plots.py                        <- generate comparison plots
  README_HOMO_LLVIP.md                 <- this file
```

---

## Step-by-Step Setup

### Step 1 — Get the HOMO Code

Clone the original MATLAB repository (we need its demo images and reference):

```bash
git clone https://github.com/MrPingQi/HOMO_Feature_ImgMatching.git
```

This creates the `HOMO_Feature_ImgMatching/` folder in your workspace.
The two images inside `misc/` (`display1.jpg` and `display2.jpg`) are used for the smoke test.

> **Note:** The original HOMO code is in MATLAB. Because MATLAB may not be available to
> everyone, this repo includes `homo_feature.py` — a complete Python reimplementation
> translated line-by-line from the MATLAB source. You do not need MATLAB at all.

---

### Step 2 — Get the LLVIP Dataset

1. Go to the LLVIP project page and register to download:
   `https://bupt-ai-cz.github.io/LLVIP/`

2. Download the full dataset archive (look for the link labelled
   **"Download (Baidu / Google Drive)"** on that page). The file is roughly **3.8 GB**.

3. Extract the archive so that the folder structure matches exactly:

   ```text
   LLVIP/
     visible/
       test/        <- .jpg files named like 190001.jpg, 190002.jpg, ...
       train/       <- .jpg files named like 010001.jpg, 010002.jpg, ...
     infrared/
       test/        <- same filenames as visible/test/
       train/       <- same filenames as visible/train/
   ```

4. Verify it looks right:

   ```bash
   ls LLVIP/visible/test | head -5
   ls LLVIP/infrared/test | head -5
   ```

   You should see matching filenames (e.g. `190001.jpg`) in both folders.
   The test split has **3,463 pairs**; the train split has **12,025 pairs**.

> **LLVIP naming convention:** Test images start with `19`–`26` (e.g. `190001.jpg`).
> Train images start with `01` (e.g. `010001.jpg`). All 50 pairs in our sample come from
> the **test split**.

---

### Step 3 — Create the save\_image Folder

The HOMO demo script expects this output folder to exist before running:

```bash
mkdir -p HOMO_Feature_ImgMatching/HOMO_image_matching_demo/save_image
```

Without this, the demo will crash when it tries to save result images.

---

### Step 4 — Run the Smoke Test

The smoke test runs the algorithm on the two built-in display images to confirm everything
is working before you spend two hours on the full sample.

```bash
python run_homo.py
```

**What to look for in the output:**

```text
NCM (Number of Correct Matches): 136
Smoke test PASS (NCM >= 50):      YES
```

- **NCM** (Number of Correct Matches) is how many feature pairs survived geometric
  verification (RANSAC). Higher = better.
- **PASS** means NCM >= 50, which is the threshold used throughout this project.
- A result of NCM = 0 or FAIL on the smoke test means something is broken —
  see the Common Issues section below before continuing.

A successful run also writes `smoke_test.log` in the workspace root.

---

### Step 5 — Run on LLVIP (50-pair sample)

We use a **fixed random sample of 50 pairs** (seed = 42) from the LLVIP test split.
Using a fixed seed means every teammate gets exactly the same 50 pairs, so results
are directly comparable.

```bash
python run_llvip_sample.py
```

That's it — no extra arguments needed. The script will:

1. Randomly select 50 pairs from `LLVIP/visible/test/` using seed 42
2. Run HOMO matching on each visible/infrared pair
3. Print progress every 5 pairs with a live ETA
4. Save outputs to `results/llvip_sample/<pair_name>/`
5. Append each result to `results/llvip_sample.csv` as it completes
6. Print and save the full summary when done

**How long it takes:** approximately **105 minutes** on a 4-core laptop CPU
(~124 seconds per pair). The CSV is written after every pair, so if you need to stop
and restart, just re-run the same command — it will overwrite from scratch (the run is
fast enough to not need a resume feature for 50 pairs).

**Optional flags:**

```bash
python run_llvip_sample.py --n 10        # run only 10 pairs (quick check)
python run_llvip_sample.py --split train # sample from train split instead
python run_llvip_sample.py --seed 123    # use a different random seed
```

---

### Step 6 — Check Your Results

**Read the CSV summary:**

```bash
cat results/llvip_sample.csv
```

Each row looks like:

```text
pair_name, split, NCM, time_s, status
190222, test, 1641, 125.53, PASS
```

**Read the summary statistics:**

```bash
cat results/llvip_sample_summary.txt
```

**Visually verify a match** — open any `overlap.png` to see the two images
side-by-side with coloured match lines drawn between them:

```bash
# Windows
start results/llvip_sample/190222/overlap.png

# Mac
open results/llvip_sample/190222/overlap.png

# Linux
xdg-open results/llvip_sample/190222/overlap.png
```

- `overlap.png` — side-by-side image with up to 200 random match lines
- `mosaic.png` — the infrared image warped onto the visible coordinate system
  and alpha-blended, showing how well they align

A good result looks like lines connecting the same physical points (road markings,
building corners, lamp posts) in both images.

---

## Our Results (for reference)

**LLVIP split:** All 50 sampled pairs come from the **test split** (filenames 19xxxx–26xxxx).
Zero pairs are from the train split.

**HOMO results:**

```text
============================================================
LLVIP 50-PAIR SAMPLE SUMMARY (seed=42)
============================================================
Total pairs sampled   : 50
PASS (NCM >= 50)      : 50
FAIL (NCM <  50)      : 0
Success rate          : 100.0%
------------------------------------------------------------
Mean NCM (PASS only)  : 1548.0
Mean NCM (all pairs)  : 1548.0
Min NCM               : 864
Max NCM               : 2767
Mean time per pair    : 123.6s
Total wall time       : 103.0min
------------------------------------------------------------
All sampled pairs PASSED.
============================================================
```

**SIFT baseline (same 50 pairs, for comparison):**

```text
PASS (NCM >= 50)      : 0 / 50   (0.0% success rate)
Mean NCM (all pairs)  : 6.0
Mean time per pair    : 0.2s
```

SIFT fails on every single pair — this is expected. SIFT's gradient-based descriptors
are polarity-sensitive: in infrared images edges can appear reversed relative to visible,
so SIFT's orientation histograms point in opposite directions and produce no valid matches.
HOMO's structure-based design (Phase Congruency + MOM) is specifically engineered to be
polarity-invariant, which is why it achieves 1548 mean NCM where SIFT gets 6 — a 353x
improvement on the exact same pairs.

See `results/ncm_distribution.png` for a histogram of both distributions, and
`results/ncm_per_pair.png` for a per-pair breakdown sorted by HOMO NCM.

If you use `--seed 42` (the default), you will get **exactly these 50 pairs** and
your HOMO NCM numbers should be very close (within ±5%) to the values above.

---

## Reproducing the Baseline Comparison

**Run SIFT on the same 50 pairs** (takes ~10 seconds total):

```bash
python run_sift_sample.py
```

This reads `results/llvip_sample.csv` to get the exact same 50 pairs in the same order,
runs OpenCV SIFT with BFMatcher + ratio test 0.8 + RANSAC (error=5 px), and saves
`results/sift_sample.csv`.

**Regenerate the comparison CSV and plots:**

```bash
python -c "
import csv
from pathlib import Path
R = Path('results')
homo = {r['pair_name']: r for r in csv.DictReader(open(R/'llvip_sample.csv'))}
sift = {r['pair_name']: r for r in csv.DictReader(open(R/'sift_sample.csv'))}
header = ['pair_name','HOMO_NCM','SIFT_NCM','HOMO_time','SIFT_time','HOMO_status','SIFT_status']
rows = [{'pair_name': n, 'HOMO_NCM': int(homo[n]['NCM']), 'SIFT_NCM': int(sift[n]['NCM']),
         'HOMO_time': homo[n]['time_s'], 'SIFT_time': sift[n]['time_s'],
         'HOMO_status': homo[n]['status'], 'SIFT_status': sift[n]['status']} for n in homo]
with open(R/'comparison.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=header); w.writeheader(); w.writerows(rows)
print('comparison.csv written')
"

python make_plots.py
```

`comparison.csv` has all raw numbers if you want to do your own analysis in Excel,
pandas, or R. The columns are: `pair_name, HOMO_NCM, SIFT_NCM, HOMO_time, SIFT_time,
HOMO_status, SIFT_status`.

---

## Understanding the Output Files

| File | What it shows |
| --- | --- |
| `result.txt` | Plain text: pair name, file paths, NCM, time, PASS/NO |
| `overlap.png` | The two images side-by-side with coloured match lines |
| `mosaic.png` | IR warped onto visible using the found transform, alpha-blended |
| `llvip_sample.csv` | One row per pair: name, split, NCM, time, status (HOMO) |
| `sift_sample.csv` | SIFT baseline results on the same 50 pairs |
| `comparison.csv` | Side-by-side HOMO vs SIFT NCM, time, and status per pair |
| `llvip_sample_summary.txt` | Aggregate stats: total, PASS/FAIL, mean/min/max NCM |
| `ncm_distribution.png` | Histogram of NCM scores for both methods (HOMO blue, SIFT orange) |
| `ncm_per_pair.png` | Per-pair bar chart sorted by HOMO NCM descending |
| `HOMO_LLVIP_report.md` | Full analysis including comparison to paper figures |
| `smoke_test.log` | Detailed log from the smoke test run |

> **What about `correspondences.mat`?** The original MATLAB demo saves matched
> coordinate pairs as a `.mat` file. The Python reimplementation saves them inside
> `result.txt` and exposes them via the `homo_match()` return value instead.

---

## Common Issues and Fixes

**`python: command not found` or wrong Python version**

Make sure Python 3.11+ is on your PATH. On Windows you may need to use `python3`
or the full path:

```bash
# Check your version
python --version

# If that shows 3.8 or lower, try:
python3 run_homo.py
```

**Missing packages (`ModuleNotFoundError: No module named 'cv2'` etc.)**

```bash
pip install numpy scipy scikit-image opencv-python matplotlib
```

If `pip` installs to the wrong Python, use:

```bash
python -m pip install numpy scipy scikit-image opencv-python matplotlib
```

**`save_image` folder missing (MATLAB demo crash)**

Only relevant if you try to run the original MATLAB demo directly.
Fix:

```bash
mkdir -p HOMO_Feature_ImgMatching/HOMO_image_matching_demo/save_image
```

**NCM = 0 for a pair**

This is not always a bug. It can happen on genuinely hard pairs where the visible
and infrared images look very different (e.g. very dark scene, mostly uniform sky).
As long as the majority of pairs pass (SR >= 90%), the algorithm is working correctly.
If every single pair returns NCM = 0, re-run the smoke test first to verify the
basic pipeline.

**LLVIP folder structure wrong**

The filenames in `visible/test/` and `infrared/test/` must match exactly
(e.g. `190001.jpg` in both). Check with:

```bash
ls LLVIP/visible/test | sort | head -5
ls LLVIP/infrared/test | sort | head -5
```

If the names differ, the script silently skips the pair.

### SIFT gives 0 matches on VIS-IR pairs

This is completely expected and is not a bug. SIFT is an intensity-based algorithm that
was never designed for cross-modal matching. In infrared images, edges often appear with
reversed contrast compared to visible — a warm object appears bright in IR but dark in
visible — so SIFT's gradient descriptors point in opposite directions and find no
correspondences. A 0% SIFT success rate on VIS-IR is the norm; this is precisely the
problem HOMO was designed to solve.

**Unicode error on Windows console**

If you see `UnicodeEncodeError` with a checkmark or arrow character, your Windows
console is using an older codepage. Fix:

```cmd
chcp 65001
```

Or just run from VS Code's integrated terminal, which handles UTF-8 by default.

---

## Key Parameters (for the curious)

These are the values used in our run, set in `run_llvip_sample.py`:

| Parameter | Value used | What it controls |
| --- | --- | --- |
| `key_type` | `PC-ShiTomasi` | Keypoint detector: Phase Congruency + Shi-Tomasi corner response |
| `Npoint` | 5000 | Maximum number of keypoints detected per image |
| `nOctaves` | 3 | Number of pyramid levels (scales) to build |
| `nLayers` | 2 | Number of Gaussian blur layers per octave |
| `patch_size` | 72 | Size (in pixels) of the descriptor patch around each keypoint |
| `NBA` | 12 | Angular bins in the polar descriptor grid |
| `NBO` | 12 | Orientation bins in the descriptor histogram |
| `error` | 10 | RANSAC inlier threshold in pixels (reprojection error tolerance) |
| `K` | 5 | Number of times RANSAC is repeated (takes the best result) |
| `trans_form` | `affine` | Geometric model assumed between images |
| `int_flag` | 1 | Use unit magnitude weights (more robust across modalities) |
| `rot_flag` | True | Apply rotation-invariant descriptor orientation assignment |

> **Why error=10?** The images are downscaled to max 512 pixels, so 10 px is about 2%
> of the image width. This is a reasonable tolerance for cross-modal pairs where the
> geometric alignment is not sub-pixel perfect after downscaling.

---

## Citation

If you use HOMO-Feature in your work, please cite the original paper:

```bibtex
@InProceedings{Gao_2025_ICCV,
    author    = {Gao, Chenzhong and Li, Wei and Weng, Desheng},
    title     = {{HOMO-Feature}: Cross-Arbitrary-Modal Image Matching with Homomorphism of Organized Major Orientation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    pages     = {10538-10548},
    year      = {2025}
}
```

Paper link: https://openaccess.thecvf.com/content/ICCV2025/html/Gao_HOMO-Feature_Cross-Arbitrary-Modal_Image_Matching_with_Homomorphism_of_Organized_Major_Orientation_ICCV_2025_paper.html

LLVIP dataset: https://bupt-ai-cz.github.io/LLVIP/
