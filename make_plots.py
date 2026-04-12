"""Generate ncm_distribution.png and ncm_per_pair.png."""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS = Path('c:/Users/ELITEBOOK/Desktop/Computer Vision/GroupAssignment/results')

rows = list(csv.DictReader(open(RESULTS / 'comparison.csv')))
pairs      = [r['pair_name'] for r in rows]
homo_ncms  = [int(r['HOMO_NCM']) for r in rows]
sift_ncms  = [int(r['SIFT_NCM']) for r in rows]
homo_sr    = 100.0   # pre-computed
sift_sr    = 0.0

# ── Plot 1: overlapping histograms ──────────────────────────────────────────
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(10, 6))

max_ncm = max(max(homo_ncms), max(sift_ncms))
bins = np.linspace(0, max_ncm + 50, 21)

ax.hist(homo_ncms, bins=bins, color='steelblue', alpha=0.7,
        label=f'HOMO  mean={np.mean(homo_ncms):.0f}  SR={homo_sr:.0f}%', zorder=3)
ax.hist(sift_ncms, bins=bins, color='darkorange', alpha=0.7,
        label=f'SIFT  mean={np.mean(sift_ncms):.0f}  SR={sift_sr:.0f}%', zorder=3)

# PASS threshold
ax.axvline(50, color='crimson', linestyle='--', linewidth=1.8,
           label='PASS threshold (NCM=50)', zorder=4)

# Mean lines
ax.axvline(np.mean(homo_ncms), color='steelblue', linestyle='--',
           linewidth=1.4, alpha=0.9, zorder=4)
ax.axvline(np.mean(sift_ncms), color='darkorange', linestyle='--',
           linewidth=1.4, alpha=0.9, zorder=4)

ax.set_title('NCM Distribution — HOMO vs SIFT on LLVIP (n=50, seed=42)',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Number of Correct Matches (NCM)', fontsize=12)
ax.set_ylabel('Number of Pairs', fontsize=12)
ax.legend(fontsize=11)
ax.yaxis.get_major_locator().set_params(integer=True)
fig.tight_layout()
fig.savefig(RESULTS / 'ncm_distribution.png', dpi=300)
plt.close(fig)
print('Saved ncm_distribution.png')

# ── Plot 2: per-pair horizontal bar chart ───────────────────────────────────
# Sort by HOMO NCM descending
order = np.argsort(homo_ncms)[::-1]
pairs_s     = [pairs[i] for i in order]
homo_ncms_s = [homo_ncms[i] for i in order]
sift_ncms_s = [sift_ncms[i] for i in order]

n = len(pairs_s)
y = np.arange(n)
bar_h = 0.38

fig, ax = plt.subplots(figsize=(12, max(8, n * 0.28)))
ax.barh(y + bar_h/2, homo_ncms_s, height=bar_h,
        color='steelblue', alpha=0.85, label='HOMO', zorder=3)
ax.barh(y - bar_h/2, sift_ncms_s, height=bar_h,
        color='darkorange', alpha=0.85, label='SIFT', zorder=3)

ax.axvline(50, color='crimson', linestyle='--', linewidth=1.6,
           label='PASS threshold (NCM=50)', zorder=4)

ax.set_yticks(y)
ax.set_yticklabels(pairs_s, fontsize=7)
ax.invert_yaxis()
ax.set_xlabel('Number of Correct Matches (NCM)', fontsize=11)
ax.set_title('Per-Pair NCM — HOMO vs SIFT on LLVIP (n=50, seed=42)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
fig.tight_layout()
fig.savefig(RESULTS / 'ncm_per_pair.png', dpi=300)
plt.close(fig)
print('Saved ncm_per_pair.png')
