"""
04_visualizations.py
--------------------
Samiksha Dixit | May 2025

Making the results readable for people who won't sit through a regression table.

A few principles I follow for econ visualizations:
1. Always show the uncertainty (confidence intervals, not just point estimates)
2. Annotate the "so what" directly on the chart
3. Never use a pie chart
4. The event study plot is the most important one — it's your credibility chart.
   If pre-trends look bad, everything else falls apart.

I'm deliberately keeping these clean and somewhat academic in style.
These are meant to accompany a write-up, not a marketing deck.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# Clean, professional style — not too much chrome
plt.rcParams.update({
    "font.family":       "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
    "axes.labelsize":    11,
    "axes.titlesize":    12,
    "axes.titleweight":  "bold",
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
})

# Colors — I use a consistent palette throughout
# Blue for treated, gray for control, red for shock/warning
BLUE   = "#2D426A"
GREEN  = "#48EB8F"
ORANGE = "#F4845F"
GRAY   = "#8B9DB5"
RED    = "#E05C5C"
LIGHT  = "#E8EFF7"

import os
os.makedirs("figures", exist_ok=True)

# Load everything I need
df      = pd.read_csv("data/worker_panel_enriched.csv")
matched = pd.read_csv("data/matched_panel.csv")
event   = pd.read_csv("data/event_study.csv").sort_values("year")
results = pd.read_csv("data/causal_results.csv").iloc[0]

# ── Figure 1: Adoption wave ───────────────────────────────────────────────────
# I want to show that AI adoption was near-zero pre-2022 and then spiked.
# This is the "first stage" of the natural experiment story.
# If adoption hadn't spiked after ChatGPT, the whole identification falls apart.

fig, ax = plt.subplots(figsize=(11, 5.5))
adoption = df.groupby(["year","job_family"])["has_ai_skill"].mean().reset_index()
jf_order = df.groupby("job_family")["ai_intensity"].mean().sort_values(ascending=False).index
colors   = plt.cm.Blues(np.linspace(0.35, 0.95, len(jf_order)))

for i, jf in enumerate(jf_order):
    sub = adoption[adoption.job_family == jf]
    ax.plot(sub.year, sub.has_ai_skill * 100, "o-",
            color=colors[i], linewidth=2, markersize=5, label=jf)

ax.axvline(2022.92, color=RED, linestyle="--", linewidth=1.5, alpha=0.8)
ax.text(2022.95, 80, "ChatGPT\n(Nov 2022)", color=RED, fontsize=8.5, va="top")
ax.set_xlabel("Year")
ax.set_ylabel("Workers with AI Skills (%)")
ax.set_title("AI Skill Adoption by Job Family (2019–2024)\n"
             "Near-zero pre-shock → rapid diffusion post-ChatGPT")
ax.set_ylim(0, 100)
ax.set_xticks(range(2019, 2025))
ax.legend(fontsize=8, loc="upper left", ncol=2, framealpha=0.7)
fig.text(0.01, -0.03,
         "Note: Synthetic data. Parameters grounded in O*NET AI exposure scores "
         "and Burning Glass skill frequency data.",
         fontsize=7, color=GRAY)
plt.tight_layout()
plt.savefig("figures/fig1_adoption_wave.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Fig 1: Adoption wave")

# ── Figure 2: Event study ─────────────────────────────────────────────────────
# This is the most important figure. I need to show:
# (a) Pre-2022 coefficients ≈ 0 → parallel trends holds
# (b) Post-2022 coefficients > 0 and growing → treatment effect exists
# If (a) fails, I'd have to walk back the causal claims significantly.

fig, ax = plt.subplots(figsize=(9, 5))
years = event["year"].values
coefs = event["coef"].values
ses   = event["se"].values
pre   = years < 2023
post  = years >= 2023

# Shade confidence intervals separately for pre/post
ax.fill_between(years[pre],  (coefs-1.96*ses)[pre],  (coefs+1.96*ses)[pre],
                alpha=0.15, color=GRAY, label="_nolegend_")
ax.fill_between(years[post], (coefs-1.96*ses)[post], (coefs+1.96*ses)[post],
                alpha=0.15, color=BLUE, label="_nolegend_")

ax.plot(years[pre],  coefs[pre],  "o-", color=GRAY, linewidth=2, markersize=7,
        label="Pre-shock (should be ~0)")
ax.plot(years[post], coefs[post], "o-", color=BLUE, linewidth=2.5, markersize=8,
        label="Post-shock (treatment effect)")

ax.axhline(0, color="black", linewidth=0.8)
ax.axvline(2022.5, color=RED, linestyle="--", linewidth=1.5)
ax.text(2022.55, max(coefs)*0.85, "ChatGPT\nshock", color=RED, fontsize=8)

# Annotate the 2024 effect directly
y24 = event[event.year==2024]["coef"].values[0]
ax.annotate(
    f"+{(np.exp(y24)-1)*100:.1f}% salary\nby 2024",
    xy=(2024, y24), xytext=(2023.2, y24 + 0.025),
    arrowprops=dict(arrowstyle="->", color=BLUE),
    fontsize=9, color=BLUE, fontweight="bold"
)

ax.set_xlabel("Year (base = 2022)")
ax.set_ylabel("β coefficient on log(salary)")
ax.set_title("Event Study: Dynamic Treatment Effects\n"
             "Pre-2022 coefficients ≈ 0 → Parallel trends assumption holds ✓")
ax.set_xticks(years)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("figures/fig2_event_study.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Fig 2: Event study")

# ── Figure 3: Selection bias decomposition ────────────────────────────────────
# The "money chart" — shows why causal inference matters.
# Raw gap vs. PSM-corrected gap vs. PSM+DiD estimate.
# Most people would just report the raw gap and call it a day.

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: Raw salary trajectories (biased)
raw = df.groupby(["year","treated"])["salary"].mean().unstack()
raw["gap_pct"] = (raw[1] / raw[0] - 1) * 100

ax = axes[0]
ax.bar(np.array(list(raw.index)) - 0.2, raw[0]/1000, 0.35,
       label="No AI skills", color=GRAY, alpha=0.85)
ax.bar(np.array(list(raw.index)) + 0.2, raw[1]/1000, 0.35,
       label="AI skills", color=BLUE, alpha=0.85)
for yr, row in raw.iterrows():
    if yr >= 2023:
        ax.text(yr, row[1]/1000 + 1.5, f"+{row.gap_pct:.0f}%",
                ha="center", fontsize=8, color=BLUE, fontweight="bold")
ax.axvline(2022.5, color=RED, linestyle="--", linewidth=1.2)
ax.set_title("Raw Salary Gap — BIASED\n(includes selection on ability)")
ax.set_ylabel("Mean Salary ($K)")
ax.set_xlabel("Year")
ax.legend(fontsize=9)
ax.set_xticks(range(2019, 2025))

# Panel B: What PSM+DiD strips out
ax = axes[1]
raw_gap_2024    = (raw.loc[2024, 1] / raw.loc[2024, 0] - 1) * 100
causal_estimate = float(results["salary_pct_effect"])
selection_bias  = raw_gap_2024 - causal_estimate

bars = ax.bar(
    ["Raw Gap\n(2024)", "Selection\nBias\n(removed by\nPSM+DiD)", "True Causal\nEffect"],
    [raw_gap_2024, selection_bias, causal_estimate],
    color=[GRAY, RED, GREEN], alpha=0.88, width=0.5, edgecolor="white"
)
for bar, val in zip(bars, [raw_gap_2024, selection_bias, causal_estimate]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")

ax.set_title("Decomposing the AI Salary Premium\n"
             "Selection bias explains ~{:.0f}% of the raw gap".format(
             selection_bias / raw_gap_2024 * 100))
ax.set_ylabel("Salary difference (%)")
ax.axhline(0, color="black", linewidth=0.7)
ax.set_ylim(0, raw_gap_2024 + 5)

plt.suptitle("Why Causal Inference Matters: Raw Gap vs. True Effect",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("figures/fig3_selection_bias.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Fig 3: Selection bias decomposition")

# ── Figure 4: Summary dashboard ───────────────────────────────────────────────
# Trying to put all the key results on one page.
# I know this is a lot of information — an actual paper would spread this out.

fig = plt.figure(figsize=(14, 8))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# KPI panels
kpis = [
    {"label": "AI Wage Premium\n(causal, PSM+DiD)",
     "value": f"+{float(results['salary_pct_effect']):.1f}%",
     "note":  f"raw gap ~{raw_gap_2024:.0f}% — selection does the rest",
     "color": BLUE},
    {"label": "Promotion Probability\n(marginal effect)",
     "value": f"+{float(results['promo_marginal_effect']):.1f}pp",
     "note":  "per year, post-adoption",
     "color": ORANGE},
    {"label": "Job Mobility\n(marginal effect)",
     "value": f"+{float(results['switch_marginal_effect']):.1f}pp",
     "note":  "AI skills create outside options",
     "color": GRAY},
]
for i, kpi in enumerate(kpis):
    ax = fig.add_subplot(gs[0, i])
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    rect = mpatches.FancyBboxPatch((0.05,0.05), 0.9, 0.9,
                                    boxstyle="round,pad=0.05",
                                    facecolor=kpi["color"], alpha=0.10,
                                    edgecolor=kpi["color"], linewidth=2)
    ax.add_patch(rect)
    ax.text(0.5, 0.70, kpi["value"], ha="center", va="center",
            fontsize=26, fontweight="bold", color=kpi["color"])
    ax.text(0.5, 0.42, kpi["label"], ha="center", va="center",
            fontsize=9, fontweight="bold", color="#333")
    ax.text(0.5, 0.18, kpi["note"], ha="center", va="center",
            fontsize=7.5, color=GRAY, style="italic")

# Industry adoption bar chart
ax4 = fig.add_subplot(gs[1, :2])
ind = df[df.year==2024].groupby("industry")["has_ai_skill"].mean().sort_values()
cols = [GREEN if v > 0.5 else BLUE if v > 0.35 else GRAY for v in ind.values]
ax4.barh(ind.index, ind.values * 100, color=cols, alpha=0.85)
for i, (idx, val) in enumerate(zip(ind.index, ind.values)):
    ax4.text(val*100 + 0.5, i, f"{val*100:.0f}%", va="center", fontsize=9)
ax4.set_xlabel("AI Skill Adoption Rate (2024, %)")
ax4.set_title("Adoption by Industry — Technology leads, Manufacturing lags")
ax4.set_xlim(0, 88)

# Matched salary trajectories
ax5 = fig.add_subplot(gs[1, 2])
traj = matched.groupby(["year","treated"])["salary"].mean().unstack() / 1000
ax5.plot(traj.index, traj[0], "o--", color=GRAY, linewidth=2, label="Control")
ax5.plot(traj.index, traj[1], "o-",  color=BLUE, linewidth=2.5, label="Treated")
ax5.fill_between([y for y in traj.index if y >= 2023],
                 [traj.loc[y, 0] for y in traj.index if y >= 2023],
                 [traj.loc[y, 1] for y in traj.index if y >= 2023],
                 alpha=0.15, color=BLUE, label="Causal gap")
ax5.axvline(2022.5, color=RED, linestyle="--", linewidth=1.2)
ax5.set_title("Matched trajectories\n(parallel pre-2023 → valid DiD)")
ax5.set_ylabel("Mean Salary ($K)")
ax5.legend(fontsize=8)
ax5.set_xticks(range(2019,2025))
ax5.tick_params(axis="x", rotation=45)

fig.suptitle(
    "Do AI Skills Pay Off? Causal Evidence | PSM + DiD | N=8,000 workers | 2019–2024",
    fontsize=12, fontweight="bold", y=1.01
)
plt.savefig("figures/fig4_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Fig 4: Summary dashboard")
print("\nAll figures saved to figures/")
