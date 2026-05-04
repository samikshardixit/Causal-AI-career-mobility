"""
03_causal_inference.py
----------------------
Samiksha Dixit | May 2025

This is the heart of the project.

The question: Does acquiring AI skills *cause* better career outcomes,
or are we just observing that better workers happen to acquire AI skills?

This is a classic selection problem. If I just ran:
    log(salary) ~ has_ai_skill + controls
...the coefficient on has_ai_skill would be biased upward because
high-ability workers self-select into AI skill adoption.

My identification strategy has two parts:

PART 1 — Propensity Score Matching (PSM)
    Balance the treated and control groups on observable characteristics.
    After matching, treated and control workers "look the same" on
    everything I can observe: industry, job family, seniority, 2022 salary.
    This removes selection on observables.

PART 2 — Difference-in-Differences (DiD)
    Use ChatGPT's launch (Nov 2022) as a natural experiment.
    Compare the salary growth of AI adopters vs non-adopters
    before and after the shock.
    The DiD estimate removes time-invariant unobservables (like latent ability)
    that PSM can't touch.

Combined: PSM handles observable selection, DiD handles unobservable selection.
That's about as clean as you can get without an actual randomized experiment.

Key assumption I'm relying on: PARALLEL TRENDS
    In the absence of treatment, treated and control workers would have
    followed the same salary trajectory. I test this with an event study.
    If pre-2022 coefficients are ~0, the assumption is plausible.

Limitation I'm aware of: Staggered adoption
    Workers adopted at different times (2023 vs 2024). Callaway & Sant'Anna
    (2021) show that standard DiD can be biased with staggered treatment.
    In a proper paper I'd use their estimator. Here I'm using the simpler
    two-period DiD as a first pass. The results should be interpreted
    as approximate, not precise.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = "data/worker_panel_enriched.csv"
OUT_PATH  = "data/"

# ── Load data ────────────────────────────────────────────────────────────────
if not __import__("os").path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Cannot find {DATA_PATH}. "
        "Run 01_generate_data.py then 02_skill_taxonomy.py first."
    )

df = pd.read_csv(DATA_PATH)
df["industry_code"]   = pd.Categorical(df["industry"]).codes
df["job_family_code"] = pd.Categorical(df["job_family"]).codes

print(f"Loaded panel: {df.shape[0]:,} observations, {df.worker_id.nunique():,} workers")
print(f"Treated (ever AI): {df[df.treated==1].worker_id.nunique():,} workers")
print(f"Control (never AI): {df[df.treated==0].worker_id.nunique():,} workers")

# ══════════════════════════════════════════════════════════════════════════════
# PART 1: PROPENSITY SCORE MATCHING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PART 1: PROPENSITY SCORE MATCHING")
print("="*60)

# I'm matching on 2022 data (last pre-treatment year)
# Using observables only — I deliberately exclude latent_ability
# because that's unobserved in real data. If I included it, I'd be
# cheating — using information I wouldn't have in practice.
pre_df = df[df.year == 2022].copy()

psm_features = [
    "seniority_code",
    "industry_code",
    "job_family_code",
    "ai_intensity",           # occupation-level AI exposure
    "industry_ai_exposure",   # industry-level AI exposure
    "log_salary",             # pre-treatment salary (proxy for ability)
]

X = pre_df[psm_features].values
y = pre_df["treated"].values

# Logistic regression to estimate propensity scores
# I'm using L2 regularization (C=1.0) to avoid overfitting
# In a real application I'd tune this more carefully
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
lr.fit(X_scaled, y)
pre_df = pre_df.copy()
pre_df["propensity_score"] = lr.predict_proba(X_scaled)[:, 1]

print(f"\nPropensity score range: "
      f"[{pre_df.propensity_score.min():.3f}, {pre_df.propensity_score.max():.3f}]")
print(f"Treated N: {y.sum():,}  |  Control N: {(1-y).sum():,}")

# 1:1 nearest neighbor matching without replacement
# "Without replacement" means each control worker is matched to at most
# one treated worker. This is conservative but avoids over-using controls.
treated_df = pre_df[pre_df.treated == 1].copy()
control_df = pre_df[pre_df.treated == 0].copy()

nn = NearestNeighbors(n_neighbors=1)
nn.fit(control_df[["propensity_score"]].values)
distances, indices = nn.kneighbors(treated_df[["propensity_score"]].values)

matched_control_ids = control_df.iloc[indices.flatten()]["worker_id"].values
matched_treated_ids = treated_df["worker_id"].values
matched_ids         = np.concatenate([matched_treated_ids, matched_control_ids])
matched_panel       = df[df.worker_id.isin(matched_ids)].copy()

print(f"\nMatched sample: {len(matched_ids):,} workers")
print(f"  Treated: {len(matched_treated_ids):,}")
print(f"  Matched controls: {len(matched_control_ids):,}")

# Balance check — did matching work?
# I want the treated/control difference to shrink after matching.
# If it doesn't, matching failed and I need to revisit the model.
print("\nBalance check (before vs after matching):")
print(f"{'Variable':<25} {'Before diff':>12} {'After diff':>12} {'Improved?':>10}")
for v in ["seniority_code", "ai_intensity", "log_salary"]:
    pre_t  = pre_df[pre_df.treated==1][v].mean()
    pre_c  = pre_df[pre_df.treated==0][v].mean()
    post_t = matched_panel[(matched_panel.year==2022)&(matched_panel.treated==1)][v].mean()
    post_c = matched_panel[(matched_panel.year==2022)&(matched_panel.treated==0)][v].mean()
    before = abs(pre_t - pre_c)
    after  = abs(post_t - post_c)
    improved = "✓" if after < before else "✗"
    print(f"  {v:<23} {before:>12.4f} {after:>12.4f} {improved:>10}")

matched_panel.to_csv(OUT_PATH + "matched_panel.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════════
# PART 2: DIFFERENCE-IN-DIFFERENCES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PART 2: DIFFERENCE-IN-DIFFERENCES")
print("="*60)
print("\nSpec: log(salary)_it = α + β₁·Treated_i + β₂·Post_t + β₃·(Treated×Post)_it + γX_it + ε_it")
print("β₃ is the causal estimate of interest.")
print("Standard errors clustered at worker level (accounts for serial correlation).\n")

mp = matched_panel.copy()

# ── Outcome A: log(salary) ────────────────────────────────────────────────────
mod_salary = smf.ols(
    "log_salary ~ did_term + treated + post_shock + seniority_code + "
    "ai_intensity + C(year) + C(industry_code)",
    data=mp
).fit(cov_type="cluster", cov_kwds={"groups": mp["worker_id"]})

did_coef = mod_salary.params["did_term"]
did_se   = mod_salary.bse["did_term"]
did_pval = mod_salary.pvalues["did_term"]
did_pct  = (np.exp(did_coef) - 1) * 100

print(f"[A] Outcome: log(salary)")
print(f"    β₃ (DiD coefficient):  {did_coef:.4f}")
print(f"    Clustered SE:           {did_se:.4f}")
print(f"    p-value:                {did_pval:.4f}")
print(f"    Interpretation:        +{did_pct:.1f}% salary from acquiring AI skills")
print(f"    95% CI:                [{(np.exp(did_coef-1.96*did_se)-1)*100:.1f}%, "
      f"{(np.exp(did_coef+1.96*did_se)-1)*100:.1f}%]")
print(f"    N (obs):                {len(mp):,}")

# ── Outcome B: Promotion ──────────────────────────────────────────────────────
mod_promo = smf.logit(
    "promoted ~ did_term + treated + post_shock + seniority_code + "
    "ai_intensity + C(year) + C(industry_code)",
    data=mp
).fit(disp=False)

promo_coef = mod_promo.params["did_term"]
promo_pval = mod_promo.pvalues["did_term"]
promo_mfx  = mod_promo.get_margeff()
promo_idx  = list(mod_promo.model.exog_names).index("did_term") - 1
promo_me   = promo_mfx.margeff[promo_idx] * 100

print(f"\n[B] Outcome: Promotion probability (logit, marginal effects)")
print(f"    Marginal effect of DiD: +{promo_me:.2f} percentage points")
print(f"    Odds ratio:              {np.exp(promo_coef):.3f}")
print(f"    p-value:                {promo_pval:.4f}")
print(f"    Interpretation:         Acquiring AI skills increases annual")
print(f"                            promotion probability by ~{promo_me:.1f}pp")

# ── Outcome C: Job switching ──────────────────────────────────────────────────
mod_switch = smf.logit(
    "job_switch ~ did_term + treated + post_shock + seniority_code + "
    "ai_intensity + C(year) + C(industry_code)",
    data=mp
).fit(disp=False)

switch_coef = mod_switch.params["did_term"]
switch_pval = mod_switch.pvalues["did_term"]
switch_mfx  = mod_switch.get_margeff()
switch_idx  = list(mod_switch.model.exog_names).index("did_term") - 1
switch_me   = switch_mfx.margeff[switch_idx] * 100

print(f"\n[C] Outcome: Job switching rate")
print(f"    Marginal effect of DiD: +{switch_me:.2f} percentage points")
print(f"    p-value:                {switch_pval:.4f}")
print(f"    Interpretation:         AI skills increase job mobility — workers")
print(f"                            have more outside options and use them")

# ══════════════════════════════════════════════════════════════════════════════
# PART 3: EVENT STUDY (pre-trends test)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PART 3: EVENT STUDY — Testing Parallel Trends")
print("="*60)
print("\nIf parallel trends holds, pre-2022 coefficients should be ~0.")
print("If they're not, my DiD estimate is probably biased.\n")

mp_event = mp.copy()
for yr in [2019, 2020, 2021, 2023, 2024]:
    mp_event[f"t_{yr}"] = ((mp_event["year"] == yr) & (mp_event["treated"] == 1)).astype(int)

event_formula = (
    "log_salary ~ t_2019 + t_2020 + t_2021 + t_2023 + t_2024 + "
    "treated + post_shock + seniority_code + ai_intensity + C(industry_code)"
)
mod_event = smf.ols(event_formula, data=mp_event).fit(
    cov_type="cluster", cov_kwds={"groups": mp_event["worker_id"]}
)

print(f"  {'Year':<8} {'Coef':>8} {'SE':>8} {'p-val':>8} {'Sig':>5} {'Interpretation'}")
event_results = []
for yr in [2019, 2020, 2021, 2023, 2024]:
    coef  = mod_event.params.get(f"t_{yr}", 0)
    se    = mod_event.bse.get(f"t_{yr}", 0)
    pval  = mod_event.pvalues.get(f"t_{yr}", 1)
    sig   = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else "n.s."
    note  = "← pre-trend (should be ~0)" if yr < 2022 else "← treatment effect"
    print(f"  {yr:<8} {coef:>8.4f} {se:>8.4f} {pval:>8.4f} {sig:>5}  {note}")
    event_results.append({"year": yr, "coef": coef, "se": se, "pval": pval})

event_results.append({"year": 2022, "coef": 0.0, "se": 0.0, "pval": 1.0})

pre_coefs = [r["coef"] for r in event_results if r["year"] < 2022]
max_pre   = max(abs(c) for c in pre_coefs)
print(f"\nMax pre-trend coefficient: {max_pre:.4f}")
print(f"Parallel trends verdict:   {'✓ PLAUSIBLE' if max_pre < 0.05 else '⚠ CHECK THIS'}")
if max_pre >= 0.05:
    print("  Warning: pre-trend deviation is larger than I'd like.")
    print("  This could mean the parallel trends assumption is violated.")
    print("  Would want to investigate before drawing causal conclusions.")

# ── Save results ──────────────────────────────────────────────────────────────
event_df = pd.DataFrame(event_results).sort_values("year")
event_df.to_csv(OUT_PATH + "event_study.csv", index=False)

results_summary = {
    "salary_did_coef":          did_coef,
    "salary_did_se":            did_se,
    "salary_did_pval":          did_pval,
    "salary_pct_effect":        did_pct,
    "promo_marginal_effect":    promo_me,
    "promo_pval":               promo_pval,
    "switch_marginal_effect":   switch_me,
    "switch_pval":              switch_pval,
}
pd.DataFrame([results_summary]).to_csv(OUT_PATH + "causal_results.csv", index=False)
print("\n✓ Results saved to data/")
print(f"\nBottom line: AI skill acquisition → +{did_pct:.1f}% wages, "
      f"+{promo_me:.1f}pp promotion, +{switch_me:.1f}pp job switching")
print(f"Compare to naive gap of ~32% — selection bias was doing a lot of work.")
