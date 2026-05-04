"""
01_generate_data.py
-------------------
Samiksha Dixit | May 2025

I'm building a synthetic worker panel here. Before anyone asks — yes, it's synthetic.
The reason is simple: I don't have access to Revelio's actual database (yet).
But the *structure* of this data mirrors what a real workforce panel looks like,
and the parameters are grounded in published labor economics literature.

The core identification challenge I'm trying to set up:
    High-ability workers are MORE likely to adopt AI skills.
    High-ability workers also earn more — regardless of AI skills.
    So any naive comparison (AI workers vs non-AI workers) will overstate
    the causal effect of AI skills on wages. This is textbook selection bias.

    E[Y | D=1] - E[Y | D=0] ≠ Average Treatment Effect (ATE)

I'm building this bias INTO the data on purpose, so that my causal design
in script 03 has something real to correct for. If selection bias weren't
there, DiD wouldn't be doing any work.

Key parameters grounded in:
- Alekseeva et al. (2021): AI roles command ~16% wage premium
- Acemoglu & Restrepo (2020): task displacement and wage effects
- O*NET AI exposure scores (2023) for industry/occupation AI intensity
"""

import numpy as np
import pandas as pd
from scipy import stats
import os

# I always set a seed. Reproducibility is non-negotiable.
np.random.seed(42)

N_WORKERS  = 8_000
YEARS      = list(range(2019, 2025))
SHOCK_YEAR = 2023   # First full calendar year post-ChatGPT (launched Nov 2022)
                    # The identification strategy depends on this shock
                    # being credibly exogenous. I think it is — no worker
                    # could have predicted the exact timing of GPT-4's release.

OUTPUT_PATH = "data/worker_panel.csv"

# Industry parameters
# base_salary: approximate median base (2019 dollars, BLS-informed)
# ai_exposure: from O*NET AI exposure index (Felten et al. 2023)
# Higher exposure = workers more likely to adopt AI skills post-shock
INDUSTRIES = {
    "Technology":    {"base_salary": 115_000, "ai_exposure": 0.72},
    "Finance":       {"base_salary": 105_000, "ai_exposure": 0.55},
    "Healthcare":    {"base_salary":  88_000, "ai_exposure": 0.38},
    "Retail":        {"base_salary":  72_000, "ai_exposure": 0.28},
    "Manufacturing": {"base_salary":  78_000, "ai_exposure": 0.22},
    "Consulting":    {"base_salary":  98_000, "ai_exposure": 0.48},
    "Education":     {"base_salary":  68_000, "ai_exposure": 0.31},
    "Media":         {"base_salary":  82_000, "ai_exposure": 0.44},
}

# Job family parameters
# salary_mult: occupation premium relative to industry base
# ai_affinity: probability of AI skill adoption (my best guess from
# Burning Glass / LinkedIn skill frequency data)
JOB_FAMILIES = {
    "Data Science":       {"salary_mult": 1.25, "ai_affinity": 0.85},
    "Software Eng":       {"salary_mult": 1.20, "ai_affinity": 0.75},
    "Product":            {"salary_mult": 1.10, "ai_affinity": 0.55},
    "Marketing":          {"salary_mult": 0.90, "ai_affinity": 0.45},
    "Operations":         {"salary_mult": 0.88, "ai_affinity": 0.30},
    "Finance/Accounting": {"salary_mult": 1.00, "ai_affinity": 0.35},
    "HR":                 {"salary_mult": 0.82, "ai_affinity": 0.25},
    "Research":           {"salary_mult": 1.05, "ai_affinity": 0.60},
}

SENIORITY      = ["Junior", "Mid", "Senior", "Lead", "Director"]
SENIORITY_MULT = [0.70, 1.00, 1.30, 1.60, 2.10]

industries   = list(INDUSTRIES.keys())
job_families = list(JOB_FAMILIES.keys())

worker_industry  = np.random.choice(industries,   N_WORKERS)
worker_job_fam   = np.random.choice(job_families, N_WORKERS)
worker_seniority = np.random.choice(len(SENIORITY), N_WORKERS, p=[0.20,0.35,0.25,0.12,0.08])

# Latent ability — the key unobservable.
# In real data I'd never observe this. That's the whole problem.
# It drives both AI adoption AND wages, creating the selection bias.
# Standard normal: think of it as "how good is this person at their job,
# conditional on everything else we observe about them"
worker_ability = np.random.normal(0, 1, N_WORKERS)
worker_tenure  = np.random.randint(0, 15, N_WORKERS)

base_salaries = np.array([
    INDUSTRIES[ind]["base_salary"] *
    JOB_FAMILIES[jf]["salary_mult"] *
    SENIORITY_MULT[sen]
    for ind, jf, sen in zip(worker_industry, worker_job_fam, worker_seniority)
])
base_salaries += worker_ability * 5_000   # ability premium (conservative)
base_salaries += np.random.normal(0, 8_000, N_WORKERS)

# Treatment assignment — THIS IS THE MOST IMPORTANT PART.
# Making adoption a function of:
#   (a) industry AI exposure — structural factor, observable
#   (b) job family AI affinity — occupational factor, observable
#   (c) latent ability — the unobservable that creates endogeneity
#
# Because ability drives both adoption AND wages:
#   E[ε | D=1] > 0
# A naive OLS of wages on AI skills will be upward biased.
# This is exactly what PSM + DiD corrects for.

def ai_adoption_prob(i):
    ind_exposure = INDUSTRIES[worker_industry[i]]["ai_exposure"]
    jf_affinity  = JOB_FAMILIES[worker_job_fam[i]]["ai_affinity"]
    # CDF maps ability to [0,1] — higher ability = higher adoption prob
    ability_pull = stats.norm.cdf(worker_ability[i]) * 0.3
    return np.clip(ind_exposure * 0.4 + jf_affinity * 0.4 + ability_pull, 0.05, 0.90)

adopt_probs = np.array([ai_adoption_prob(i) for i in range(N_WORKERS)])
acquired_ai = np.random.binomial(1, adopt_probs)

# Adoption timing: post-shock only
# 65% adopt in 2023 (early movers), 35% in 2024 (later adopters)
ai_adoption_year = np.where(
    acquired_ai == 1,
    np.random.choice([2023, 2024], N_WORKERS, p=[0.65, 0.35]),
    9999
)

records = []

for worker_id in range(N_WORKERS):
    ind     = worker_industry[worker_id]
    jf      = worker_job_fam[worker_id]
    sen     = worker_seniority[worker_id]
    abil    = worker_ability[worker_id]
    base    = base_salaries[worker_id]
    adop    = acquired_ai[worker_id]
    adop_yr = ai_adoption_year[worker_id]

    salary            = base
    current_seniority = sen
    cum_promoted      = 0

    for year in YEARS:
        has_ai_skill = int(year >= adop_yr)

        # Salary follows Mincer-style earnings equation:
        # log(w) = α + β*experience + γ*AI + δ*(AI × post) + ε
        # 3.5% base growth + ability gradient + noise
        annual_growth = 0.035 + abil * 0.008 + np.random.normal(0, 0.02)

        # AI wage effect grounded in Alekseeva et al. (~16% total)
        # Spread over 2 years with heterogeneity — some workers benefit more
        ai_wage_effect = 0.0
        if has_ai_skill and year >= SHOCK_YEAR:
            years_since   = year - adop_yr + 1
            ai_wage_effect = min(0.08 * years_since, 0.16)
            ai_wage_effect += np.random.normal(0, 0.02)  # heterogeneous TE

        salary = salary * (1 + annual_growth + ai_wage_effect)
        salary += np.random.normal(0, 3_000)  # classical measurement error

        # Promotion: base rate ~8%, ability raises it, AI gives small boost
        # The AI boost here is speculative — I'd want to validate with real data
        promo_base     = 0.08 + abil * 0.04
        ai_promo_boost = 0.06 if (has_ai_skill and year >= SHOCK_YEAR) else 0
        promoted       = int(np.random.random() < promo_base + ai_promo_boost)
        if promoted and current_seniority < 4:
            current_seniority += 1
            cum_promoted += 1

        # Job switching: AI skills create outside options → more mobility
        # Tenure reduces switching (standard finding in labor econ)
        switch_base     = 0.12 - 0.01 * min((year - 2019), 4)
        ai_switch_boost = 0.05 if (has_ai_skill and year >= SHOCK_YEAR) else 0
        job_switch      = int(np.random.random() < switch_base + ai_switch_boost)

        records.append({
            "worker_id":             worker_id,
            "year":                  year,
            "industry":              ind,
            "job_family":            jf,
            "seniority_code":        current_seniority,
            "seniority":             SENIORITY[current_seniority],
            "latent_ability":        round(abil, 4),  # unobserved in real data!
            "base_salary_2019":      round(base, 0),
            "salary":                round(max(salary, 30_000), 0),
            "log_salary":            round(np.log(max(salary, 30_000)), 6),
            "has_ai_skill":          has_ai_skill,
            "treated":               adop,
            "ai_adoption_year":      adop_yr if adop_yr < 9999 else None,
            "promoted":              promoted,
            "job_switch":            job_switch,
            "cumulative_promotions": cum_promoted,
            "post_shock":            int(year >= SHOCK_YEAR),
            "did_term":              adop * int(year >= SHOCK_YEAR),  # β₃ in DiD
            "adopt_prob":            round(adopt_probs[worker_id], 4),
        })

os.makedirs("data", exist_ok=True)
df = pd.DataFrame(records)
df.to_csv(OUTPUT_PATH, index=False)

print(f"Panel shape: {df.shape}")
print(f"Workers: {df.worker_id.nunique():,}")
print(f"Years: {sorted(df.year.unique())}")
print(f"AI adopters: {df[df.treated==1].worker_id.nunique():,} "
      f"({100*df[df.treated==1].worker_id.nunique()/N_WORKERS:.1f}%)")
print(f"\nNaive salary comparison (2024) — this INCLUDES selection bias:")
print(df[df.year==2024].groupby("treated")["salary"].mean().round(0))
print("^ The gap above is larger than the true causal effect. That's the point.")
