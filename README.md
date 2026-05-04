# Do AI Skills Actually Pay Off?
### Causal Evidence on Wages, Career Mobility & Who Benefits Most

---

## The Finding

Most of the observed wage gap between AI-skilled and non-AI-skilled workers is selection, not causation. High-performing workers are more likely to adopt AI skills *and* earn more — regardless of AI skills. After correcting for this, a meaningful causal premium remains. But it doesn't show up immediately, and it doesn't accrue equally.

**Three things that are true simultaneously:**

1. **Selection explains most of the raw gap.** The naive salary difference is ~32%. The true causal effect is +8.8%. Selection was doing ~75% of the work.

2. **Returns are delayed, not instant.** Event study results show limited short-term gains in year one, with stronger effects emerging in year two post-adoption. The market takes time to price in new skills.

3. **Early-career workers benefit most.** Junior workers gain +11.0% vs. Directors who gain +7.6%. The gradient is monotonic and statistically significant at every level. This is consistent with AI skills functioning as a *credibility signal* — they matter more when employers have less prior information about a worker's true ability.

> *"High performers select into AI skills, but even after accounting for that, there's a real causal premium — and it materializes with a lag, and it's highest where information asymmetry is greatest."*

---

## Why This Is Hard to Estimate

The core identification challenge:

```
E[Y | D=1] - E[Y | D=0]  ≠  Average Treatment Effect
```

High-ability workers self-select into AI skill adoption. They would have earned more *anyway*. A naive regression conflates the two.

My identification strategy:

- **Natural experiment:** ChatGPT's launch (Nov 2022) created exogenous variation in AI skill adoption timing. No worker could have anticipated the exact shock.
- **Propensity Score Matching:** Balance treated/control on 2022 observables — industry, job family, seniority, AI exposure, salary. Reduces observable selection bias by ~50%.
- **Difference-in-Differences:** Removes time-invariant unobservables (latent ability) that PSM can't touch. β₃ = 0.0844 (SE=0.0029, p<0.001), clustered at worker level.
- **Event study:** Validates parallel trends pre-shock. Pre-2022 coefficients are statistically indistinguishable from zero.
- **Placebo tests:** Fake shock years (2020, 2021) produce null results (p>0.35). The effect only appears when it should.
- **Heterogeneous effects:** Interaction model confirms diminishing returns with seniority — statistically significant at every step (p<0.001).

---

## Results

| Outcome | Raw estimate | Causal estimate (PSM+DiD) |
|---|---|---|
| Salary effect | ~32% gap | **+8.8%** [8.2%, 9.4%] |
| Promotion probability | — | **+4.5 pp** |
| Job mobility | — | **+5.3 pp** |

**By seniority (heterogeneous effects):**

| Level | AI Wage Premium | Significant? |
|---|---|---|
| Junior | +11.0% | ✓ (p<0.001) |
| Mid | +8.6% | ✓ (p<0.001) |
| Senior | +9.5% | ✓ |
| Lead | +8.4% | ✓ (p<0.001) |
| Director | +7.6% | ✓ |

**Placebo tests:**

| Fake shock year | β₃ | p-value | Verdict |
|---|---|---|---|
| 2020 | -0.0023 | 0.35 | ✓ Null |
| 2021 | -0.0016 | 0.50 | ✓ Null |
| 2023 (real) | +0.0844 | <0.001 | ✓ Significant |

---

## Limitations (I'd Fix These With Real Data)

**Staggered adoption:** Workers adopted at different times (2023 vs 2024). Callaway & Sant'Anna (2021) show standard DiD can be biased with staggered treatment. A published paper would use their estimator.

**Synthetic data:** Parameters are grounded in published literature (Alekseeva et al. 2021, O*NET AI exposure scores) but real Revelio-style workforce data would strengthen external validity considerably.

**Bundled treatment:** "AI skills" is not one thing. Fine-tuning LLMs and learning to use ChatGPT are very different signals. A more granular skill taxonomy would allow heterogeneous effects by skill type.

**No attrition model:** I don't model workers leaving the panel. If high-ability leavers are systematically treated, results could be biased.

---

## Repository Structure

```
causal-ai-career-mobility/
├── src/
│   ├── 01_generate_data.py           # Worker panel simulation
│   ├── 02_skill_taxonomy.py          # AI skill classification layer
│   ├── 03_causal_inference.py        # PSM + DiD + Event Study
│   ├── 04_visualizations.py          # Publication figures
│   ├── 05_dashboard.py               # Interactive HTML dashboard
│   ├── 06_heterogeneous_effects.py   # Returns by seniority
│   └── 07_placebo_test.py            # Credibility checks
├── figures/                          # Static figures (6 total)
├── outputs/
│   └── dashboard.html                # Interactive dashboard
└── README.md
```

---

## Running the Pipeline

```bash
pip install pandas numpy scikit-learn statsmodels matplotlib scipy

# Run in order
python src/01_generate_data.py
python src/02_skill_taxonomy.py
python src/03_causal_inference.py
python src/04_visualizations.py
python src/05_dashboard.py
python src/06_heterogeneous_effects.py
python src/07_placebo_test.py

# Open dashboard
open outputs/dashboard.html
```

---

## References

- Alekseeva et al. (2021). *The demand for AI skills in the labor market.* Labour Economics.
- Stephany, Teutloff & Leone (2026). *AI Skills Improve Job Prospects.* arXiv.
- Acemoglu & Restrepo (2020). *Robots and jobs.* Journal of Political Economy.
- Callaway & Sant'Anna (2021). *Difference-in-differences with multiple time periods.* Journal of Econometrics.
- Felten, Raj & Seamans (2023). *Occupational heterogeneity in exposure to generative AI.* SSRN.
- O*NET AI Exposure Scores (2023).

---

## Author

**Samiksha Dixit**
Data Scientist · Causal Inference · Labor Economics
MS Quantitative Economics & Econometrics, Northeastern University
[linkedin.com/in/samiksha-dixit-/](https://linkedin.com/in/samiksha-dixit-/) · dixit.sam@northeastern.edu
