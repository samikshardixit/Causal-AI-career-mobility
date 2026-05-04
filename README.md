# Do AI Skills Actually Pay Off?
### A Causal Analysis of Career Mobility & Wages

> *"Does acquiring AI-related skills improve career outcomes — or are high-performing workers just more likely to acquire them?"*

This project applies labor economics methodology to one of the most contested questions in workforce intelligence: **is the AI wage premium causal, or is it entirely selection?**

---

## Key Findings

| Outcome | Raw Gap | Causal Effect (PSM+DiD) | Selection Bias |
|---|---|---|---|
| Salary (2024) | ~32% | **+8.8%** | ~23 pp |
| Promotion probability | — | **+4.5 pp** | — |
| Job mobility | — | **+5.3 pp** | — |

The naive comparison overstates the wage premium by ~4x. After correcting for observable and unobservable selection, the true causal effect is economically significant but far more modest.

---

## Why This Matters for Workforce Intelligence

Revelio-style workforce data is uniquely positioned to answer causal questions about skill transitions — but it requires the right identification strategy. This project demonstrates:

1. **The selection problem is real and large.** High-ability workers disproportionately adopt AI skills. Any correlation-based analysis will be heavily confounded.
2. **Natural experiments unlock causality.** ChatGPT's launch (Nov 2022) created exogenous variation in AI skill adoption timing — a rare instrument in labor economics.
3. **Event studies validate the design.** Pre-treatment parallel trends hold, giving us confidence the DiD estimator is clean.
4. **Skill taxonomy matters.** Not all "AI skills" are equal. The analysis distinguishes core ML, LLM/GenAI, MLOps, and AI-adjacent skills — mirroring how Revelio's taxonomy works.

---

## Methodology

### 1. Data Construction
- Synthetic worker panel: 8,000 workers × 6 years (2019–2024)
- 8 industries × 8 job families × 5 seniority levels
- Worker timelines include: salary, promotions, job switches, skill acquisition events
- Selection bias baked in: latent ability drives both AI adoption AND outcomes (the identification challenge)

### 2. Skill Intelligence Layer
Revelio-style taxonomy classifying skills into:
- `core_ml`: PyTorch, TensorFlow, scikit-learn, XGBoost
- `llm_genai`: LLMs, GPT, prompt engineering, RAG, LangChain
- `mlops_infra`: MLflow, SageMaker, model monitoring
- `ai_adjacent`: NLP, computer vision, forecasting, recommendation systems

AI intensity scores assigned to job families using O*NET-grounded exposure measures.

### 3. Causal Design

**Step 1 — Natural Experiment**
ChatGPT's launch (Nov 2022) as exogenous shock. Workers who adopted AI skills post-shock vs. comparable workers who did not.

**Step 2 — Propensity Score Matching (PSM)**
Match treated/control workers on 2022 pre-treatment observables:
- Seniority, industry, job family, AI exposure score, log salary

**Step 3 — Difference-in-Differences (DiD)**
Within matched sample, estimate:

```
log(salary)ᵢₜ = α + β₁·Treatedᵢ + β₂·PostShockₜ + β₃·(Treated × PostShock)ᵢₜ + γXᵢₜ + εᵢₜ
```

β₃ = causal effect of AI skill acquisition (clustered SEs at worker level)

**Step 4 — Event Study**
Year-by-year treatment effects to test parallel trends (pre-2022 coefficients ≈ 0 ✓)

### 4. Results

```
[A] log(salary):
    β₃ = 0.0844  (SE=0.0029, p<0.001)
    → +8.8% salary from acquiring AI skills
    → 95% CI: [8.2%, 9.4%]

[B] Promotion probability:
    Marginal effect = +4.51 pp  (p<0.001)

[C] Job switching rate:
    Marginal effect = +5.31 pp  (p<0.001)

Pre-trend test: max deviation = 0.0475 → Parallel trends hold ✓
```

---

## Repository Structure

```
ai_skills_project/
├── src/
│   ├── 01_generate_data.py       # Worker panel simulation
│   ├── 02_skill_taxonomy.py      # AI skill classification layer
│   ├── 03_causal_inference.py    # PSM + DiD + Event Study
│   ├── 04_visualizations.py      # Publication-quality figures
│   └── 05_dashboard.py           # Interactive HTML dashboard
├── data/
│   ├── worker_panel.csv           # Raw panel (8K workers × 6 years)
│   ├── worker_panel_enriched.csv  # + skill features
│   ├── matched_panel.csv          # PSM-matched sample
│   ├── event_study.csv            # Dynamic DiD coefficients
│   └── causal_results.csv         # Summary results
├── figures/                       # Static publication figures
├── outputs/
│   └── dashboard.html             # Interactive dashboard
└── README.md
```

---

## Running the Analysis

```bash
# Install dependencies
pip install pandas numpy scikit-learn statsmodels matplotlib plotly scipy linearmodels

# Run full pipeline
python src/01_generate_data.py
python src/02_skill_taxonomy.py
python src/03_causal_inference.py
python src/04_visualizations.py
python src/05_dashboard.py

# Open dashboard
open outputs/dashboard.html
```

---

## Limitations & Extensions

**Limitations:**
- Synthetic data: real Revelio data would strengthen external validity
- Selection on unobservables: DiD removes time-invariant confounds but not time-varying ones
- Single instrument: ChatGPT shock may not generalize to future AI waves
- Heterogeneous adoption: adoption year varies; a staggered DiD (Callaway-Sant'Anna) would be more robust

**Natural Extensions with Real Data:**
- Heterogeneous treatment effects by seniority, gender, and geography
- Skill-specific premiums (LLMs vs. classical ML vs. MLOps)
- Industry-level spillovers: does one team's AI adoption affect adjacent roles?
- Survival analysis: how long does the AI wage premium persist?

---

## References

- Alekseeva et al. (2021). *The demand for AI skills in the labor market.* Labour Economics.
- Stephany, Teutloff & Leone (2026). *AI Skills Improve Job Prospects: Causal Evidence from a Hiring Experiment.* arXiv.
- Bone et al. (2025). *AI wage premiums.* Referenced in Stephany et al.
- Acemoglu & Restrepo (2020). *Robots and jobs.* Journal of Political Economy.
- O*NET AI Exposure Scores (2023).

---

## Author

**Samiksha Dixit**  
Data Scientist | Causal Inference | Labor Economics  
MS Quantitative Economics & Econometrics, Northeastern University  
[LinkedIn](https://linkedin.com/in/samiksha-dixit-/) | dixit.sam@northeastern.edu
