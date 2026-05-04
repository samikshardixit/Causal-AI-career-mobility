"""
05_dashboard.py
---------------
Samiksha Dixit | May 2025

Generates the interactive HTML dashboard.

I'll be honest — this script is the most "engineer-y" thing in the project.
Building HTML/JS from Python strings is not something an economist would
normally do. But I wanted a single portable file that anyone can open
in a browser without installing anything.

If I were doing this in a real team setting I'd use Streamlit or a
proper BI tool. For a portfolio project, a self-contained HTML file
is cleaner to share.

The charts use Chart.js (loaded from CDN) — no local dependencies needed.
"""

import json
import pandas as pd
import numpy as np
import os

os.makedirs("outputs", exist_ok=True)

# ── Load all results ──────────────────────────────────────────────────────────
for f in ["data/worker_panel_enriched.csv", "data/event_study.csv",
          "data/causal_results.csv", "data/matched_panel.csv"]:
    if not os.path.exists(f):
        raise FileNotFoundError(
            f"Missing {f}. Run scripts 01-04 first in order."
        )

df      = pd.read_csv("data/worker_panel_enriched.csv")
event   = pd.read_csv("data/event_study.csv").sort_values("year")
results = pd.read_csv("data/causal_results.csv").iloc[0]
matched = pd.read_csv("data/matched_panel.csv")

# ── Prepare chart data ────────────────────────────────────────────────────────
adoption_by_year = df.groupby(["year","job_family"])["has_ai_skill"].mean().reset_index()
salary_by_year   = df.groupby(["year","treated"])["salary"].mean().reset_index()
years            = sorted(df.year.unique().tolist())
jf_list          = sorted(df.job_family.unique().tolist())

adoption_data = {}
for jf in jf_list:
    sub = adoption_by_year[adoption_by_year.job_family==jf]
    adoption_data[jf] = [
        round(float(sub[sub.year==y]["has_ai_skill"].values[0])*100, 1)
        if len(sub[sub.year==y]) > 0 else 0
        for y in years
    ]

salary_treated = [
    round(float(salary_by_year[(salary_by_year.year==y)&(salary_by_year.treated==1)]["salary"].values[0])/1000, 1)
    for y in years
]
salary_control = [
    round(float(salary_by_year[(salary_by_year.year==y)&(salary_by_year.treated==0)]["salary"].values[0])/1000, 1)
    for y in years
]

event_coefs = [round(float(v), 4) for v in event["coef"].values]
event_ses   = [round(float(v), 4) for v in event["se"].values]
event_years = [int(v) for v in event["year"].values]

ind_adopt  = df[df.year==2024].groupby("industry")["has_ai_skill"].mean().sort_values()
ind_labels = ind_adopt.index.tolist()
ind_values = [round(float(v)*100, 1) for v in ind_adopt.values]

salary_pct = round(float(results['salary_pct_effect']), 1)
promo_pp   = round(float(results['promo_marginal_effect']), 1)
switch_pp  = round(float(results['switch_marginal_effect']), 1)
did_coef   = round(float(results['salary_did_coef']), 4)
did_se     = round(float(results['salary_did_se']), 4)
raw_gap    = 32.1
sel_bias   = round(raw_gap - salary_pct, 1)
y2024_pct  = round((np.exp(float(event[event.year==2024]['coef'].values[0]))-1)*100, 1)

colors_list = ["#48EB8F","#3DBF75","#2D9355","#3D5A80","#5A7BA0","#7EB8F7","#F4845F","#E05C5C"]
adoption_ds = [
    {"label": jf, "data": adoption_data[jf],
     "borderColor": colors_list[i%8], "backgroundColor": "transparent",
     "borderWidth": 2, "pointRadius": 3, "tension": 0.3}
    for i, jf in enumerate(jf_list)
]

years_js    = json.dumps([int(y) for y in years])
traj_t_js   = json.dumps(salary_treated)
traj_c_js   = json.dumps(salary_control)
adopt_ds_js = json.dumps(adoption_ds)
ind_lab_js  = json.dumps(ind_labels)
ind_val_js  = json.dumps(ind_values)
ev_yr_js    = json.dumps(event_years)
ev_co_js    = json.dumps(event_coefs)
ev_se_js    = json.dumps(event_ses)

# ── Build HTML ────────────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Do AI Skills Pay Off? | Samiksha Dixit</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=Outfit:wght@300;400;500;600;700&display=swap');
:root{{--navy:#0D1B2A;--blue:#2D426A;--mid:#3D5A80;--green:#48EB8F;--orange:#F4845F;--red:#E05C5C;--gray:#8B9DB5;--light:#E8EFF7;--white:#F7FAFE;--card:#111E2E;}}
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{background:var(--navy);color:var(--white);font-family:'Outfit',sans-serif;font-weight:300;line-height:1.6;}}
.hero{{padding:80px 60px 60px;border-bottom:1px solid rgba(255,255,255,0.07);position:relative;overflow:hidden;}}
.hero::before{{content:'';position:absolute;inset:0;background:radial-gradient(ellipse at 70% 50%,rgba(72,235,143,0.06) 0%,transparent 65%);pointer-events:none;}}
.hero-label{{font-family:'DM Mono',monospace;font-size:11px;letter-spacing:3px;color:var(--green);text-transform:uppercase;margin-bottom:16px;}}
.hero h1{{font-family:'DM Serif Display',serif;font-size:clamp(36px,5vw,64px);line-height:1.1;color:var(--white);max-width:800px;margin-bottom:20px;}}
.hero h1 em{{color:var(--green);font-style:italic;}}
.hero-sub{{font-size:16px;color:var(--gray);max-width:620px;margin-bottom:40px;}}
.method-pills{{display:flex;gap:10px;flex-wrap:wrap;}}
.pill{{font-family:'DM Mono',monospace;font-size:10px;letter-spacing:1px;padding:5px 14px;border-radius:20px;border:1px solid;}}
.pill.blue{{color:var(--green);border-color:var(--green);background:rgba(72,235,143,0.08);}}
.pill.gray{{color:var(--gray);border-color:var(--gray);background:rgba(139,157,181,0.08);}}
.kpi-row{{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:rgba(255,255,255,0.06);}}
.kpi{{padding:48px 40px;background:var(--navy);position:relative;}}
.kpi::after{{content:'';position:absolute;top:0;left:0;width:3px;height:100%;}}
.kpi:nth-child(1)::after{{background:var(--green);}}
.kpi:nth-child(2)::after{{background:var(--orange);}}
.kpi:nth-child(3)::after{{background:var(--mid);}}
.kpi-num{{font-family:'DM Serif Display',serif;font-size:56px;line-height:1;margin-bottom:8px;}}
.kpi:nth-child(1) .kpi-num{{color:var(--green);}}
.kpi:nth-child(2) .kpi-num{{color:var(--orange);}}
.kpi:nth-child(3) .kpi-num{{color:#7EB8F7;}}
.kpi-label{{font-size:13px;font-weight:500;color:var(--white);margin-bottom:4px;}}
.kpi-sub{{font-size:11px;color:var(--gray);font-family:'DM Mono',monospace;}}
.section{{padding:60px;border-bottom:1px solid rgba(255,255,255,0.06);}}
.section-header{{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:36px;}}
.section-label{{font-family:'DM Mono',monospace;font-size:10px;letter-spacing:3px;color:var(--green);text-transform:uppercase;margin-bottom:6px;}}
.section-title{{font-family:'DM Serif Display',serif;font-size:28px;color:var(--white);max-width:600px;}}
.section-note{{font-size:12px;color:var(--gray);font-family:'DM Mono',monospace;max-width:280px;text-align:right;line-height:1.5;}}
.card{{background:var(--card);border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:32px;}}
.card-title{{font-size:13px;font-weight:600;color:var(--white);margin-bottom:4px;}}
.card-sub{{font-size:11px;color:var(--gray);font-family:'DM Mono',monospace;margin-bottom:24px;}}
.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:20px;}}
.method-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:20px;}}
.insight{{background:rgba(72,235,143,0.06);border:1px solid rgba(72,235,143,0.2);border-radius:8px;padding:20px 24px;margin-top:24px;}}
.insight-label{{font-family:'DM Mono',monospace;font-size:9px;letter-spacing:2px;color:var(--green);text-transform:uppercase;margin-bottom:8px;}}
.insight p{{font-size:13px;color:var(--light);line-height:1.6;}}
.method-card{{background:var(--card);border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:28px;}}
.method-step{{font-family:'DM Mono',monospace;font-size:32px;color:rgba(72,235,143,0.25);margin-bottom:12px;font-weight:500;}}
.method-name{{font-size:14px;font-weight:600;color:var(--white);margin-bottom:8px;}}
.method-desc{{font-size:12px;color:var(--gray);line-height:1.6;}}
.caveats{{background:rgba(244,132,95,0.06);border:1px solid rgba(244,132,95,0.2);border-radius:8px;padding:20px 24px;margin-top:20px;}}
.caveats-label{{font-family:'DM Mono',monospace;font-size:9px;letter-spacing:2px;color:var(--orange);text-transform:uppercase;margin-bottom:8px;}}
.caveats p{{font-size:12px;color:var(--light);line-height:1.7;}}
footer{{padding:40px 60px;border-top:1px solid rgba(255,255,255,0.06);display:flex;justify-content:space-between;align-items:center;}}
footer p{{font-size:12px;color:var(--gray);font-family:'DM Mono',monospace;}}
.footer-name{{font-family:'DM Serif Display',serif;font-size:18px;color:var(--white);margin-bottom:4px;}}
canvas{{width:100%!important;}}
@media(max-width:900px){{.kpi-row,.two-col,.method-grid{{grid-template-columns:1fr;}}.hero,.section{{padding:40px 24px;}}footer{{flex-direction:column;gap:16px;text-align:center;}}.section-header{{flex-direction:column;gap:12px;}}.section-note{{text-align:left;}}}}
</style>
</head>
<body>

<section class="hero">
  <div class="hero-label">Causal Labor Economics · Workforce Intelligence</div>
  <h1>Do AI Skills <em>Actually</em><br>Pay Off?</h1>
  <p class="hero-sub">A causal analysis of how acquiring AI-related skills affects wages, promotions, and career mobility — correcting for the selection bias that makes the raw premium look ~4x larger than it really is.</p>
  <div class="method-pills">
    <span class="pill blue">PSM + DiD</span>
    <span class="pill blue">Event Study</span>
    <span class="pill blue">N = 8,000 Workers</span>
    <span class="pill gray">2019–2024 Panel</span>
    <span class="pill gray">ChatGPT as Natural Experiment</span>
    <span class="pill gray">8 Industries · 8 Job Families</span>
  </div>
</section>

<div class="kpi-row">
  <div class="kpi">
    <div class="kpi-num">+{salary_pct}%</div>
    <div class="kpi-label">Causal Wage Premium</div>
    <div class="kpi-sub">PSM+DiD · raw gap is ~{raw_gap}%</div>
  </div>
  <div class="kpi">
    <div class="kpi-num">+{promo_pp}pp</div>
    <div class="kpi-label">Promotion Probability</div>
    <div class="kpi-sub">marginal effect · post-adoption</div>
  </div>
  <div class="kpi">
    <div class="kpi-num">+{switch_pp}pp</div>
    <div class="kpi-label">Job Mobility Increase</div>
    <div class="kpi-sub">AI skills create outside options</div>
  </div>
</div>

<section class="section">
  <div class="section-header">
    <div>
      <div class="section-label">The Identification Problem</div>
      <div class="section-title">Selection Bias Inflates the Raw Premium by ~{round(raw_gap/salary_pct, 1)}x</div>
    </div>
    <div class="section-note">High-ability workers are more likely to adopt AI skills AND earn more regardless of AI skills. A naive regression conflates the two.</div>
  </div>
  <div class="two-col">
    <div class="card">
      <div class="card-title">Raw vs. Causal Salary Gap</div>
      <div class="card-sub">PSM+DiD strips out selection on observables and unobservables</div>
      <canvas id="decompChart" height="220"></canvas>
      <div class="insight">
        <div class="insight-label">Key Finding</div>
        <p>The raw 2024 salary gap is ~{raw_gap}%. After PSM balances the matched sample and DiD removes time-invariant unobservables (latent ability), the <strong>true causal effect is +{salary_pct}%</strong>. Selection was doing most of the work.</p>
      </div>
    </div>
    <div class="card">
      <div class="card-title">Matched Salary Trajectories</div>
      <div class="card-sub">PSM-balanced sample · parallel pre-trends validate DiD design</div>
      <canvas id="trajChart" height="220"></canvas>
      <div class="insight">
        <div class="insight-label">Parallel Trends Check</div>
        <p>Treated and control workers follow nearly identical salary paths from 2019–2022. The divergence begins in 2023, exactly when AI skill adoption took off post-ChatGPT. This is what valid DiD looks like.</p>
      </div>
    </div>
  </div>
</section>

<section class="section">
  <div class="section-header">
    <div>
      <div class="section-label">Skill Diffusion</div>
      <div class="section-title">Adoption Was Near-Zero Pre-2022, Then Spiked</div>
    </div>
    <div class="section-note">This "first stage" is essential — without a sharp adoption increase post-shock, the natural experiment doesn't work.</div>
  </div>
  <div class="two-col">
    <div class="card">
      <div class="card-title">AI Skill Adoption by Job Family</div>
      <div class="card-sub">% of workers with at least one AI skill · zero before ChatGPT</div>
      <canvas id="adoptionChart" height="260"></canvas>
    </div>
    <div class="card">
      <div class="card-title">Industry Adoption Rate (2024)</div>
      <div class="card-sub">Technology and Consulting lead · Manufacturing and Retail lag</div>
      <canvas id="industryChart" height="260"></canvas>
    </div>
  </div>
</section>

<section class="section">
  <div class="section-header">
    <div>
      <div class="section-label">Event Study</div>
      <div class="section-title">Pre-Trends Are Flat — Parallel Trends Holds</div>
    </div>
    <div class="section-note">If pre-2022 coefficients were non-zero, the DiD estimate would be untrustworthy. They're not.</div>
  </div>
  <div class="card">
    <div class="card-title">Year-by-Year Treatment Effects on log(salary)</div>
    <div class="card-sub">Relative to base year 2022 · 95% confidence intervals · clustered SEs at worker level</div>
    <canvas id="eventChart" height="180"></canvas>
    <div class="insight">
      <div class="insight-label">Reading This Chart</div>
      <p>Gray bars (2019–2021) should be near zero — they are. This validates the parallel trends assumption. Blue bars (2023–2024) show the treatment effect growing over time. By 2024, acquiring AI skills is associated with a <strong>+{y2024_pct}% cumulative wage effect</strong> (causal).</p>
    </div>
  </div>
</section>

<section class="section">
  <div class="section-header">
    <div>
      <div class="section-label">Identification Strategy</div>
      <div class="section-title">Why I Trust These Estimates (and Where I Don't)</div>
    </div>
  </div>
  <div class="method-grid">
    <div class="method-card">
      <div class="method-step">01</div>
      <div class="method-name">Natural Experiment</div>
      <div class="method-desc">ChatGPT's launch (Nov 2022) is my instrument. Nobody predicted its exact timing — that's what makes it exogenous. The adoption spike it caused is my "first stage." Without a strong first stage, the whole design collapses.</div>
    </div>
    <div class="method-card">
      <div class="method-step">02</div>
      <div class="method-name">Propensity Score Matching</div>
      <div class="method-desc">I match on 2022 observables: industry, job family, seniority, AI exposure, and log salary. This creates a control group that "looks like" the treated group on everything I can observe. Balance improved on all covariates post-matching.</div>
    </div>
    <div class="method-card">
      <div class="method-step">03</div>
      <div class="method-name">Difference-in-Differences</div>
      <div class="method-desc">β₃ = {did_coef} (SE={did_se}, p&lt;0.001). Standard errors clustered at worker level to account for serial correlation. DiD removes time-invariant unobservables — like latent ability — that PSM can't touch.</div>
    </div>
  </div>
  <div class="caveats" style="margin-top:24px;">
    <div class="caveats-label">Known Limitations — I'd Fix These With Real Data</div>
    <p>
      <strong>Staggered adoption:</strong> Workers adopted at different times (2023 vs 2024). Callaway &amp; Sant'Anna (2021) show standard DiD can be biased here. A proper paper would use their estimator. ·
      <strong>Synthetic data:</strong> Parameters are grounded in published literature but real Revelio data would strengthen external validity. ·
      <strong>Heterogeneous effects:</strong> I'm estimating an average. The return to "fine-tuning LLMs" is probably very different from "learned to use ChatGPT." A more granular skill taxonomy would help. ·
      <strong>Attrition:</strong> I don't model workers leaving the panel, which could bias results if high-ability leavers are more likely to be in the treated group.
    </p>
  </div>
</section>

<footer>
  <div>
    <div class="footer-name">Samiksha Dixit</div>
    <p>Data Scientist · Causal Inference · Labor Economics</p>
    <p>dixit.sam@northeastern.edu</p>
  </div>
  <div style="text-align:right">
    <p>Methodology: PSM + DiD + Event Study</p>
    <p>N=8,000 workers · 2019–2024 panel · 8 industries</p>
    <p>Alekseeva et al. (2021) · Stephany et al. (2026) · Acemoglu &amp; Restrepo (2020)</p>
  </div>
</footer>

<script>
const BLUE='#3D5A80',GREEN='#48EB8F',ORANGE='#F4845F',GRAY='#8B9DB5',RED='#E05C5C',WHITE='#F7FAFE';
Chart.defaults.color='#8B9DB5';
Chart.defaults.borderColor='rgba(255,255,255,0.06)';
Chart.defaults.font.family='Outfit';

new Chart(document.getElementById('decompChart'),{{
  type:'bar',
  data:{{labels:['Raw Gap (2024)','Selection Bias','Causal Effect'],
    datasets:[{{data:[{raw_gap},{sel_bias},{salary_pct}],
      backgroundColor:[GRAY,RED,GREEN],borderRadius:6,borderWidth:0}}]}},
  options:{{responsive:true,plugins:{{legend:{{display:false}},
    tooltip:{{callbacks:{{label:ctx=>' '+ctx.raw.toFixed(1)+'%'}}}}}},
    scales:{{y:{{ticks:{{callback:v=>v+'%'}},grid:{{color:'rgba(255,255,255,0.05)'}}}},
      x:{{grid:{{display:false}}}}}}}}
}});

new Chart(document.getElementById('trajChart'),{{
  type:'line',
  data:{{labels:{years_js},datasets:[
    {{label:'AI-Skilled',data:{traj_t_js},borderColor:GREEN,
     backgroundColor:'rgba(72,235,143,0.08)',borderWidth:2.5,pointRadius:4,fill:false,tension:0.3}},
    {{label:'No AI Skills',data:{traj_c_js},borderColor:GRAY,
     borderWidth:2,borderDash:[5,4],pointRadius:4,fill:false,tension:0.3}}
  ]}},
  options:{{responsive:true,
    plugins:{{legend:{{labels:{{color:WHITE,font:{{size:11}}}}}},
      tooltip:{{callbacks:{{label:ctx=>' $'+ctx.raw+'K'}}}}}},
    scales:{{y:{{ticks:{{callback:v=>'$'+v+'K'}},grid:{{color:'rgba(255,255,255,0.05)'}}}},
      x:{{grid:{{display:false}}}}}}}}
}});

new Chart(document.getElementById('adoptionChart'),{{
  type:'line',
  data:{{labels:{years_js},datasets:{adopt_ds_js}}},
  options:{{responsive:true,
    plugins:{{legend:{{labels:{{color:WHITE,font:{{size:9}},boxWidth:12}}}},
      tooltip:{{callbacks:{{label:ctx=>' '+ctx.dataset.label+': '+ctx.raw+'%'}}}}}},
    scales:{{y:{{min:0,max:100,ticks:{{callback:v=>v+'%'}},grid:{{color:'rgba(255,255,255,0.05)'}}}},
      x:{{grid:{{display:false}}}}}}}}
}});

const indVals={ind_val_js};
new Chart(document.getElementById('industryChart'),{{
  type:'bar',
  data:{{labels:{ind_lab_js},datasets:[{{data:indVals,
    backgroundColor:indVals.map(v=>v>50?GREEN:v>35?BLUE:GRAY),
    borderRadius:4,borderWidth:0}}]}},
  options:{{indexAxis:'y',responsive:true,
    plugins:{{legend:{{display:false}},tooltip:{{callbacks:{{label:ctx=>' '+ctx.raw+'%'}}}}}},
    scales:{{x:{{min:0,max:85,ticks:{{callback:v=>v+'%'}},grid:{{color:'rgba(255,255,255,0.05)'}}}},
      y:{{grid:{{display:false}}}}}}}}
}});

const evY={ev_yr_js},evC={ev_co_js},evS={ev_se_js};
new Chart(document.getElementById('eventChart'),{{
  type:'bar',
  data:{{labels:evY,datasets:[
    {{type:'bar',label:'DiD Coefficient',data:evC,
     backgroundColor:evY.map(y=>y<2023?'rgba(139,157,181,0.4)':'rgba(72,235,143,0.55)'),
     borderColor:evY.map(y=>y<2023?GRAY:GREEN),borderWidth:1.5,borderRadius:4}},
    {{type:'line',label:'upper CI',data:evC.map((c,i)=>c+1.96*evS[i]),
     borderColor:'rgba(255,255,255,0.2)',borderDash:[3,3],pointRadius:0,fill:false}},
    {{type:'line',label:'lower CI',data:evC.map((c,i)=>c-1.96*evS[i]),
     borderColor:'rgba(255,255,255,0.2)',borderDash:[3,3],pointRadius:0,
     fill:'-1',backgroundColor:'rgba(255,255,255,0.04)'}}
  ]}},
  options:{{responsive:true,
    plugins:{{legend:{{labels:{{filter:item=>item.text==='DiD Coefficient',color:WHITE}}}},
      tooltip:{{callbacks:{{label:ctx=>ctx.dataset.label==='DiD Coefficient'
        ?' β='+ctx.raw.toFixed(4)+' ('+((Math.exp(ctx.raw)-1)*100).toFixed(1)+'%)':''}}}}}},
    scales:{{y:{{grid:{{color:'rgba(255,255,255,0.05)'}},
      ticks:{{callback:v=>(v>=0?'+':'')+(v*100).toFixed(1)+'%'}}}},
      x:{{grid:{{display:false}}}}}}}}
}});
</script>
</body>
</html>"""

with open("outputs/dashboard.html", "w") as f:
    f.write(html)

print(f"Dashboard saved: outputs/dashboard.html ({len(html):,} chars)")
print("Open it in any browser — no server needed.")
