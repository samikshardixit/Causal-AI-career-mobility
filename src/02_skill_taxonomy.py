"""
02_skill_taxonomy.py
--------------------
Samiksha Dixit | May 2025

This script builds the skill intelligence layer — essentially a poor man's
version of what Revelio does with their full taxonomy and NLP pipeline.

In a real analysis, I'd use Revelio's actual skill ontology (or something
like Burning Glass / EMSI) to classify skills from job postings and resumes.
Here I'm assigning AI intensity scores at the job-family level, which is
a reasonable proxy when you don't have skill-level data.

The taxonomy structure mirrors how labor economists typically think about
skill categories — following Autor, Levy & Murnane (2003) task framework
but updated for the AI era.

One thing I keep reminding myself: "AI skills" is not one thing.
There's a big difference between:
  - Someone who learned to prompt ChatGPT (low barrier, low signal)
  - Someone who fine-tunes LLMs and deploys ML pipelines (high barrier, high signal)
The treatment variable here bundles all of this together, which is
a known limitation. Heterogeneous treatment effects analysis would help.
"""

import pandas as pd
import numpy as np
import os

# Revelio-style skill taxonomy
# Organized by cluster — mirrors how Burning Glass and O*NET group skills
# I'm using this to assign AI intensity scores, not to classify actual text
# (that would require NLP, which is a separate project)

AI_SKILLS_TAXONOMY = {
    "core_ml": [
        # Classical ML — the foundation
        "machine learning", "scikit-learn", "xgboost", "lightgbm",
        "random forest", "gradient boosting", "neural networks",
        "pytorch", "tensorflow", "keras",
    ],
    "llm_genai": [
        # Post-2022 skills — the ChatGPT wave
        # These are what drove the adoption spike I'm modeling
        "large language models", "llm", "gpt", "prompt engineering",
        "fine-tuning", "rag", "retrieval augmented generation",
        "langchain", "hugging face", "transformers",
    ],
    "mlops": [
        # Production ML — often overlooked but very valued
        "mlflow", "kubeflow", "sagemaker", "vertex ai",
        "model monitoring", "feature store", "ml pipelines",
    ],
    "data_engineering_ai": [
        # The plumbing that makes AI possible
        "pyspark", "databricks", "vector databases",
        "embedding models", "etl for ml",
    ],
    "ai_adjacent": [
        # Applied ML domains
        "nlp", "computer vision", "recommendation systems",
        "time series ml", "anomaly detection", "forecasting ml",
    ],
}

NON_AI_SKILLS = {
    "traditional_analytics": ["excel", "tableau", "powerbi", "sql", "stata", "sas"],
    "soft_skills":           ["communication", "project management", "leadership"],
    "domain_specific":       ["financial modeling", "supply chain", "hr management"],
}
# Note: SQL is in non-AI here, but "advanced SQL" or "SQL for ML" would be
# in the AI-adjacent bucket in a more granular taxonomy

# AI intensity by job family
# Scale: 0 = no AI exposure, 1 = core AI role
# These are my estimates informed by O*NET AI exposure scores
# (Felten, Raj & Seamans 2023) mapped to occupation categories
JOB_FAMILY_AI_INTENSITY = {
    "Data Science":       0.92,  # this is basically an AI job by definition now
    "Software Eng":       0.78,  # Copilot, code generation, ML infra
    "Research":           0.65,  # depends heavily on the research domain
    "Product":            0.52,  # AI-powered features, experimentation
    "Marketing":          0.41,  # AI content, targeting, analytics
    "Finance/Accounting": 0.35,  # some automation, but slower adoption
    "Operations":         0.30,  # process automation, but mostly non-ML
    "HR":                 0.22,  # lowest AI exposure — relationship-heavy work
}

# Adoption wave classification — Rogers (1962) diffusion of innovations
# applied to AI skills in the labor market
AI_ADOPTION_WAVE = {
    "Data Science":       "Early Adopter",
    "Software Eng":       "Early Adopter",
    "Research":           "Early Majority",
    "Product":            "Early Majority",
    "Marketing":          "Late Majority",
    "Finance/Accounting": "Late Majority",
    "Operations":         "Laggard",
    "HR":                 "Laggard",
}


def build_skill_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add skill taxonomy features to the worker panel.

    In a real Revelio-style analysis, this function would:
    1. Take raw job posting text or resume text
    2. Run it through a skill extraction NLP model
    3. Map extracted skills to the taxonomy above
    4. Compute intensity scores from actual skill frequencies

    Here I'm taking a shortcut: assigning intensity at the job-family level.
    This is reasonable for a proof-of-concept but would need to be replaced
    with actual skill-level data in production.
    """
    df = df.copy()

    df["ai_intensity"]    = df["job_family"].map(JOB_FAMILY_AI_INTENSITY)
    df["adoption_wave"]   = df["job_family"].map(AI_ADOPTION_WAVE)

    # Cluster by AI exposure — useful for heterogeneous treatment effect analysis
    df["ai_skill_cluster"] = pd.cut(
        df["ai_intensity"],
        bins=[0, 0.35, 0.55, 0.75, 1.0],
        labels=["Low AI Exposure", "Moderate AI Exposure",
                "High AI Exposure", "Core AI Role"]
    )

    industry_ai_exposure = {
        "Technology": 0.72, "Finance": 0.55, "Healthcare": 0.38,
        "Retail": 0.28, "Manufacturing": 0.22, "Consulting": 0.48,
        "Education": 0.31, "Media": 0.44,
    }
    df["industry_ai_exposure"] = df["industry"].map(industry_ai_exposure)

    # Composite AI exposure: weighted average of occupation and industry exposure
    # Weights are somewhat arbitrary — I'd want to validate these with real data
    df["composite_ai_exposure"] = (
        df["ai_intensity"] * 0.6 +
        df["industry_ai_exposure"] * 0.4
    )

    # Skill transition: workers in low-AI roles who adopted AI skills
    # This is an interesting subgroup — they crossed an occupational boundary
    # which probably has different returns than adopting within a high-AI role
    df["skill_transition"] = (
        (df["has_ai_skill"] == 1) &
        (df["ai_intensity"] < 0.55) &
        (df["year"] >= 2023)
    ).astype(int)

    return df


if __name__ == "__main__":
    input_path  = "data/worker_panel.csv"
    output_path = "data/worker_panel_enriched.csv"

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Could not find {input_path}. "
            "Run 01_generate_data.py first."
        )

    df = pd.read_csv(input_path)
    df = build_skill_features(df)
    df.to_csv(output_path, index=False)

    print(f"Enriched panel saved: {df.shape}")
    print(f"\nAI skill cluster distribution (2024):")
    print(df[df.year==2024]["ai_skill_cluster"].value_counts())
    print(f"\nAI adoption rate by year:")
    print(df.groupby("year")["has_ai_skill"].mean().round(3))
    print("\nAdoption goes from 0 pre-shock to ~53% by 2024.")
    print("This mirrors real patterns in LinkedIn skill data post-ChatGPT.")
