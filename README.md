# B2B Revenue Intelligence
### White Space Analysis + Next Best Action for Enterprise CPaaS

> *"A B2B Revenue Intelligence system that uses ML and LLMs to transform client data into concrete commercial actions — reducing reliance on intuition and increasing revenue expansion potential in enterprise portfolios."*

---

## The Problem

Account Managers in enterprise CPaaS companies manage large client portfolios **without data-driven visibility** to decide where to focus.

Two critical pain points that cost revenue daily:

**1. Invisible White Space**
AMs don't know which products to recommend based on each client's profile. Decisions are made by intuition — not data. A fintech client using SMS should likely adopt Verification/2FA, but without a model this opportunity stays invisible.

**2. No Intelligent Prioritization**
With dozens of accounts to manage, the AM doesn't know where to focus. Who has the highest expansion potential? Who is silently at risk? Without data, the AM reacts — instead of acting proactively.

---

## The Solution

A three-model Revenue Intelligence system, all connected in a single pipeline:

| Model | Type | Output |
|---|---|---|
| **White Space Analyzer** | Association Rules + Collaborative Filtering | Ranked product recommendations per account |
| **Next Best Action (NBA)** | Multi-class Classification + SHAP | Upsell / Expand / Retain / Nurture label + reasoning |
| **LTV Predictor** | LightGBM Regression | Estimated 12-month revenue per account |

All three models feed into:
- A **Streamlit App** with an **AI Copilot** (LLM-powered chat over portfolio data)
- A **Tableau Dashboard** for portfolio-level visibility

---

## Project Structure

```
b2b-revenue-intelligence/
│
├── data/
│   ├── raw/                        ← generated CSVs
│   └── processed/                  ← post feature engineering
│
├── notebooks/
│   ├── 01_EDA.ipynb                ← exploratory analysis with business narrative
│   ├── 02_white_space_model.ipynb  ← association rules + collaborative filtering
│   ├── 03_nba_ltv_models.ipynb     ← NBA classifier + LTV regression + SHAP
│   └── 04_pipeline.ipynb           ← end-to-end inference pipeline
│
├── app/
│   └── streamlit_app.py            ← interactive app + AM Copilot
│
├── models/                         ← saved model files (.pkl)
│
├── src/
│   ├── generate_dataset.py         ← synthetic dataset generator
│   └── utils.py                    ← shared helper functions
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## The Dataset

The dataset is **100% synthetic**, built from scratch to reflect real B2B CPaaS enterprise dynamics — modeled on the author's 4 years of experience in enterprise sales at a CPaaS company.

Every design decision is anchored in real business logic:

| Feature | Logic |
|---|---|
| Churn rate | 5% Enterprise / 10% Mid-Market / 18% SMB — aligned with B2B SaaS benchmarks |
| Product adoption | Fintechs adopt Verification first, then SMS. Retail leads with SMS, then WhatsApp |
| NBA labels | Rule-based labeling grounded in real AM experience (documented in notebook) |
| LTV estimate | 12-month forward projection based on MoM revenue trend |

**5 tables, fully relational:**

| Table | Rows | Description |
|---|---|---|
| `account_managers.csv` | 14 | AMs with name and region |
| `accounts.csv` | 500 | One row per client — segment, region, industry, ARR, tenure |
| `product_subscriptions.csv` | ~1,200 | Which products each client uses, on which platform, to which country |
| `monthly_metrics.csv` | ~15,000 | Revenue, gross profit and volume per client-product-month |
| `account_health.csv` | 5,000 | Aggregated feature table — direct input to ML models |

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data Generation | Python, NumPy, Pandas |
| ML Models | Scikit-learn, LightGBM, mlxtend |
| Explainability | SHAP |
| App | Streamlit |
| AI Copilot | Claude API / OpenAI GPT-4o |
| Dashboard | Tableau Public |
| Version Control | GitHub |

---

## Business Impact

Applied to a portfolio of **500 enterprise accounts**, this system:
- Surfaces upsell opportunities that are invisible without data
- Prioritizes AM actions by estimated LTV — highest value accounts first
- Delivers account-level AI narratives that any AM can act on immediately, without SQL

---

## About This Project

This project was built as a **portfolio capstone** combining:
- 4 years of enterprise B2B sales and account management experience at a CPaaS company
- Data Science skills in ML, NLP, and model deployment
- Domain expertise in the CPaaS/cloud communications market

The problems it solves are real. The author experienced them firsthand as an Account Manager.

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/b2b-revenue-intelligence.git
cd b2b-revenue-intelligence

# Install dependencies
pip install -r requirements.txt

# Generate the dataset
python src/generate_dataset.py

# Launch the Streamlit app
streamlit run app/streamlit_app.py
```

---

## Project Status

| Week | Focus | Status |
|---|---|---|
| 1 | Data Strategy + Dataset Design | ✅ Complete |
| 2 | EDA with Business Narrative | 🔄 In progress |
| 3 | White Space Model | ⏳ Upcoming |
| 4 | NBA Classifier + LTV Predictor | ⏳ Upcoming |
| 5 | Inference Pipeline + Business Impact | ⏳ Upcoming |
| 6 | Streamlit App + AI Copilot | ⏳ Upcoming |
| 7 | Tableau Dashboard + Polish | ⏳ Upcoming |

---

*Built by Thais — Data Analyst transitioning into Data Science, with a commercial background in B2B SaaS.*
