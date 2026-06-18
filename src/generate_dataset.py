"""
B2B Revenue Intelligence — Synthetic Dataset Generator
Modeled on real CPaaS/Sinch data structure.

Tables generated:
    1. account_managers   — AMs and their regions
    2. accounts           — one row per client
    3. product_subscriptions — which products each client uses
    4. monthly_metrics    — revenue/volume/GP per client-product-month
    5. account_health     — aggregated signals per client per month (model input)

Author: Thais | Portfolio Project
"""

import pandas as pd
import numpy as np
import random
import json
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────

N_ACCOUNTS      = 500
N_MONTHS        = 12       # months of history
START_DATE      = datetime(2023, 1, 1)

# Sinch Platforms
PLATFORMS = ["Sinch Core", "Sinch Engage", "MessageMedia", "MailerSend"]

# Product Lines & Hierarchy (mirroring Sinch structure)
PRODUCT_CATALOG = {
    "SMS": {
        "hierarchy": ["A2P SMS", "P2P SMS", "SMS API"],
        "base_price_per_unit": 0.04,   # SEK per message
        "base_volume": (50_000, 2_000_000),
    },
    "Voice": {
        "hierarchy": ["Voice API", "SIP Trunking", "Contact Center Voice"],
        "base_price_per_unit": 0.12,
        "base_volume": (10_000, 500_000),
    },
    "Verification": {
        "hierarchy": ["2FA SMS", "Flash Call Verification", "Data Verification"],
        "base_price_per_unit": 0.08,
        "base_volume": (20_000, 800_000),
    },
    "WhatsApp Business": {
        "hierarchy": ["WhatsApp API", "WhatsApp Business Platform"],
        "base_price_per_unit": 0.55,
        "base_volume": (5_000, 200_000),
    },
    "Email": {
        "hierarchy": ["Transactional Email", "Marketing Email", "Email API"],
        "base_price_per_unit": 0.005,
        "base_volume": (100_000, 5_000_000),
    },
    "RCS": {
        "hierarchy": ["RCS Business Messaging", "RCS API"],
        "base_price_per_unit": 0.35,
        "base_volume": (2_000, 100_000),
    },
}

ALL_PRODUCTS = list(PRODUCT_CATALOG.keys())

# Industry verticals & typical product adoption patterns
INDUSTRY_PROFILES = {
    "Fintech": {
        "weight": 0.20,
        "first_products": ["Verification", "SMS"],
        "expansion_products": ["Voice", "WhatsApp Business"],
        "segment_dist": {"Enterprise": 0.35, "Mid-Market": 0.45, "SMB": 0.20},
        "avg_arr_multiplier": 1.4,
    },
    "Retail & E-commerce": {
        "weight": 0.25,
        "first_products": ["SMS", "Email"],
        "expansion_products": ["WhatsApp Business", "RCS"],
        "segment_dist": {"Enterprise": 0.25, "Mid-Market": 0.40, "SMB": 0.35},
        "avg_arr_multiplier": 1.1,
    },
    "Healthcare": {
        "weight": 0.15,
        "first_products": ["SMS", "Verification"],
        "expansion_products": ["Voice", "Email"],
        "segment_dist": {"Enterprise": 0.30, "Mid-Market": 0.50, "SMB": 0.20},
        "avg_arr_multiplier": 1.2,
    },
    "Logistics": {
        "weight": 0.15,
        "first_products": ["SMS", "Voice"],
        "expansion_products": ["WhatsApp Business", "Email"],
        "segment_dist": {"Enterprise": 0.20, "Mid-Market": 0.45, "SMB": 0.35},
        "avg_arr_multiplier": 0.95,
    },
    "Gaming & Entertainment": {
        "weight": 0.10,
        "first_products": ["SMS", "Verification"],
        "expansion_products": ["Email", "Voice"],
        "segment_dist": {"Enterprise": 0.15, "Mid-Market": 0.35, "SMB": 0.50},
        "avg_arr_multiplier": 0.85,
    },
    "Technology & SaaS": {
        "weight": 0.15,
        "first_products": ["SMS", "Email"],
        "expansion_products": ["Verification", "Voice", "WhatsApp Business"],
        "segment_dist": {"Enterprise": 0.30, "Mid-Market": 0.45, "SMB": 0.25},
        "avg_arr_multiplier": 1.3,
    },
}

# Client segments & their ARR ranges (SEK)
SEGMENT_CONFIG = {
    "Enterprise":   {"arr_range": (500_000, 5_000_000), "churn_rate": 0.05, "n_products": (2, 5)},
    "Mid-Market":   {"arr_range": (100_000, 500_000),   "churn_rate": 0.10, "n_products": (2, 4)},
    "SMB":          {"arr_range": (20_000, 100_000),    "churn_rate": 0.18, "n_products": (1, 3)},
}

# Regions
CLIENT_REGIONS = ["Nordics", "DACH", "UK & Ireland", "Southern Europe", "North America", "APAC", "LATAM"]
AM_REGIONS     = ["Nordics & DACH", "UK & Southern Europe", "North America", "APAC & LATAM"]

DESTINATION_COUNTRIES = {
    "Nordics":          ["Sweden", "Norway", "Denmark", "Finland"],
    "DACH":             ["Germany", "Austria", "Switzerland"],
    "UK & Ireland":     ["United Kingdom", "Ireland"],
    "Southern Europe":  ["Spain", "Italy", "France", "Portugal"],
    "North America":    ["United States", "Canada"],
    "APAC":             ["Australia", "Japan", "Singapore", "India"],
    "LATAM":            ["Brazil", "Mexico", "Colombia", "Argentina"],
}

# Fake company name parts
COMPANY_PREFIXES = ["Nord", "Euro", "Global", "Digital", "Smart", "Tech", "Rapid", "Flex", "Cloud", "Open", "Fast", "Prime"]
COMPANY_SUFFIXES = ["Pay", "Health", "Commerce", "Connect", "Retail", "Bank", "Logistics", "Group", "Solutions", "Systems", "Labs", "AI"]
COMPANY_TYPES    = ["AB", "GmbH", "Ltd", "Inc", "SAS", "Oy", "AS", "BV"]

# Fake AM names
AM_NAMES = [
    "Emma Lindgren", "Lars Johansson", "Sofia Bergström", "Marcus Holm",
    "Anna Karlsson", "Erik Nilsson", "Maria Svensson", "Johan Andersson",
    "Olivia Schmidt", "Thomas Müller", "Sara Hansen", "David Cohen",
    "Isabella Ferrari", "James Thompson"
]


# ─── HELPER FUNCTIONS ────────────────────────────────────────────────────────

def fake_company_name(industry):
    prefix = random.choice(COMPANY_PREFIXES)
    suffix = random.choice(COMPANY_SUFFIXES)
    company_type = random.choice(COMPANY_TYPES)
    return f"{prefix}{suffix} {company_type}"

def weighted_choice(options_weights: dict):
    options = list(options_weights.keys())
    weights = list(options_weights.values())
    return random.choices(options, weights=weights, k=1)[0]

def date_n_months_ago(n):
    return (START_DATE + timedelta(days=30 * n)).strftime("%Y-%m-%d")

def generate_mom_trend(base_trend, volatility=0.05, n=12):
    """Generate month-over-month growth rates with a base trend."""
    return [base_trend + np.random.normal(0, volatility) for _ in range(n)]


# ─── TABLE 1: ACCOUNT MANAGERS ───────────────────────────────────────────────

def generate_account_managers():
    rows = []
    for i, name in enumerate(AM_NAMES):
        region_idx = i % len(AM_REGIONS)
        rows.append({
            "am_id":     f"AM{str(i+1).zfill(3)}",
            "am_name":   name,
            "am_region": AM_REGIONS[region_idx],
        })
    return pd.DataFrame(rows)


# ─── TABLE 2: ACCOUNTS ───────────────────────────────────────────────────────

def generate_accounts(am_df):
    rows = []
    industries = list(INDUSTRY_PROFILES.keys())
    industry_weights = [INDUSTRY_PROFILES[i]["weight"] for i in industries]

    for i in range(N_ACCOUNTS):
        client_id = f"CLT{str(i+1).zfill(4)}"

        # Industry
        industry = random.choices(industries, weights=industry_weights, k=1)[0]
        profile  = INDUSTRY_PROFILES[industry]

        # Segment
        segment = weighted_choice(profile["segment_dist"])
        seg_cfg = SEGMENT_CONFIG[segment]

        # Region
        client_region = random.choice(CLIENT_REGIONS)

        # Tenure (months) — enterprise tend to be older clients
        tenure_base = {"Enterprise": (18, 60), "Mid-Market": (6, 36), "SMB": (1, 24)}[segment]
        tenure_months = random.randint(*tenure_base)

        # ARR
        arr_base = random.randint(*seg_cfg["arr_range"])
        arr = int(arr_base * profile["avg_arr_multiplier"])

        # Platform
        platform = random.choice(PLATFORMS)

        # AM assignment (match region loosely)
        region_to_am_region = {
            "Nordics": "Nordics & DACH", "DACH": "Nordics & DACH",
            "UK & Ireland": "UK & Southern Europe", "Southern Europe": "UK & Southern Europe",
            "North America": "North America",
            "APAC": "APAC & LATAM", "LATAM": "APAC & LATAM",
        }
        am_region_target = region_to_am_region.get(client_region, random.choice(AM_REGIONS))
        eligible_ams = am_df[am_df["am_region"] == am_region_target]["am_id"].tolist()
        am_id = random.choice(eligible_ams) if eligible_ams else am_df["am_id"].sample(1).iloc[0]

        # Churn — influenced by segment, tenure (short tenure = higher risk)
        base_churn_prob = seg_cfg["churn_rate"]
        if tenure_months < 6:
            base_churn_prob *= 1.5
        elif tenure_months > 24:
            base_churn_prob *= 0.5
        churned = np.random.random() < base_churn_prob

        # NPS proxy (1-10): correlated with tenure and segment
        nps_base = {"Enterprise": 7.5, "Mid-Market": 6.8, "SMB": 6.2}[segment]
        nps_tenure_bonus = min(tenure_months / 60, 1.0)  # max +1 for long tenure
        nps = round(min(10, max(1, nps_base + nps_tenure_bonus + np.random.normal(0, 0.8))), 1)

        # Contract type
        contract_weights = {"Enterprise": {"Multi-year": 0.5, "Annual": 0.4, "Monthly": 0.1},
                            "Mid-Market": {"Multi-year": 0.2, "Annual": 0.55, "Monthly": 0.25},
                            "SMB":        {"Multi-year": 0.05,"Annual": 0.40, "Monthly": 0.55}}
        contract_type = weighted_choice(contract_weights[segment])

        # Last QBR (Enterprise gets more frequent QBRs)
        qbr_frequency = {"Enterprise": 90, "Mid-Market": 120, "SMB": 180}[segment]
        days_since_qbr = random.randint(0, qbr_frequency)
        last_qbr_date = (datetime.now() - timedelta(days=days_since_qbr)).strftime("%Y-%m-%d")

        # Renewal date
        months_to_renewal = random.randint(1, 12)
        renewal_date = (datetime.now() + timedelta(days=30 * months_to_renewal)).strftime("%Y-%m-%d")

        rows.append({
            "client_id":        client_id,
            "client_name":      fake_company_name(industry),
            "client_segment":   segment,
            "client_region":    client_region,
            "industry_vertical":industry,
            "platform":         platform,
            "am_id":            am_id,
            "tenure_months":    tenure_months,
            "contract_type":    contract_type,
            "arr_sek":          arr,
            "nps_score_proxy":  nps,
            "last_qbr_date":    last_qbr_date,
            "renewal_date":     renewal_date,
            "churned":          churned,
        })

    return pd.DataFrame(rows)


# ─── TABLE 3: PRODUCT SUBSCRIPTIONS ──────────────────────────────────────────

def generate_product_subscriptions(accounts_df):
    rows = []
    for _, acc in accounts_df.iterrows():
        industry  = acc["industry_vertical"]
        segment   = acc["client_segment"]
        profile   = INDUSTRY_PROFILES[industry]
        seg_cfg   = SEGMENT_CONFIG[segment]
        client_id = acc["client_id"]
        region    = acc["client_region"]

        # Determine number of products
        n_products = random.randint(*seg_cfg["n_products"])

        # Build product list: start with industry first products
        selected = list(profile["first_products"])

        # Add expansion products based on tenure
        if acc["tenure_months"] > 12:
            extra = [p for p in profile["expansion_products"] if p not in selected]
            selected += extra[:1]
        if acc["tenure_months"] > 24:
            extra = [p for p in ALL_PRODUCTS if p not in selected]
            selected += extra[:1]

        # Trim or pad to n_products
        random.shuffle(selected)
        selected = selected[:n_products]

        # Destination countries for this client's region
        dest_countries = DESTINATION_COUNTRIES.get(region, ["Global"])

        for product_line in selected:
            cat = PRODUCT_CATALOG[product_line]
            hierarchy = random.choice(cat["hierarchy"])
            adoption_date = (datetime.now() - timedelta(days=30 * acc["tenure_months"] + random.randint(0, 60))).strftime("%Y-%m-%d")

            rows.append({
                "client_id":         client_id,
                "platform":          acc["platform"],
                "product_line":      product_line,
                "product_hierarchy": hierarchy,
                "destination_country": random.choice(dest_countries),
                "adoption_date":     adoption_date,
                "is_active":         not acc["churned"],
            })

    return pd.DataFrame(rows)


# ─── TABLE 4: MONTHLY METRICS ────────────────────────────────────────────────

def generate_monthly_metrics(accounts_df, subscriptions_df):
    rows = []
    for _, acc in accounts_df.iterrows():
        client_id = acc["client_id"]
        segment   = acc["client_segment"]
        churned   = acc["churned"]

        # Get this account's products
        client_subs = subscriptions_df[subscriptions_df["client_id"] == client_id]

        for _, sub in client_subs.iterrows():
            product_line = sub["product_line"]
            cat = PRODUCT_CATALOG[product_line]

            # Base volume and revenue for this account-product
            base_volume  = random.randint(*cat["base_volume"])
            # Enterprise uses more volume
            volume_multiplier = {"Enterprise": 3.0, "Mid-Market": 1.5, "SMB": 0.6}[segment]
            base_volume = int(base_volume * volume_multiplier)

            # Trend: churned accounts show declining trend
            if churned:
                base_trend = random.uniform(-0.08, -0.02)   # declining
            else:
                base_trend = random.uniform(-0.01, 0.06)    # stable to growing

            monthly_trends = generate_mom_trend(base_trend, volatility=0.04, n=N_MONTHS)

            volume = base_volume
            for month_idx in range(N_MONTHS):
                month_date = (START_DATE + timedelta(days=30 * month_idx)).strftime("%Y-%m")

                # Apply trend
                volume = max(100, int(volume * (1 + monthly_trends[month_idx])))

                revenue      = round(volume * cat["base_price_per_unit"], 2)
                gross_profit = round(revenue * random.uniform(0.35, 0.60), 2)  # 35-60% GP margin

                rows.append({
                    "client_id":           client_id,
                    "product_line":        product_line,
                    "product_hierarchy":   sub["product_hierarchy"],
                    "destination_country": sub["destination_country"],
                    "month":               month_date,
                    "volume":              volume,
                    "revenue_sek":         revenue,
                    "gross_profit_sek":    gross_profit,
                })

    return pd.DataFrame(rows)


# ─── TABLE 5: ACCOUNT HEALTH (Model Input) ───────────────────────────────────

def generate_account_health(accounts_df, subscriptions_df, metrics_df):
    """
    Aggregated signals per client per month — this is the feature table
    that feeds directly into the ML models.
    """
    rows = []

    for _, acc in accounts_df.iterrows():
        client_id = acc["client_id"]
        segment   = acc["client_segment"]
        churned   = acc["churned"]
        nps       = acc["nps_score_proxy"]
        tenure    = acc["tenure_months"]

        client_metrics = metrics_df[metrics_df["client_id"] == client_id]
        client_subs    = subscriptions_df[subscriptions_df["client_id"] == client_id]
        active_products = client_subs["product_line"].nunique()
        product_lines   = client_subs["product_line"].tolist()

        months = sorted(client_metrics["month"].unique())

        prev_revenue = None
        for month in months:
            month_data = client_metrics[client_metrics["month"] == month]

            total_revenue     = round(month_data["revenue_sek"].sum(), 2)
            total_gp          = round(month_data["gross_profit_sek"].sum(), 2)
            total_volume      = int(month_data["volume"].sum())
            gp_margin         = round(total_gp / total_revenue, 3) if total_revenue > 0 else 0

            # MoM revenue growth
            if prev_revenue and prev_revenue > 0:
                mom_revenue_growth = round((total_revenue - prev_revenue) / prev_revenue, 4)
            else:
                mom_revenue_growth = 0.0
            prev_revenue = total_revenue

            # Synthetic engagement signals
            support_tickets  = max(0, int(np.random.poisson(2 if not churned else 4)))
            days_since_contact = random.randint(1, 90) if not churned else random.randint(30, 180)

            # ── NBA Label Logic (business rules based on real AM experience) ──
            # Upsell: healthy account, growing revenue, not using all products
            if (mom_revenue_growth > 0.03 and nps >= 7
                    and active_products < 4 and not churned):
                nba_label = "Upsell"

            # Expand: large account already using multiple products, stable
            elif (segment == "Enterprise" and active_products >= 3
                  and mom_revenue_growth >= 0 and not churned):
                nba_label = "Expand"

            # Retain: declining revenue or low NPS — intervention needed
            elif (mom_revenue_growth < -0.05 or nps < 6
                  or support_tickets > 4 or churned):
                nba_label = "Retain"

            # Nurture: stable but not ready for upsell
            else:
                nba_label = "Nurture"

            # ── LTV Estimate (12-month forward-looking) ──
            ltv_12m = round(total_revenue * 12 * (1 + max(0, mom_revenue_growth)), 2)

            rows.append({
                "client_id":            client_id,
                "month":                month,
                "client_segment":       segment,
                "client_region":        acc["client_region"],
                "industry_vertical":    acc["industry_vertical"],
                "tenure_months":        tenure,
                "active_products_count":active_products,
                "product_lines":        "|".join(product_lines),
                "total_revenue_sek":    total_revenue,
                "total_gp_sek":         total_gp,
                "gp_margin":            gp_margin,
                "total_volume":         total_volume,
                "mom_revenue_growth":   mom_revenue_growth,
                "nps_score_proxy":      nps,
                "support_tickets_30d":  support_tickets,
                "days_since_last_contact": days_since_contact,
                "contract_type":        acc["contract_type"],
                "months_to_renewal":    random.randint(1, 12),
                "churned":              churned,
                "nba_label":            nba_label,
                "ltv_12m_estimate_sek": ltv_12m,
            })

    return pd.DataFrame(rows)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating B2B CPaaS synthetic dataset...")
    print(f"  Accounts: {N_ACCOUNTS} | Months: {N_MONTHS}")

    print("\n[1/5] Account Managers...")
    am_df = generate_account_managers()
    am_df.to_csv("data/raw/am.csv", index=False)
    print(f"      → {len(am_df)} rows")

    print("[2/5] Accounts...")
    acc_df = generate_accounts(am_df)
    acc_df.to_csv("data/raw/accounts.csv", index=False)
    print(f"      → {len(acc_df)} rows")

    print("[3/5] Product Subscriptions...")
    subs_df = generate_product_subscriptions(acc_df)
    subs_df.to_csv("data/raw/product_subscriptions.csv", index=False)
    print(f"      → {len(subs_df)} rows")

    print("[4/5] Monthly Metrics...")
    metrics_df = generate_monthly_metrics(acc_df, subs_df)
    metrics_df.to_csv("data/raw/monthly_metrics.csv", index=False)
    print(f"      → {len(metrics_df)} rows")

    print("[5/5] Account Health (model input)...")
    health_df = generate_account_health(acc_df, subs_df, metrics_df)
    health_df.to_csv("data/raw/account_health.csv", index=False)
    print(f"      → {len(health_df)} rows")

    # ── Quick Validation ──
    print("\n── Dataset Summary ──────────────────────────────────")
    print(f"Segments:   {acc_df['client_segment'].value_counts().to_dict()}")
    print(f"Industries: {acc_df['industry_vertical'].value_counts().to_dict()}")
    print(f"Churn rate: {acc_df['churned'].mean():.1%}")
    print(f"Products:   {subs_df['product_line'].value_counts().to_dict()}")
    print(f"NBA labels: {health_df[health_df['month'] == health_df['month'].max()]['nba_label'].value_counts().to_dict()}")
    print(f"Avg ARR:    SEK {acc_df['arr_sek'].mean():,.0f}")
    print(f"Avg tenure: {acc_df['tenure_months'].mean():.1f} months")
    print("\n✅ All tables saved to CSV.")
