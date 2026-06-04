"""
generate_dataset.py
Generates synthetic B2B customer dataset for revenue intelligence modeling.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

random.seed(42)
np.random.seed(42)

# --- Config ---
N_CLIENTS = 500
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')

# --- Helpers ---
def random_date(start, end):
      delta = end - start
      return start + timedelta(days=random.randint(0, delta.days))

# --- Generate ---
start_date = datetime(2021, 1, 1)
end_date = datetime(2024, 12, 31)

segments = ['Enterprise', 'Mid-Market', 'SMB']
industries = ['Finance', 'Healthcare', 'Retail', 'Tech', 'Manufacturing', 'Logistics']
csm_list = [f'CSM_{i}' for i in range(1, 11)]

records = []
for i in range(N_CLIENTS):
      segment = random.choice(segments)
      mrr_ranges = {
          'Enterprise': (5000, 50000),
          'Mid-Market': (1000, 10000),
          'SMB': (100, 2000),
      }
      lo, hi = mrr_ranges[segment]
      mrr = np.random.uniform(lo, hi)

    health_score = np.random.beta(5, 2) * 100
    nps = np.random.choice(range(0, 11), p=[0.02]*6 + [0.05, 0.1, 0.15, 0.2, 0.42])
    tenure_days = random.randint(30, 1500)
    last_login_days_ago = random.randint(0, 60)
    support_tickets = np.random.poisson(3)
    products_used = random.randint(1, 8)
    exec_sponsor = random.choice([True, False])
    churned = 1 if (health_score < 30 and random.random() > 0.4) else 0
    upsell_potential = round(mrr * np.random.uniform(0.1, 1.5), 2)

    records.append({
              'client_id': f'CLIENT_{i+1:04d}',
              'segment': segment,
              'industry': random.choice(industries),
              'csm': random.choice(csm_list),
              'mrr': round(mrr, 2),
              'health_score': round(health_score, 2),
              'nps': nps,
              'tenure_days': tenure_days,
              'last_login_days_ago': last_login_days_ago,
              'support_tickets_last_90d': support_tickets,
              'products_used': products_used,
              'has_exec_sponsor': int(exec_sponsor),
              'upsell_potential_usd': upsell_potential,
              'churned': churned,
              'contract_start': random_date(start_date, end_date).strftime('%Y-%m-%d'),
    })

df = pd.DataFrame(records)

os.makedirs(OUTPUT_DIR, exist_ok=True)
output_path = os.path.join(OUTPUT_DIR, 'clients.csv')
df.to_csv(output_path, index=False)
print(f"Dataset saved to {output_path} ({len(df)} rows)")
