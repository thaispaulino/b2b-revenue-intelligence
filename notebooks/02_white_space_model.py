"""
B2B Revenue Intelligence — White Space Model
Week 3: Association Rules + Collaborative Filtering

No external ML libraries required beyond pandas, numpy, sklearn.

HOW TO USE IN COLAB:
    !git clone https://github.com/thaispaulino/b2b-revenue-intelligence.git
    %cd b2b-revenue-intelligence
    !python src/generate_dataset.py
    exec(open('notebooks/02_white_space_model.py').read())

OUTPUT:
    data/processed/white_space_recommendations.csv
    data/processed/association_rules.csv
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from itertools import combinations
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings('ignore')

# ── Style ──────────────────────────────────────────────────────────────────
NAVY    = '#1E2761'
TEAL    = '#028090'
MINT    = '#02C39A'
ACCENT  = '#00B4D8'
GRAY    = '#64748B'
PALETTE = [NAVY, TEAL, MINT, ACCENT, '#F4A261', '#E76F51']

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.spines.top': False,    'axes.spines.right': False,
    'axes.labelcolor': NAVY,     'axes.titlecolor': NAVY,
    'axes.titlesize': 13,        'axes.titleweight': 'bold',
    'font.family': 'sans-serif', 'xtick.color': GRAY, 'ytick.color': GRAY,
})

DATA_PATH = 'data/raw'
OUT_PATH  = 'data/processed'
os.makedirs(OUT_PATH, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("Loading data...")
acc  = pd.read_csv(f'{DATA_PATH}/accounts.csv')
subs = pd.read_csv(f'{DATA_PATH}/product_subscriptions.csv')

ALL_PRODUCTS = sorted(subs['product_line'].unique().tolist())
basket = pd.crosstab(subs['client_id'], subs['product_line']).astype(bool)

print(f"  Accounts:    {len(acc):,}")
print(f"  Products:    {ALL_PRODUCTS}")
print(f"  Basket:      {basket.shape[0]} × {basket.shape[1]}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — ASSOCIATION RULES (manual Apriori — no external library needed)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  PART 1 — ASSOCIATION RULES (Apriori)")
print("=" * 60)

MIN_SUPPORT    = 0.10
MIN_CONFIDENCE = 0.40

def get_support(itemset, basket):
    return basket[list(itemset)].all(axis=1).sum() / len(basket)

print(f"\n  Settings: min_support={MIN_SUPPORT}, min_confidence={MIN_CONFIDENCE}")
print("\n  Product adoption rates:")
for prod in ALL_PRODUCTS:
    sup = get_support([prod], basket)
    bar = '█' * int(sup * 25)
    print(f"    {prod:25s} {bar} {sup:.1%}")

# Generate all rules
print("\n  Generating association rules...")
rules_list = []
for r in range(1, len(ALL_PRODUCTS)):
    for antecedent in combinations(ALL_PRODUCTS, r):
        ant_sup = get_support(antecedent, basket)
        if ant_sup < MIN_SUPPORT:
            continue
        remaining = [p for p in ALL_PRODUCTS if p not in antecedent]
        for consequent in remaining:
            both_sup = get_support(list(antecedent) + [consequent], basket)
            if both_sup < MIN_SUPPORT:
                continue
            confidence = both_sup / ant_sup
            if confidence >= MIN_CONFIDENCE:
                lift = confidence / get_support([consequent], basket)
                rules_list.append({
                    'antecedents': ' + '.join(antecedent),
                    'consequents': consequent,
                    'support':    round(both_sup, 4),
                    'confidence': round(confidence, 4),
                    'lift':       round(lift, 4),
                })

rules_df = pd.DataFrame(rules_list).sort_values('lift', ascending=False).reset_index(drop=True)
rules_df.to_csv(f'{OUT_PATH}/association_rules.csv', index=False)

print(f"  Found {len(rules_df)} rules")
print()
print(f"  {'IF client uses':<35} {'→ ALSO likely':<25} {'Confidence':>10} {'Lift':>8}")
print("  " + "-" * 82)
for _, row in rules_df.head(10).iterrows():
    print(f"  {row['antecedents']:<35} {row['consequents']:<25} {row['confidence']:>9.1%} {row['lift']:>7.2f}x")


# ── Visualizations ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Association Rules — Product Co-Adoption Patterns',
             fontsize=15, fontweight='bold', color=NAVY, y=1.02)

sc = axes[0].scatter(rules_df['confidence'], rules_df['lift'],
                     c=rules_df['support'], cmap='YlOrRd', s=120,
                     alpha=0.8, edgecolors='white')
plt.colorbar(sc, ax=axes[0], label='Support')
axes[0].axhline(1, color=GRAY, linestyle='--', linewidth=1, label='Lift=1 (random)')
axes[0].set_xlabel('Confidence')
axes[0].set_ylabel('Lift')
axes[0].set_title('Confidence vs Lift')
axes[0].legend(fontsize=9)

top8   = rules_df.head(8)
labels = [f"{a}\n→ {c}" for a, c in zip(top8['antecedents'], top8['consequents'])]
axes[1].barh(labels[::-1], top8['confidence'].values[::-1], color=PALETTE[:8])
axes[1].set_xlabel('Confidence')
axes[1].set_title('Top 8 Rules by Lift')
axes[1].set_xlim(0, 1.1)
for i, val in enumerate(top8['confidence'].values[::-1]):
    axes[1].text(val + 0.01, i, f'{val:.0%}', va='center', fontsize=10, color=NAVY)

plt.tight_layout()
plt.savefig(f'{OUT_PATH}/09_association_rules.png', dpi=150, bbox_inches='tight')
plt.show()

# Co-occurrence heatmap
co  = basket.astype(int).T.dot(basket.astype(int))
co_pct = (co / np.diag(co.values)).round(2) * 100

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(co_pct, annot=True, fmt='.0f', cmap='Blues',
            linewidths=0.5, annot_kws={'size': 13},
            cbar_kws={'label': '% of row-product accounts also using column-product'})
ax.set_title('Product Co-Adoption Matrix\n(row → column adoption rate %)',
             fontweight='bold', color=NAVY)
ax.tick_params(axis='x', rotation=20)
plt.tight_layout()
plt.savefig(f'{OUT_PATH}/10_cooccurrence_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n  💡 KEY PATTERNS:")
for _, row in rules_df.head(3).iterrows():
    print(f"    {row['antecedents']} → {row['consequents']}")
    print(f"    {row['confidence']:.0%} confidence | {row['lift']:.1f}x more likely than random")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — COLLABORATIVE FILTERING
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  PART 2 — COLLABORATIVE FILTERING")
print("=" * 60)
print("\n  Finding similar accounts → recommend what they use")

le      = LabelEncoder()
acc_enc = acc[['client_id','client_segment','industry_vertical',
               'client_region','tenure_months','arr_sek','nps_score_proxy']].copy()
for col in ['client_segment','industry_vertical','client_region']:
    acc_enc[col] = le.fit_transform(acc_enc[col])

scaler         = MinMaxScaler()
feature_matrix = scaler.fit_transform(acc_enc.drop('client_id', axis=1))
sim_matrix     = cosine_similarity(feature_matrix)
sim_df         = pd.DataFrame(sim_matrix,
                               index=acc['client_id'].values,
                               columns=acc['client_id'].values)
print(f"  Similarity matrix computed: {sim_df.shape}")

def get_cf_recommendations(client_id, top_k=15):
    current = set(basket.columns[basket.loc[client_id]].tolist()) \
              if client_id in basket.index else set()
    similar = sim_df[client_id].drop(client_id).sort_values(ascending=False).head(top_k)
    scores  = {}
    for product in ALL_PRODUCTS:
        if product in current or product not in basket.columns:
            continue
        w_score = sum(sim * int(basket.loc[cid, product])
                      for cid, sim in similar.items() if cid in basket.index)
        w_total = similar.sum()
        scores[product] = round(w_score / w_total, 4) if w_total > 0 else 0.0
    return scores

print("  Generating recommendations for all accounts...")
cf_rows = []
for cid in acc['client_id'].values:
    for product, score in get_cf_recommendations(cid).items():
        cf_rows.append({'client_id': cid, 'recommended_product': product, 'cf_score': score})

cf_df = pd.DataFrame(cf_rows)
print(f"  Generated {len(cf_df):,} recommendations")


# ─────────────────────────────────────────────────────────────────────────────
# PART 3 — ENSEMBLE: COMBINE BOTH MODELS
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("  PART 3 — ENSEMBLE (60% CF + 40% Rules)")
print("=" * 60)

def get_rule_score(client_id, product):
    if client_id not in basket.index:
        return 0.0
    current    = frozenset(basket.columns[basket.loc[client_id]].tolist())
    best_conf  = 0.0
    for _, rule in rules_df.iterrows():
        ant = frozenset(rule['antecedents'].split(' + '))
        con = rule['consequents']
        if con == product and ant <= current:
            best_conf = max(best_conf, rule['confidence'])
    return round(best_conf, 4)

print("  Computing rule scores...")
cf_df['rule_score']     = cf_df.apply(lambda r: get_rule_score(r['client_id'], r['recommended_product']), axis=1)
cf_df['ensemble_score'] = (0.60 * cf_df['cf_score'] + 0.40 * cf_df['rule_score']).round(4)
cf_df['confidence_label'] = pd.cut(cf_df['ensemble_score'],
                                    bins=[-0.01, 0.35, 0.60, 1.01],
                                    labels=['Low','Medium','High'])
cf_df = cf_df.merge(acc[['client_id','client_segment','industry_vertical','client_region']], on='client_id')
cf_df = cf_df.sort_values(['client_id','ensemble_score'], ascending=[True, False])
cf_df.to_csv(f'{OUT_PATH}/white_space_recommendations.csv', index=False)

print(f"  High confidence:   {(cf_df['confidence_label']=='High').sum():,}")
print(f"  Medium confidence: {(cf_df['confidence_label']=='Medium').sum():,}")
print(f"  Low confidence:    {(cf_df['confidence_label']=='Low').sum():,}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('White Space Recommendations — Portfolio View',
             fontsize=15, fontweight='bold', color=NAVY, y=1.02)

top_prods = cf_df[cf_df['confidence_label'].isin(['High','Medium'])] \
    .groupby('recommended_product').size().sort_values()
axes[0].barh(top_prods.index, top_prods.values, color=PALETTE[:len(top_prods)])
axes[0].set_title('Expansion Opportunities by Product\n(High + Medium confidence)')
axes[0].set_xlabel('Number of accounts')
for i, val in enumerate(top_prods.values):
    axes[0].text(val + 0.5, i, str(val), va='center', fontsize=11, color=NAVY)

pivot = cf_df.groupby(['industry_vertical','recommended_product'])['ensemble_score'].mean().unstack(fill_value=0)
sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd',
            linewidths=0.5, ax=axes[1], annot_kws={'size': 11})
axes[1].set_title('Avg Score by Industry × Product')
axes[1].set_xlabel('Product')
axes[1].set_ylabel('')
axes[1].tick_params(axis='x', rotation=20)

plt.tight_layout()
plt.savefig(f'{OUT_PATH}/11_white_space_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# DEMO — SINGLE ACCOUNT
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("  DEMO — SINGLE ACCOUNT RECOMMENDATIONS")
print("=" * 60)

demo = acc[(acc['industry_vertical']=='Fintech') & (acc['client_segment']=='Enterprise')].iloc[0]
demo_id   = demo['client_id']
demo_prod = subs[subs['client_id'] == demo_id]['product_line'].tolist()
demo_recs = cf_df[cf_df['client_id'] == demo_id].sort_values('ensemble_score', ascending=False)

print(f"\n  Account:  {demo['client_name']}")
print(f"  Segment:  {demo['client_segment']} | Industry: {demo['industry_vertical']}")
print(f"  Tenure:   {demo['tenure_months']} months | ARR: SEK {demo['arr_sek']:,.0f}")
print(f"  Current products: {demo_prod}")
print()
print(f"  {'Product':<25} {'CF Score':>10} {'Rule Score':>12} {'Final':>10} {'Confidence':>12}")
print("  " + "-" * 72)
for _, row in demo_recs.iterrows():
    print(f"  {row['recommended_product']:<25} {row['cf_score']:>10.3f} "
          f"{row['rule_score']:>12.3f} {row['ensemble_score']:>10.3f} "
          f"{str(row['confidence_label']):>12}")


# ─────────────────────────────────────────────────────────────────────────────
# LEAVE-ONE-OUT EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("  MODEL EVALUATION — Leave-One-Out (100 accounts)")
print("=" * 60)

hits, total = 0, 0
for cid in acc['client_id'].values[:100]:
    if cid not in basket.index: continue
    prods = basket.columns[basket.loc[cid]].tolist()
    if len(prods) < 2: continue

    hidden        = prods[-1]
    masked_basket = basket.copy()
    masked_basket.loc[cid, hidden] = False

    similar = sim_df[cid].drop(cid).sort_values(ascending=False).head(15)
    w_total = similar.sum()
    current = set(masked_basket.columns[masked_basket.loc[cid]].tolist())

    all_scores = {}
    for p in ALL_PRODUCTS:
        if p in current: continue
        ws = sum(sim * int(masked_basket.loc[c2, p])
                 for c2, sim in similar.items() if c2 in masked_basket.index)
        all_scores[p] = ws / w_total if w_total > 0 else 0

    if all_scores and hidden == max(all_scores, key=all_scores.get):
        hits += 1
    total += 1

print(f"\n  Hit rate (top-1):    {hits/total:.1%}  ({hits}/{total} accounts)")
print("\n  💡 Hit rate = model correctly ranked the hidden product")
print("     as the #1 recommendation using only similar accounts data.")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
top_opp = cf_df[cf_df['confidence_label']=='High'].groupby('recommended_product').size().idxmax()
top_ind = cf_df[cf_df['confidence_label']=='High'].groupby('industry_vertical').size().idxmax()

print()
print("=" * 60)
print("  WHITE SPACE MODEL — COMPLETE")
print("=" * 60)
print(f"""
📐 ASSOCIATION RULES
   {len(rules_df)} rules | Top: {rules_df.iloc[0]['antecedents']} → {rules_df.iloc[0]['consequents']}
   ({rules_df.iloc[0]['confidence']:.0%} confidence, {rules_df.iloc[0]['lift']:.1f}x lift)

🤝 COLLABORATIVE FILTERING
   Cosine similarity across {len(acc)} accounts
   Features: segment, industry, region, tenure, ARR, NPS

🎯 ENSEMBLE OUTPUT
   {len(cf_df):,} recommendations | {(cf_df['confidence_label']=='High').sum():,} high-confidence
   Best opportunity: {top_opp}
   Most opportunistic industry: {top_ind}

📁 FILES SAVED
   white_space_recommendations.csv
   association_rules.csv
""")
print("  White Space Model ✅ — Next: 03_nba_ltv_models.py")
print("=" * 60)
