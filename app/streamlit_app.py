"""
B2B Revenue Intelligence — Streamlit App
Week 6: Interactive dashboard + AI Narrative + AM Copilot

HOW TO RUN:
    pip install streamlit anthropic
    streamlit run app/streamlit_app.py

DEPLOY ON STREAMLIT CLOUD:
    1. Push to GitHub
    2. Go to share.streamlit.io
    3. Connect repo → app/streamlit_app.py
    4. Add ANTHROPIC_API_KEY in Secrets

API KEY:
    Set via .streamlit/secrets.toml:
        ANTHROPIC_API_KEY = "your-key-here"
    Or via environment variable:
        export ANTHROPIC_API_KEY="your-key-here"
"""

import os
import sys
import json
import warnings
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from itertools import combinations

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="B2B Revenue Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Colors ────────────────────────────────────────────────────────────────────
NAVY   = '#1E2761'
TEAL   = '#028090'
MINT   = '#02C39A'
ACCENT = '#00B4D8'
GRAY   = '#64748B'
NBA_COLORS = {'Upsell': '#02C39A', 'Expand': '#028090', 'Nurture': '#00B4D8', 'Retain': '#E76F51'}

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
    /* Main background */
    .stApp {{ background-color: #F0F4F8; }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {NAVY};
    }}
    [data-testid="stSidebar"] * {{ color: white !important; }}
    [data-testid="stSidebar"] .stSelectbox label {{ color: #A0AEC0 !important; }}

    /* Header */
    .app-header {{
        background: linear-gradient(135deg, {NAVY} 0%, {TEAL} 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }}
    .app-header h1 {{ color: white; margin: 0; font-size: 1.8rem; }}
    .app-header p  {{ color: #A0AEC0; margin: 0.3rem 0 0; font-size: 0.95rem; }}

    /* Metric cards */
    .metric-card {{
        background: white;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid {TEAL};
        margin-bottom: 0.8rem;
    }}
    .metric-card.upsell  {{ border-color: #02C39A; }}
    .metric-card.expand  {{ border-color: #028090; }}
    .metric-card.retain  {{ border-color: #E76F51; }}
    .metric-card.nurture {{ border-color: #00B4D8; }}
    .metric-label {{ font-size: 0.78rem; color: {GRAY}; font-weight: 600;
                     text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.3rem; }}
    .metric-value {{ font-size: 1.6rem; font-weight: 700; color: {NAVY}; line-height: 1.1; }}
    .metric-sub   {{ font-size: 0.82rem; color: {GRAY}; margin-top: 0.2rem; }}

    /* Section headers */
    .section-header {{
        font-size: 0.75rem; font-weight: 700; color: {TEAL};
        text-transform: uppercase; letter-spacing: 0.1em;
        margin: 1.2rem 0 0.6rem; padding-bottom: 0.3rem;
        border-bottom: 2px solid {TEAL};
    }}

    /* NBA badge */
    .nba-badge {{
        display: inline-block; padding: 0.3rem 0.9rem;
        border-radius: 20px; font-weight: 700;
        font-size: 0.85rem; color: white;
    }}

    /* AI narrative box */
    .ai-narrative {{
        background: linear-gradient(135deg, #F8FAFF 0%, #EEF2FF 100%);
        border: 1px solid #C7D2FE; border-radius: 10px;
        padding: 1.2rem 1.5rem; margin: 0.8rem 0;
        font-size: 0.93rem; line-height: 1.7; color: {NAVY};
    }}

    /* Product tag */
    .product-tag {{
        display: inline-block; background: {TEAL}20;
        color: {TEAL}; border: 1px solid {TEAL}40;
        border-radius: 6px; padding: 0.15rem 0.6rem;
        font-size: 0.78rem; font-weight: 600; margin: 0.15rem;
    }}

    /* WS recommendation */
    .ws-rec {{
        background: white; border-radius: 8px;
        padding: 0.8rem 1rem; margin: 0.4rem 0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        display: flex; align-items: center;
    }}

    /* Chat */
    .chat-user {{ background: {NAVY}10; border-radius: 10px;
                  padding: 0.8rem 1rem; margin: 0.4rem 0; }}
    .chat-ai   {{ background: white; border-radius: 10px;
                  padding: 0.8rem 1rem; margin: 0.4rem 0;
                  border-left: 3px solid {TEAL}; }}

    /* Priority bar */
    .priority-bar-bg {{
        background: #E2E8F0; border-radius: 4px;
        height: 8px; margin-top: 0.4rem;
    }}
    .priority-bar-fill {{
        background: linear-gradient(90deg, {TEAL}, {MINT});
        border-radius: 4px; height: 8px;
    }}

    button[data-testid="baseButton-primary"] {{
        background-color: {TEAL} !important;
        border: none !important;
    }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA & MODEL LOADING (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    acc     = pd.read_csv('data/raw/accounts.csv')
    subs    = pd.read_csv('data/raw/product_subscriptions.csv')
    health  = pd.read_csv('data/raw/account_health.csv')
    metrics = pd.read_csv('data/raw/monthly_metrics.csv')
    latest  = health[health['month'] == health['month'].max()].copy()
    basket  = pd.crosstab(subs['client_id'], subs['product_line']).astype(bool)
    return acc, subs, health, metrics, latest, basket

@st.cache_resource
def load_models(latest):
    FEATURES = [
        'tenure_months', 'active_products_count', 'total_revenue_sek',
        'mom_revenue_growth', 'nps_score_proxy', 'support_tickets_30d',
        'days_since_last_contact', 'months_to_renewal', 'gp_margin',
        'client_segment_enc', 'industry_vertical_enc',
        'client_region_enc', 'contract_type_enc',
    ]

    def encode(df):
        le  = LabelEncoder()
        out = df.copy()
        for col in ['client_segment', 'client_region', 'industry_vertical', 'contract_type']:
            out[f'{col}_enc'] = le.fit_transform(out[col])
        return out[FEATURES]

    X = encode(latest)

    if os.path.exists('models/nba_classifier.pkl'):
        nba = joblib.load('models/nba_classifier.pkl')
        ltv = joblib.load('models/ltv_predictor.pkl')
    else:
        os.makedirs('models', exist_ok=True)
        nba = RandomForestClassifier(n_estimators=200, max_depth=8,
                                      class_weight='balanced', random_state=42)
        nba.fit(X, latest['nba_label'])
        ltv = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                         max_depth=5, random_state=42)
        ltv.fit(X, latest['ltv_12m_estimate_sek'])
        joblib.dump(nba, 'models/nba_classifier.pkl')
        joblib.dump(ltv, 'models/ltv_predictor.pkl')

    return nba, ltv, encode, FEATURES

@st.cache_data
def compute_rules(basket_dict, all_products):
    basket = pd.DataFrame(basket_dict)
    MIN_S, MIN_C = 0.10, 0.40
    def sup(items): return basket[list(items)].all(axis=1).sum() / len(basket)
    rules = []
    for r in range(1, len(all_products)):
        for ant in combinations(all_products, r):
            as_ = sup(ant)
            if as_ < MIN_S: continue
            for con in all_products:
                if con in ant: continue
                bs = sup(list(ant) + [con])
                if bs < MIN_S: continue
                conf = bs / as_
                if conf >= MIN_C:
                    rules.append({'antecedents': set(ant), 'consequent': con,
                                  'confidence': conf, 'lift': conf / sup([con])})
    return rules


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def run_inference(client_id, acc, subs, latest, basket, nba_model, ltv_model,
                  encode_fn, rules, all_products):
    acc_row    = acc[acc['client_id'] == client_id].iloc[0]
    health_row = latest[latest['client_id'] == client_id].iloc[0]
    current    = subs[subs['client_id'] == client_id]['product_line'].tolist()

    # White Space
    ws = []
    for product in all_products:
        if product in current: continue
        score = max(
            (r['confidence'] for r in rules
             if r['consequent'] == product and r['antecedents'] <= set(current)),
            default=0.0
        )
        ws.append({'product': product, 'score': round(score, 3),
                   'label': 'High' if score > 0.6 else 'Medium' if score > 0.35 else 'Low'})
    ws = sorted(ws, key=lambda x: x['score'], reverse=True)

    # NBA + LTV
    feat         = encode_fn(pd.DataFrame([health_row]))
    nba_pred     = nba_model.predict(feat)[0]
    nba_proba    = nba_model.predict_proba(feat)[0]
    nba_conf     = float(nba_proba.max())
    ltv_pred     = float(ltv_model.predict(feat)[0])

    # Priority
    aw       = {'Upsell': 1.0, 'Expand': 0.85, 'Nurture': 0.5, 'Retain': 0.3}
    ltv_norm = min(ltv_pred / latest['ltv_12m_estimate_sek'].max(), 1.0)
    priority = round(0.40 * ltv_norm + 0.35 * nba_conf + 0.25 * aw.get(nba_pred, 0.5), 3)

    return {
        'name':        acc_row['client_name'],
        'segment':     acc_row['client_segment'],
        'industry':    acc_row['industry_vertical'],
        'region':      acc_row['client_region'],
        'tenure':      int(acc_row['tenure_months']),
        'arr':         float(acc_row['arr_sek']),
        'nps':         float(acc_row['nps_score_proxy']),
        'products':    current,
        'white_space': ws,
        'nba':         {'action': nba_pred, 'confidence': nba_conf,
                        'proba':  dict(zip(nba_model.classes_, nba_proba.round(3)))},
        'ltv':         ltv_pred,
        'priority':    priority,
        'mom_growth':  float(health_row['mom_revenue_growth']),
        'tickets':     int(health_row['support_tickets_30d']),
    }


# ─────────────────────────────────────────────────────────────────────────────
# AI FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def get_api_key():
    try:
        return st.secrets.get("ANTHROPIC_API_KEY", "")
    except:
        return os.environ.get("ANTHROPIC_API_KEY", "")

def generate_narrative(result, api_key):
    if not api_key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        ws_top = result['white_space'][0]['product'] if result['white_space'] else 'N/A'
        prompt = f"""You are a B2B revenue intelligence assistant helping an Account Manager.
Generate a concise, actionable account briefing (3-4 sentences) for this enterprise client:

Account: {result['name']}
Segment: {result['segment']} | Industry: {result['industry']} | Region: {result['region']}
Tenure: {result['tenure']} months | ARR: SEK {result['arr']:,.0f} | NPS: {result['nps']}/10
Current products: {', '.join(result['products'])}
MoM Revenue Growth: {result['mom_growth']:.1%}
Support tickets (30d): {result['tickets']}

Model Outputs:
- Recommended next product: {ws_top}
- NBA Action: {result['nba']['action']} (confidence: {result['nba']['confidence']:.0%})
- Predicted LTV 12m: SEK {result['ltv']:,.0f}
- Priority Score: {result['priority']:.3f}/1.000

Write the briefing in a professional, direct tone. Start with the most important insight.
End with one specific recommended action for the AM this week."""

        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except Exception as e:
        return f"⚠️ AI narrative unavailable: {str(e)}"

def chat_with_copilot(user_message, portfolio_summary, chat_history, api_key):
    if not api_key:
        return "Please add your Anthropic API key in the sidebar to use the AM Copilot."
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        system = f"""You are an AM Copilot — an AI assistant helping Account Managers
prioritize their B2B enterprise portfolio at a CPaaS company.

You have access to the following portfolio data:
{portfolio_summary}

Answer questions about:
- Which accounts to prioritize
- Upsell and expansion opportunities
- Accounts at risk (Retain label)
- Product recommendations (White Space)
- Revenue predictions

Be concise, specific, and always reference actual account data.
Format numbers clearly (e.g. SEK 1.2M, not 1200000)."""

        messages = []
        for msg in chat_history[-6:]:  # last 6 messages for context
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_message})

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=400,
            system=system,
            messages=messages
        )
        return response.content[0].text
    except Exception as e:
        return f"⚠️ Copilot unavailable: {str(e)}"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Load data
    try:
        acc, subs, health, metrics, latest, basket = load_data()
        ALL_PRODUCTS = sorted(subs['product_line'].unique().tolist())
        nba_model, ltv_model, encode_fn, FEATURES = load_models(latest)
        rules = compute_rules(basket.to_dict(), ALL_PRODUCTS)
    except FileNotFoundError:
        st.error("⚠️ Dataset not found. Please run `python src/generate_dataset.py` first.")
        st.code("python src/generate_dataset.py")
        st.stop()

    # ── SIDEBAR ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f"""
        <div style='text-align:center; padding: 1rem 0;'>
            <div style='font-size:2rem'>📊</div>
            <div style='font-size:1rem; font-weight:700; color:white'>Revenue Intelligence</div>
            <div style='font-size:0.75rem; color:#A0AEC0'>B2B Enterprise CPaaS</div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # Filters
        st.markdown("**FILTER ACCOUNTS**")
        seg_filter = st.multiselect("Segment", acc['client_segment'].unique().tolist(),
                                     default=acc['client_segment'].unique().tolist())
        ind_filter = st.multiselect("Industry", acc['industry_vertical'].unique().tolist(),
                                     default=acc['industry_vertical'].unique().tolist())

        filtered_acc = acc[
            acc['client_segment'].isin(seg_filter) &
            acc['industry_vertical'].isin(ind_filter)
        ]

        st.divider()

        # Account selector
        st.markdown("**SELECT ACCOUNT**")

        # Pre-compute priority scores for sorting
        @st.cache_data
        def get_priority_scores(_latest, _acc):
            results = []
            for cid in _acc['client_id'].values[:50]:  # sample for speed in sidebar
                try:
                    r = run_inference(cid, _acc, subs, _latest, basket,
                                      nba_model, ltv_model, encode_fn, rules, ALL_PRODUCTS)
                    results.append({'client_id': cid, 'priority': r['priority'],
                                    'name': r['name'], 'nba': r['nba']['action']})
                except:
                    pass
            return pd.DataFrame(results).sort_values('priority', ascending=False)

        priority_df = get_priority_scores(latest, filtered_acc)

        account_options = {
            f"{row['name']} [{row['nba']}]": row['client_id']
            for _, row in priority_df.iterrows()
        }

        selected_label = st.selectbox("Account (sorted by priority)",
                                       list(account_options.keys()))
        selected_id = account_options[selected_label]

        st.divider()

        # API Key
        st.markdown("**AI COPILOT**")
        api_key_input = st.text_input("Anthropic API Key", type="password",
                                       value=get_api_key(),
                                       placeholder="sk-ant-...")
        api_key = api_key_input or get_api_key()
        if api_key:
            st.success("✅ AI features enabled")
        else:
            st.warning("Add key to enable AI")

        st.divider()
        st.markdown(f"<div style='font-size:0.75rem; color:#64748B'>"
                    f"Portfolio: {len(filtered_acc)} accounts<br>"
                    f"Products: {len(ALL_PRODUCTS)}<br>"
                    f"Rules: {len(rules)}</div>", unsafe_allow_html=True)

    # ── HEADER ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class='app-header'>
        <h1>📊 B2B Revenue Intelligence</h1>
        <p>White Space Analysis · Next Best Action · LTV Prediction · AM Copilot</p>
    </div>
    """, unsafe_allow_html=True)

    # ── RUN INFERENCE ─────────────────────────────────────────────────────────
    result = run_inference(selected_id, acc, subs, latest, basket,
                           nba_model, ltv_model, encode_fn, rules, ALL_PRODUCTS)

    # ── ACCOUNT OVERVIEW ──────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Account Overview</div>", unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Account</div>
            <div class='metric-value' style='font-size:1rem'>{result['name']}</div>
            <div class='metric-sub'>{result['segment']} · {result['region']}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Annual Revenue</div>
            <div class='metric-value'>SEK {result['arr']/1e6:.1f}M</div>
            <div class='metric-sub'>MoM: {result['mom_growth']:+.1%}</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Tenure</div>
            <div class='metric-value'>{result['tenure']}m</div>
            <div class='metric-sub'>{result['industry']}</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>NPS Score</div>
            <div class='metric-value'>{result['nps']:.1f}<span style='font-size:1rem'>/10</span></div>
            <div class='metric-sub'>Tickets: {result['tickets']} (30d)</div>
        </div>""", unsafe_allow_html=True)
    with col5:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Priority Score</div>
            <div class='metric-value'>{result['priority']:.3f}</div>
            <div class='priority-bar-bg'><div class='priority-bar-fill'
                 style='width:{result["priority"]*100:.0f}%'></div></div>
        </div>""", unsafe_allow_html=True)

    # Current products
    products_html = " ".join([f"<span class='product-tag'>✓ {p}</span>"
                               for p in result['products']])
    st.markdown(f"**Current products:** {products_html}", unsafe_allow_html=True)

    st.divider()

    # ── THREE MODEL OUTPUTS ───────────────────────────────────────────────────
    col_ws, col_nba, col_ltv = st.columns([1.3, 1, 1])

    # White Space
    with col_ws:
        st.markdown("<div class='section-header'>🗺️ White Space — Next Product</div>",
                    unsafe_allow_html=True)
        if result['white_space']:
            for ws in result['white_space']:
                color     = MINT if ws['label'] == 'High' else TEAL if ws['label'] == 'Medium' else GRAY
                bar_width = int(ws['score'] * 100)
                st.markdown(f"""
                <div class='ws-rec'>
                    <div style='flex:1'>
                        <div style='font-weight:600; color:{NAVY}'>{ws['product']}</div>
                        <div style='font-size:0.78rem; color:{GRAY}'>{ws['label']} confidence</div>
                        <div style='background:#E2E8F0; border-radius:3px; height:5px; margin-top:4px'>
                            <div style='background:{color}; width:{bar_width}%;
                                        height:5px; border-radius:3px'></div>
                        </div>
                    </div>
                    <div style='font-weight:700; color:{color}; margin-left:1rem'>
                        {ws['score']:.0%}
                    </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("Account uses all available products 🎉")

    # NBA
    with col_nba:
        st.markdown("<div class='section-header'>🎯 Next Best Action</div>",
                    unsafe_allow_html=True)
        action = result['nba']['action']
        color  = NBA_COLORS.get(action, TEAL)
        conf   = result['nba']['confidence']

        st.markdown(f"""
        <div class='metric-card {action.lower()}'>
            <div class='metric-label'>Recommended Action</div>
            <div style='margin: 0.5rem 0'>
                <span class='nba-badge' style='background:{color}'>{action}</span>
            </div>
            <div class='metric-sub'>Confidence: {conf:.0%}</div>
        </div>""", unsafe_allow_html=True)

        # Probability breakdown
        st.markdown("**Action probabilities:**")
        for label, prob in sorted(result['nba']['proba'].items(),
                                   key=lambda x: x[1], reverse=True):
            c = NBA_COLORS.get(label, GRAY)
            st.markdown(f"""
            <div style='display:flex; align-items:center; margin:0.2rem 0'>
                <div style='width:70px; font-size:0.8rem; color:{NAVY}'>{label}</div>
                <div style='flex:1; background:#E2E8F0; border-radius:3px; height:8px'>
                    <div style='background:{c}; width:{prob*100:.0f}%;
                                height:8px; border-radius:3px'></div>
                </div>
                <div style='width:40px; text-align:right; font-size:0.8rem;
                            color:{GRAY}; margin-left:0.5rem'>{prob:.0%}</div>
            </div>""", unsafe_allow_html=True)

    # LTV
    with col_ltv:
        st.markdown("<div class='section-header'>💰 LTV Prediction (12 months)</div>",
                    unsafe_allow_html=True)
        ltv_m  = result['ltv'] / 1e6
        arr_m  = result['arr'] / 1e6
        growth = (result['ltv'] - result['arr']) / result['arr'] if result['arr'] > 0 else 0

        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Predicted 12m Revenue</div>
            <div class='metric-value'>SEK {ltv_m:.2f}M</div>
            <div class='metric-sub'>vs current ARR: SEK {arr_m:.2f}M
                ({growth:+.1%})</div>
        </div>
        <div class='metric-card'>
            <div class='metric-label'>Expected Growth</div>
            <div class='metric-value' style='color:{"#02C39A" if growth>0 else "#E76F51"}'>
                {growth:+.1%}
            </div>
            <div class='metric-sub'>Based on MoM trend</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ── AI NARRATIVE ──────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>🤖 AI Account Briefing</div>",
                unsafe_allow_html=True)

    if 'narrative' not in st.session_state or \
       st.session_state.get('narrative_account') != selected_id:
        if api_key:
            with st.spinner("Generating AI briefing..."):
                narrative = generate_narrative(result, api_key)
            st.session_state['narrative'] = narrative
            st.session_state['narrative_account'] = selected_id
        else:
            st.session_state['narrative'] = None

    if st.session_state.get('narrative'):
        st.markdown(f"""<div class='ai-narrative'>
            💡 {st.session_state['narrative']}
        </div>""", unsafe_allow_html=True)
    else:
        ws_top = result['white_space'][0]['product'] if result['white_space'] else 'N/A'
        st.markdown(f"""<div class='ai-narrative' style='color:{GRAY}'>
            📋 <strong>{result['name']}</strong> is a {result['segment']} account
            in {result['industry']} with {result['tenure']} months tenure
            and SEK {result['arr']/1e6:.1f}M ARR. Current products:
            {', '.join(result['products'])}.<br><br>
            <strong>Recommended action:</strong> {result['nba']['action']}
            ({result['nba']['confidence']:.0%} confidence) ·
            <strong>Next product:</strong> {ws_top} ·
            <strong>LTV 12m:</strong> SEK {result['ltv']/1e6:.1f}M<br><br>
            <em>Add an Anthropic API key in the sidebar to enable AI-generated briefings.</em>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ── AM COPILOT ────────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>💬 AM Copilot — Ask About Your Portfolio</div>",
                unsafe_allow_html=True)

    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # Portfolio summary for context
    pipeline_file = 'data/processed/pipeline_output.csv'
    if os.path.exists(pipeline_file):
        pdf = pd.read_csv(pipeline_file)
        portfolio_summary = f"""
Portfolio: {len(pdf)} accounts
Segments: {pdf['segment'].value_counts().to_dict()}
NBA Distribution: {pdf['nba_action'].value_counts().to_dict()}
Top 5 priority accounts:
{pdf[['client_name','segment','industry','nba_action','ltv_12m_sek','priority_score']].head(5).to_string(index=False)}
Upsell opportunities: {(pdf['nba_action']=='Upsell').sum()}
Accounts at risk (Retain): {(pdf['nba_action']=='Retain').sum()}
Avg LTV: SEK {pdf['ltv_12m_sek'].mean():,.0f}
"""
    else:
        portfolio_summary = f"Current account: {result['name']} | {result['segment']} | {result['industry']}"

    # Show suggested questions
    if not st.session_state['chat_history']:
        st.markdown("**💡 Try asking:**")
        suggestions = [
            "Which accounts should I prioritize this week?",
            "Who has the highest upsell potential?",
            "Which Fintech accounts are missing Verification?",
            "Show me accounts at risk of churning",
        ]
        cols = st.columns(2)
        for i, s in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(s, key=f"sug_{i}"):
                    st.session_state['chat_history'].append(
                        {"role": "user", "content": s})
                    response = chat_with_copilot(s, portfolio_summary,
                                                  st.session_state['chat_history'], api_key)
                    st.session_state['chat_history'].append(
                        {"role": "assistant", "content": response})
                    st.rerun()

    # Chat history
    for msg in st.session_state['chat_history']:
        if msg['role'] == 'user':
            st.markdown(f"""<div class='chat-user'>
                👤 <strong>You:</strong> {msg['content']}
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class='chat-ai'>
                🤖 <strong>Copilot:</strong> {msg['content']}
            </div>""", unsafe_allow_html=True)

    # Input
    with st.form("chat_form", clear_on_submit=True):
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            user_input = st.text_input("Ask about your portfolio...",
                                        placeholder="e.g. Which accounts should I focus on today?",
                                        label_visibility="collapsed")
        with col_btn:
            submitted = st.form_submit_button("Send", type="primary")

    if submitted and user_input:
        st.session_state['chat_history'].append(
            {"role": "user", "content": user_input})
        with st.spinner("Copilot thinking..."):
            response = chat_with_copilot(user_input, portfolio_summary,
                                          st.session_state['chat_history'], api_key)
        st.session_state['chat_history'].append(
            {"role": "assistant", "content": response})
        st.rerun()

    if st.session_state['chat_history']:
        if st.button("Clear conversation"):
            st.session_state['chat_history'] = []
            st.rerun()


if __name__ == "__main__":
    main()
