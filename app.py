import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import shap
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(page_title="GeoImpact Risk Engine", page_icon="🌍", layout="wide")

# Custom CSS for Premium Design
st.markdown("""
    <style>
    .main { background-color: #0f172a; color: #f8fafc; }
    h1, h2, h3 { color: #38bdf8; font-family: 'Inter', sans-serif; }
    .stMetric { background-color: #1e293b; padding: 15px; border-radius: 10px; border: 1px solid #334155; }
    .stMetric label { color: #94a3b8 !important; }
    .stMetric div[data-testid="stMetricValue"] { color: #f8fafc !important; }
    div.stButton > button { width: 100%; border-radius: 8px; background: linear-gradient(90deg, #38bdf8, #818cf8); border: none; font-weight: bold; color: white; }
    </style>
""", unsafe_allow_html=True)

# --- Load Models & Data ---
@st.cache_resource
def load_models():
    reg_inflation = joblib.load('models/reg_inflation.pkl')
    reg_gdp = joblib.load('models/reg_gdp.pkl')
    reg_import = joblib.load('models/reg_import.pkl')
    classifier = joblib.load('models/risk_classifier.pkl')
    kmeans = joblib.load('models/kmeans_country.pkl')
    encoders = joblib.load('models/encoders.pkl')
    scaler = joblib.load('models/kmeans_scaler.pkl')
    df = pd.read_csv('apac_fuel_import_dependency.csv')
    return reg_inflation, reg_gdp, reg_import, classifier, kmeans, encoders, scaler, df

try:
    reg_inflation, reg_gdp, reg_import, classifier, kmeans, encoders, scaler, raw_df = load_models()
except FileNotFoundError:
    st.error("Models not found! Please run ml_training.py first.")
    st.stop()

# --- Helper Functions ---
def get_gauge_chart(value, title, min_val, max_val, invert_colors=False):
    colors = ['#10b981', '#f59e0b', '#ef4444'] if not invert_colors else ['#ef4444', '#f59e0b', '#10b981']
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        title = {'text': title, 'font': {'color': '#94a3b8', 'size': 16}},
        gauge = {
            'axis': {'range': [min_val, max_val], 'tickcolor': "#334155"},
            'bar': {'color': "#38bdf8"},
            'bgcolor': "#1e293b",
            'borderwidth': 2,
            'bordercolor': "#334155",
            'steps': [
                {'range': [min_val, (max_val-min_val)*0.33 + min_val], 'color': colors[0]},
                {'range': [(max_val-min_val)*0.33 + min_val, (max_val-min_val)*0.66 + min_val], 'color': colors[1]},
                {'range': [(max_val-min_val)*0.66 + min_val, max_val], 'color': colors[2]}
            ],
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': "#f8fafc"})
    return fig

# --- Sidebar: Scenario Simulator ---
st.sidebar.title("Scenario Simulator")
st.sidebar.markdown("Configure a geopolitical shock scenario here to predict its economic effects.")

selected_country = st.sidebar.selectbox("Country", list(encoders['Country'].classes_), help="Select the APAC country you want to simulate the shock for.")
selected_fuel = st.sidebar.selectbox("Fuel Type", list(encoders['Fuel_Type'].classes_), help="Different fuels have different supply chains and import dependencies.")
selected_conflict = st.sidebar.selectbox("Conflict Phase", list(encoders['Conflict_Phase'].classes_), help="The stage of the geopolitical conflict, which impacts global trade panic.")

# Get default values for the selected country & fuel
country_mask = (raw_df['Country'] == selected_country) & (raw_df['Fuel_Type'] == selected_fuel)
default_vol = float(raw_df[country_mask]['Import_Volume_KBPD'].mean()) if not country_mask.empty else 1000.0
default_me = float(raw_df[country_mask]['ME_Share_Pct'].mean()) if not country_mask.empty else 50.0
default_alt = float(raw_df[country_mask]['Alternative_Source_Pct'].mean()) if not country_mask.empty else 50.0
default_spr = float(raw_df[country_mask]['SPR_Days_Cover'].mean()) if not country_mask.empty else 60.0

st.sidebar.subheader("Adjust Shock Factors")

# Calculate realistic upper bounds for sliders based on the specific country's baseline
max_vol = float(max(default_vol * 2.5, 500.0)) 
max_spr = float(max(default_spr * 2.0, 90.0))

import_vol = st.sidebar.slider("Import Volume (KBPD)", 0.0, max_vol, default_vol, help="Total fuel volume imported. Higher volume means heavier economic reliance.")
me_share = st.sidebar.slider("Middle East Dependency (%)", 0.0, 100.0, default_me, help="Percentage of this fuel imported directly from the Middle East. Higher % = Higher vulnerability to regional conflict.")
alt_source = st.sidebar.slider("Alternative Source (%)", 0.0, 100.0, default_alt, help="Percentage of fuel that can be quickly sourced from non-conflict regions. Acts as a safety buffer.")
price_premium = st.sidebar.slider("Oil Price Premium Shock (%)", 0.0, 150.0, 20.0, help="The sudden spike in global fuel prices caused by the conflict. A 150% spike reflects a historically catastrophic event.")
spr_days = st.sidebar.slider("SPR Days Cover", 0.0, max_spr, default_spr, help="Strategic Petroleum Reserves. How many days the country's economy can run purely on stored reserves before facing a crisis.")

# --- Main Dashboard ---
st.title("🌍 GeoImpact Risk Engine")
st.markdown("**Built by Khannak P Gupta** | Powered by Machine Learning")

with st.expander("👤 How to Use", expanded=True):
    st.markdown("""
    **Welcome to my interactive risk engine!** The news nowadays is always about the severe war in the Middle East and the impact it has on our economies. However, it often becomes hard for us to visualise how different parameters decide the consequences of such events on a country. Thus, I built *GeoShock AI* to  demonstrate just how fragile the APAC energy grid really is. While the AI runs the mathematical models behind the scenes, this dashboard lets you play out real-world "what-if" scenarios.
    
    **To run a simulation:**
    1. **Go to the sidebar on the left** and choose a Country, Fuel Type, and Conflict Phase.
    2. **Tweak the Shock Factors.** What happens if global oil prices spike by 150%? What if a country's Strategic Petroleum Reserve (SPR) drops to just 10 days? Move the sliders to test the economy's breaking point.
    3. **Click 'Run Simulation 🚀'**.
    4. **Analyze the results.** The engine will calculate the resulting Inflation, GDP contraction, Import Cost surge, and generate actionable policy recommendations.
    """)

# Prepare Input Features
input_data = pd.DataFrame({
    'Country_Encoded': [encoders['Country'].transform([selected_country])[0]],
    'Fuel_Type_Encoded': [encoders['Fuel_Type'].transform([selected_fuel])[0]],
    'Conflict_Phase_Encoded': [encoders['Conflict_Phase'].transform([selected_conflict])[0]],
    'Import_Volume_KBPD': [import_vol],
    'ME_Share_Pct': [me_share],
    'Alternative_Source_Pct': [alt_source],
    'Price_Premium_Pct': [price_premium],
    'SPR_Days_Cover': [spr_days]
})

features = ['Country_Encoded', 'Fuel_Type_Encoded', 'Conflict_Phase_Encoded', 
            'Import_Volume_KBPD', 'ME_Share_Pct', 'Alternative_Source_Pct', 
            'Price_Premium_Pct', 'SPR_Days_Cover']

X_input = input_data[features]

# Predictions
if st.sidebar.button("Run Simulation 🚀"):
    inf_pred = reg_inflation.predict(X_input)[0]
    gdp_pred = reg_gdp.predict(X_input)[0]
    imp_pred = reg_import.predict(X_input)[0]
    risk_pred = classifier.predict(X_input)[0]
    
    risk_labels = {0: "Low Risk 🟢", 1: "Medium Risk 🟡", 2: "High Risk 🟠", 3: "Critical Risk 🔴"}
    
    st.markdown("### 📊 Economic Impact Forecast")
    st.caption("These metrics represent the direct economic blow to the selected country under the current simulated shock scenario.")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Inflation Impact", f"+{inf_pred:.2f}%", delta_color="inverse")
    with col2:
        st.metric("GDP Impact", f"{gdp_pred:.2f}%")
    with col3:
        st.metric("Import Cost Surge", f"${imp_pred:.2f}M", delta_color="inverse")
    with col4:
        st.metric("Risk Level", risk_labels[risk_pred])
        
    st.markdown("---")
    
    # Visualizations
    st.markdown("### 📈 Impact Analysis")
    chart_col1, chart_col2, chart_col3 = st.columns(3)
    
    with chart_col1:
        st.plotly_chart(get_gauge_chart(inf_pred, "Inflation Shock (%)", 0, 10), use_container_width=True)
        st.info("💡 **Inflation Shock:** Measures how much consumer prices will rise. Values in the red zone indicate severe runaway inflation.")
    with chart_col2:
        st.plotly_chart(get_gauge_chart(gdp_pred, "GDP Contraction (%)", -10, 0, invert_colors=True), use_container_width=True)
        st.info("💡 **GDP Contraction:** Measures the percentage drop in economic growth. A more negative number means a deeper recession.")
    with chart_col3:
        st.plotly_chart(get_gauge_chart(risk_pred, "Geopolitical Risk Score", 0, 3), use_container_width=True)
        st.info("💡 **Risk Score:** An aggregate AI classification combining supply chain vulnerability, reserve levels, and price shocks.")
        
    st.markdown("---")
    
    # SHAP Explainability
    st.markdown("### 🧠 AI Explainability (What's driving the inflation?)")
    st.markdown("This chart shows exactly how much each specific factor contributed to the final Inflation prediction. ")
    
    explainer = shap.TreeExplainer(reg_inflation)
    shap_values = explainer(X_input)
    
    # Clean up feature names for the UI
    friendly_names = {
        'Price_Premium_Pct': 'Oil Price Premium',
        'ME_Share_Pct': 'Middle East Dependency',
        'Import_Volume_KBPD': 'Import Volume',
        'SPR_Days_Cover': 'SPR Days Cover',
        'Alternative_Source_Pct': 'Alternative Sources',
        'Country_Encoded': 'Country Profile',
        'Fuel_Type_Encoded': 'Fuel Type',
        'Conflict_Phase_Encoded': 'Conflict Phase'
    }
    
    shap_df = pd.DataFrame({
        'Factor': [friendly_names.get(f, f) for f in features],
        'Impact': shap_values.values[0]
    })
    
    # Sort by absolute impact to show the biggest drivers first
    shap_df['Abs_Impact'] = shap_df['Impact'].abs()
    shap_df = shap_df.sort_values(by='Abs_Impact', ascending=True)
    
    # Red for pushing inflation up, Green for pulling it down
    shap_df['Color'] = shap_df['Impact'].apply(lambda x: '#ef4444' if x > 0 else '#10b981')
    
    fig = go.Figure(go.Bar(
        x=shap_df['Impact'],
        y=shap_df['Factor'],
        orientation='h',
        marker_color=shap_df['Color'],
        text=shap_df['Impact'].apply(lambda x: f"+{x:.2f}%" if x > 0 else f"{x:.2f}%"),
        textposition='outside'
    ))
    
    fig.update_layout(
        xaxis_title="Impact on Inflation (%)",
        yaxis_title="",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "#f8fafc", 'size': 14},
        height=350,
        margin=dict(l=20, r=40, t=20, b=20)
    )
    
    fig.update_xaxes(showgrid=True, gridcolor='#334155', zeroline=True, zerolinecolor='#94a3b8')
    fig.update_yaxes(showgrid=False)
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("📊 **How to read this chart:** Look at the longest bars. **Red bars** show the factors pushing inflation higher (making things worse). **Green bars** show the factors acting as a safety buffer, pulling inflation back down.")
    
    st.markdown("---")
    
    # Recommendations Engine & AI Rationale
    st.markdown("### 💡 Strategic Recommendations & AI Rationale")
    st.caption("The AI has analyzed your scenario and generated these specific policy interventions:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🚨 Primary Vulnerabilities")
        vulnerabilities = False
        if me_share > 40:
            st.error(f"**Heavy Regional Reliance:** The AI notes a {me_share:.1f}% dependency on the Middle East for {selected_fuel}. *Why this matters:* Any supply chain disruption will instantly halt imports, driving up inflation.")
            vulnerabilities = True
        if spr_days < 60:
            st.error(f"**Critically Low Reserves:** The country only has {spr_days:.0f} days of SPR cover. *Why this matters:* The AI predicts a severe GDP contraction because the country cannot outlast a prolonged supply shock.")
            vulnerabilities = True
        if price_premium > 30:
            st.warning(f"**Massive Cost Surge:** A {price_premium:.1f}% price premium was detected. *Why this matters:* The AI correlates this directly to a ${imp_pred:.1f}M surge in import costs, draining foreign exchange reserves.")
            vulnerabilities = True
            
        if not vulnerabilities:
            st.success("✅ **No severe vulnerabilities detected** in the current configuration.")
            
    with col2:
        st.markdown("#### 🛡️ Recommended Actions")
        if me_share > 40:
            st.info("🔄 **Diversify Supply Chain:** Route imports to alternative sources (e.g., US, Africa) to drop Middle East dependency below 40%.")
        if spr_days < 60:
            st.info("🛢️ **Mandate Stockpiling:** Enforce legislation requiring energy companies to build and maintain a minimum of 90 days of reserve capacity.")
        if price_premium > 30:
            st.info("📉 **Demand Interventions:** Implement temporary fuel rationing, subsidize public transport, and release existing SPR reserves to cool domestic prices.")
        if not vulnerabilities:
            st.info("🌟 **Maintain Strategy:** Your energy policy is highly resilient. Continue strengthening alternative source relationships.")

else:
    st.info("👈 Please adjust the parameters in the Scenario Simulator on the sidebar and click **Run Simulation 🚀** to see the economic impact.")
