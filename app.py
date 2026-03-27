import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import google.generativeai as genai
#---chatbot---
from chatbot import build_context_library
#import anthropic
import os
import json

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EV Charging Intelligence",
    # Title shown in browser tab
    page_icon="⚡",
    # Icon shown in browser tab
    layout="wide",
    # Use full browser width
    initial_sidebar_state="expanded"
    # Sidebar open by default
)

# ── Custom CSS — inject styles into the page ──────────────────────────────────
st.markdown("""
<style>
/* Main background */
.main { background-color: #0e1117; }

/* Metric card style */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1e2130, #252940);
    border: 1px solid #2d3250;
    border-radius: 12px;
    padding: 16px 20px;
}

/* Metric label */
[data-testid="stMetric"] label {
    color: #8b92a5 !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.8px;
    text-transform: uppercase;
}

/* Metric value */
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #e8eaf0 !important;
    font-size: 28px !important;
    font-weight: 700 !important;
}

/* Tab styling */
.stTabs [data-baseweb="tab"] {
    background-color: #1a1d2e;
    border-radius: 8px 8px 0 0;
    color: #8b92a5;
    font-weight: 600;
    padding: 10px 20px;
}

/* Active tab */
.stTabs [aria-selected="true"] {
    background-color: #2d3250;
    color: #7c83e0 !important;
    border-bottom: 3px solid #7c83e0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #13151f;
    border-right: 1px solid #2d3250;
}

/* Sidebar text */
[data-testid="stSidebar"] .css-1d391kg {
    color: #e8eaf0;
}

/* Headers */
h1, h2, h3 {
    color: #e8eaf0 !important;
    font-weight: 700 !important;
}

/* Divider */
hr { border-color: #2d3250; }

/* Info box */
.info-box {
    background: linear-gradient(135deg, #1e2a3a, #1a2535);
    border: 1px solid #2d4a6e;
    border-left: 4px solid #4d9de0;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #8bb8d4;
    font-size: 13px;
}

/* Success box */
.success-box {
    background: linear-gradient(135deg, #1a2e1a, #1e331e);
    border: 1px solid #2d6e2d;
    border-left: 4px solid #4dde4d;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #8bd48b;
    font-size: 13px;
}

/* Warning box */
.warning-box {
    background: linear-gradient(135deg, #2e2a1a, #33301e);
    border: 1px solid #6e5a2d;
    border-left: 4px solid #deb84d;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #d4c08b;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)
# unsafe_allow_html=True: required to inject raw HTML and CSS
# This CSS customises colours, fonts, borders, and spacing

print("Part 1 written — imports and page config")

# ═══════════════════════════════════════════════════════════
# DATA AND MODEL LOADING
# ═══════════════════════════════════════════════════════════

RFG_MODEL_PATH = "model/ev_rfg_model.pkl"
CLF_MODEL_PATH = "model/ev_clf_model.pkl"
RG_FEATURES_PATH = "model/ev_rg_features.pkl"
CLF_FEATURES_PATH = "model/clf_features.pkl"
DATASET_PATH = "data_set/ev_dataset.parquet"
STATION_DATA_PATH = "data_set/station_info.parquet"
THRESHOLD = 0.5  # Classification threshold for availability predictions

@st.cache_data
def load_data():
    """
    Load the EV dataset efficiently without aggressive dtype conversion.
    Parquet format is already memory-efficient; conversion can cause allocation spikes.
    """
    # Load Main Dataset directly without dtype conversion
    # Parquet is already space-efficient for storage
    df = pd.read_parquet(DATASET_PATH)
    
    # Load Station Data
    station_data = pd.read_parquet(STATION_DATA_PATH)
    
    # Merge station info (categorical columns) with main dataset for filtering
    # Keep only the categorical columns from station_data
    station_info_for_merge = station_data[['station_id', 'city', 'network', 'location_type', 'charger_type']]
    df = df.merge(station_info_for_merge, on='station_id', how='left')
    
    # Create availability target: 1 if low utilization (<0.5), 0 if high utilization
    # This represents whether a port is available (not fully occupied)
    #df['target_avail_t1'] = (df['targets_utilization_t+1'] < 0.5).astype(bool)
    df['target_avail_t1'] = (
    df.groupby('station_id')['ports_available'].shift(-1) > 0).astype(int)
    
    return df, station_data

@st.cache_resource
def load_models():
    """
    Load all ML models and config files.
    @st.cache_resource is for objects that should not be serialised
    (like ML models) — they stay alive in memory across reruns.
    """
    import joblib
    try:
        # Define paths here to avoid scope issues with cached functions
        rfg_model = joblib.load("model/ev_rfg_model.pkl")
        clf_model = joblib.load("model/ev_clf_model.pkl")
        rg_features = joblib.load("model/ev_rg_features.pkl")
        clf_features = joblib.load("model/clf_features.pkl")  # Note: clf_features, not ev_clf_features
        return rfg_model, clf_model, rg_features, clf_features
    except Exception as e:
        st.error(f"⚠️ Error loading model files: {str(e)}")
        st.info("Model files are missing or corrupted.\nPlease regenerate models by running: python retrain_models.py")
        st.stop()

print("Part 2 written — data and model loading functions")

# Load data and models at the start of the app
df, station_data = load_data()
rfg_model, clf_model, rg_features, clf_features = load_models()

# Build RAG context library for Tab 6 chatbot
context_lib = build_context_library(df)

print("Part 3 written — data and models loaded into memory")

# Derive filters and options from the dataset
All_cities = sorted(station_data['city'].unique().tolist())
All_networks = sorted(station_data['network'].unique().tolist())
All_locations = sorted(station_data['location_type'].unique().tolist())
All_charger_types = sorted(station_data['charger_type'].unique().tolist())
All_Station_IDs = sorted(station_data['station_id'].unique().tolist())
print("Part 4 written — filter options derived from dataset")
print(f"  Available cities: {len(All_cities)}")
print(f"  Available networks: {len(All_networks)}")
print(f"  Available location types: {len(All_locations)}")
print(f"  Available charger types: {len(All_charger_types)}")
print(f"  Available stations: {len(All_Station_IDs)}")

#app_part3_lines = len(app.splitlines())

# ════════════════════════════════════════════════════════════════════════
# RAG HELPERS  —  used by Tab 6 chatbot
# Defined here so they are available throughout the file
# ════════════════════════════════════════════════════════════════════════
KEYWORD_MAP = {
    "overall"        :["total","overall","dataset","summary","how many",
                        "overview","describe","all stations","general"],
    "peak_hours"     :["peak","rush","busy","hour","morning","evening",
                        "when","time","surge","congested","demand"],
    "cities"         :["city","cities","metro","where","region",
                        "area","which city","highest city"],
    "networks"       :["network","operator","provider","company",
                        "brand","which network","best network"],
    "location_type"  :["highway","mall","parking","office","hotel",
                        "location type","venue","what type"],
    "weather"        :["weather","rain","snow","temperature","cold",
                        "hot","precipitation","climate","season"],
    "pricing_traffic":["price","pricing","cost","expensive","traffic",
                        "congestion","gas","fuel","surge pricing"],
    "charger_type"   :["charger","level","dc fast","power","kw",
                        "fast charging","speed","level 2","level 3"],
    "ml_models"      :["model","predict","forecast","accuracy","mae",
                        "f1","auc","performance","xgboost","score"],
    "operations"     :["wait","maintenance","anomaly","recommend",
                        "strategy","insight","improve","infrastructure"]
}

def rag_retrieve(question, max_chunks=3):
    """
    Scan question for keywords.
    Return text blocks for the top matching topics.
    This is the RETRIEVAL step of RAG.
    """
    q      = question.lower()
    # Score each topic by counting keyword matches
    scores = {
        topic: sum(1 for kw in kws if kw in q)
        for topic, kws in KEYWORD_MAP.items()
    }
    # Keep only topics that had at least one match
    scores = {t: s for t, s in scores.items() if s > 0}
    # If no keywords matched, return overall summary
    if not scores:
        return context_lib.get("overall", "")
    # Sort by score, take top max_chunks
    top = sorted(scores, key=scores.get, reverse=True)[:max_chunks]
    # Combine their text blocks
    return "\\n\\n".join(
        context_lib[t] for t in top if t in context_lib
    )


# ═══════════════════════════════════════════════════════════
# SIDEBAR — ALL FILTERS
# ═══════════════════════════════════════════════════════════

# ── Sidebar header ────────────────────────────────────────────────────────────
import base64
def get_base64(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode()
img1 = get_base64("assets/ev1.png")

# st.markdown(f"""
# <img src="data:image/png;base64,{img1}" width="100">
# """, unsafe_allow_html=True)

# Sidebar row layout
col1, col2 = st.sidebar.columns([1,1])

with col1:
    st.image(f"data:image/png;base64,{img1}", width=90)

with col2:
    st.markdown("""
    <div style="padding-top:5px;">
        <div style="font-size:18px; font-weight:700;
                    color:#7c83e0;">
            EV INTEL
        </div>
        <div style="font-size:11px; color:#8b92a5;">
            Charging Analytics Platform
        </div>
    </div>
    """, unsafe_allow_html=True)


#st.sidebar.image(f"data:image/png;base64,{img1}", width=100)


# st.sidebar.markdown("""
# <div style="text-align:center; padding: 10px 0 20px 0;">
#     <div style="font-size:40px;">⚡</div>
#     <div style="font-size:18px; font-weight:700;
#                 color:#7c83e0; letter-spacing:1px;">
#         EV INTEL
#     </div>
#     <div style="font-size:11px; color:#8b92a5;
#                 margin-top:4px; letter-spacing:0.5px;">
#         Charging Analytics Platform
#     </div>
# </div>
# """, unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<p style='color:#8b92a5; font-size:14px;"
    " font-weight:600; letter-spacing:1px;'>"
    "🔽  FILTERS</p>",
    unsafe_allow_html=True
)

# ── City filter ────────────────────────────────────────────────────────────────
selected_cities = st.sidebar.multiselect(
    " 🌆 City",
    options=All_cities,
    default=All_cities[:5],
    # Default: first 5 cities so dashboard loads fast
    help="Filter all panels to selected cities"
)

# ── Network filter ─────────────────────────────────────────────────────────────
selected_networks = st.sidebar.multiselect(
    "🔌 Network",
    options=All_networks,
    default=All_networks,
    help="Filter by charging network operator"
)

# ── Location type filter ───────────────────────────────────────────────────────
selected_loc_types = st.sidebar.multiselect(
    "📍 Location Type",
    options=All_locations,
    default=All_locations,
    help="Highway, Mall, Office, etc."
)

# ── Charger type filter ────────────────────────────────────────────────────────
selected_chargers = st.sidebar.multiselect(
    "⚙️ Charger Type",
    options=All_charger_types,
    default=All_charger_types,
    help="Level 2, DC Fast Charger, etc."
)

st.sidebar.markdown("---")

# ── Date range filter ──────────────────────────────────────────────────────────
min_date = df["timestamp"].min().date()
max_date = df["timestamp"].max().date()

date_range = st.sidebar.date_input(
    "📅 Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    help="Select start and end dates"
)

# ── Unpack date range safely ───────────────────────────────────────────────────
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range
# isinstance: check if date_range is a tuple
# Handles edge case where user selects only one date

st.sidebar.markdown("---")

# ── Apply all filters ──────────────────────────────────────────────────────────
@st.cache_data
def apply_filters(cities, networks, loc_types,
                  chargers, start_dt, end_dt):
    """Apply sidebar filters and return filtered DataFrame."""
    mask = (
        df["city"].isin(cities)           &
        df["network"].isin(networks)       &
        df["location_type"].isin(loc_types) &
        df["charger_type"].isin(chargers)  &
        (df["timestamp"].dt.date >= start_dt) &
        (df["timestamp"].dt.date <= end_dt)
    )
    # Each condition creates a boolean Series (True/False per row)
    # & combines them: True only where ALL conditions are True
    return df[mask].copy()

filtered_df = apply_filters(
    selected_cities, selected_networks,
    selected_loc_types, selected_chargers,
    start_date, end_date
)

# ── Sidebar stats ──────────────────────────────────────────────────────────────
st.sidebar.markdown(
    "<p style='color:#8b92a5; font-size:11px;"
    " font-weight:600; letter-spacing:1px;'>"
    "📊  DATA IN VIEW</p>",
    unsafe_allow_html=True
)

col_s1, col_s2 = st.sidebar.columns(2)
col_s1.metric("Records",  f"{len(filtered_df):,}")
col_s2.metric("Stations", f"{filtered_df['station_id'].nunique()}")
# st.metric: displays a labelled number card

print("Part 3 written — sidebar filters")
#print(f"Lines: {app.count(chr(10))}")


# ==================== MAIN DASHBOARD ====================

st.markdown("""
<div style="background: linear-gradient(135deg, #1e2130, #252950);
            border: 1px solid #2d3250;
            border-radius: 16px;
            padding: 10px 18px;
            margin-bottom: 24px;">
    <div style="display:flex; align-items:center; gap:12px;">
        <div style="font-size:40px;">⚡</div>
        <div>
            <h1 style="margin:0; font-size:22px; color:#e8eaf0;
                       font-weight:800; letter-spacing:1px;">
                EV Charging Availability & Demand Dashboard
            </h1>
            <p style="margin:2px 0 0 0; color:#8b92a5;
                      font-size:14px; letter-spacing:0.5px;">
                Availability Forecasting · Demand Analytics ·
                Station Monitoring
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Summary caption ────────────────────────────────────────────────────────────
st.caption(
    f"Showing **{len(filtered_df):,}** Records. · "
    f"**{filtered_df['station_id'].nunique()}** Stations · "
    f"**{filtered_df['city'].nunique()}** Cities · "
    f"**{filtered_df['network'].nunique()}** Networks"
)



# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3,tab4,tab5,tab6 = st.tabs([
    "📊  Overview",
    "📈  Trends",
    "🔮  Forecast",
    "🎯  Availability",
    "🗺️  Station Map",
    "🤖  AI Assistant"
])


# ----------CHATBOT summery---------------------

#--------------------------------------------------

# st.tabs: creates clickable tabs across the top
# Content inside each tab only renders when that tab is active
# This keeps the app fast — inactive panels are not computed
print("Part 4  — header and tabs")

# ═══════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════

with tab1:
    # ── Section title ──────────────────────────────────────────────────────────
    st.markdown("#### 📊 Fleet-Level KPIs")
     # ── Row of 4 metric cards ──────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    # st.columns(4): divide page into 4 equal columns
    with col1:
        avg_util = filtered_df["utilization_rate"].mean()
        # .mean(): average utilization across all filtered rows
        delta_val = avg_util - df["utilization_rate"].mean()
        # delta: difference vs overall average (all data)
        st.metric(
            label="⚡ Avg Utilization",
            value=f"{avg_util:.1%}",
            # .1% format: shows as percentage with 1 decimal
           # delta=f"{delta_val:+.1%}",
            # delta: shown in green/red below the main value
            help="Mean utilization rate across selected stations"
        )
    with col2:
        pct_full = (filtered_df["ports_available"] == 0).mean()
        # (ports_available == 0): True where no ports are free
        # .mean(): fraction of intervals where station was fully occupied
        st.metric(
            label="🔴 Fully Occupied",
            value=f"{pct_full:.1%}",
            # delta=f"{pct_full - (df['ports_available']==0).mean():+.1%}",
            help="% of intervals where ALL ports were occupied"
        )

    with col3:
        avg_wait = filtered_df["estimated_wait_time_mins"].mean()
        st.metric(
            label="⏱️ Avg Wait Time",
            value=f"{avg_wait:.1f} min",
            help="Average estimated wait time across selection"
        )

    with col4:
        n_stations = filtered_df["station_id"].nunique()
        st.metric(
            label="🏪 Stations",
            value=f"{n_stations}",
            help="Number of unique stations in current filter"
        )

    #st.markdown("---")
    # ── Utilization distribution / heatmap chart ───────────────────────────────────────────
    st.markdown("#### 📈 Utilization heatmap Hour Vs Day of Week")
    # Step 1: Create pivot table
    pivot = (filtered_df
        .groupby(["day_of_week", "hour_of_day"])["utilization_rate"]
        .mean()
        .unstack("hour_of_day"))
    # groupby two columns → mean → unstack
    # unstack: pivot hours into columns → creates matrix
    # Rows = days (0=Mon...6=Sun), Columns = hours (0-23)
    # Step 2: Plot heatmap
    # Step 2: Rename rows
    pivot.index = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

    # Step 3: Create heatmap
    fig_heat = px.imshow(
        pivot,
        color_continuous_scale=[
            [0.0,  "#0d1117"],
            [0.3,  "#1a3a5c"],
            [0.6,  "#2166ac"],
            [0.8,  "#f4a261"],
            [1.0,  "#e63946"]
        ],
        # Custom colour scale: dark → blue → orange → red
        labels=dict(x="Hour of Day", y="Day ", color="Utilization"),
        aspect="auto",
        title=""
    )
    fig_heat.update_layout(
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        # transparent background
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8eaf0", size=12),
        margin=dict(l=40, r=20, t=10, b=40),
        coloraxis_colorbar=dict(
            tickformat=".0%",
            # show colorbar values as percentages
            tickfont=dict(color="#8b92a5", size=11)
        )
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    # use_container_width=True: chart fills full column width
    #st.markdown("---")
    st.markdown("#### 📍 Avg Utilization by Hour and Location Type")
    pivot_util = filtered_df.pivot_table(
        index=filtered_df["timestamp"].dt.hour,
        columns="location_type",
        values="utilization_rate",
        aggfunc="mean"
    )   
    # pivot_table: reshape data to show avg utilization by hour and location type
    # index: rows by hour of day, columns by location type  
    # values: what to aggregate (utilization_rate), aggfunc: how to aggregate (mean)
    pivot_util = pivot_util.fillna(0)  # Replace NaN with 0
    fig_util = px.imshow(
        pivot_util.T,
        labels={"x": "Hour of Day", "y": "Location Type", "color": "Avg Utilization"},
        color_continuous_scale="Viridis",
        aspect="auto"
    )
    fig_util.update_layout(
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8eaf0", size=12),
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis=dict(tickmode='linear'),
        yaxis=dict(tickmode='linear')
    )
    st.plotly_chart(fig_util, use_container_width=True)

# ── Two charts side by side ────────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 🌐 By Network")

        # Step 1: Compute average utilization per network
        net_data = (filtered_df
            .groupby("network")["utilization_rate"]
            .mean()
            .sort_values(ascending=True)
            .reset_index())

        # Step 2: Create horizontal bar chart
        fig_net = px.bar(
            net_data,
            x="utilization_rate",
            y="network",
            orientation="h",
            # orientation="h": horizontal bars
            color="utilization_rate",
            color_continuous_scale=["#1a3a5c","#2166ac","#f4a261","#e63946"],
            labels={"utilization_rate": "Avg Utilization",
                    "network": ""}
        )
        fig_net.update_layout(
            height=320,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e8eaf0", size=11),
            margin=dict(l=10, r=20, t=10, b=30),
            showlegend=False,
            coloraxis_showscale=False
        )
        fig_net.update_xaxes(
            tickformat=".0%",
            gridcolor="#1e2130",
            color="#8b92a5"
        )
        fig_net.update_yaxes(color="#e8eaf0")
        st.plotly_chart(fig_net, use_container_width=True)

    with col_b:
        st.markdown("#### ⚡ By Charger Type")

        loc_data = (filtered_df
            .groupby("charger_type")["utilization_rate"]
            .mean()
            .sort_values(ascending=True)
            .reset_index())

        fig_loc = px.bar(
            loc_data,
            x="utilization_rate",
            y="charger_type",
            orientation="h",
            color="utilization_rate",
            color_continuous_scale=["#1a3a5c","#2166ac","#f4a261","#e63946"],
            labels={"utilization_rate": "Avg Utilization",
                    "charger_type": ""}
        )
        fig_loc.update_layout(
            height=320,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e8eaf0", size=11),
            margin=dict(l=10, r=20, t=10, b=30),
            showlegend=False,
            coloraxis_showscale=False
        )
        fig_loc.update_xaxes(
            tickformat=".0%",
            gridcolor="#1e2130",
            color="#8b92a5"
        )
        fig_loc.update_yaxes(color="#e8eaf0")
        st.plotly_chart(fig_loc, use_container_width=True)
    
# ═══════════════════════════════════════════════════════════
# TAB 2 — TRENDS
# ═══════════════════════════════════════════════════════════

with tab2:

    st.markdown("### 📈 Utilization & Availability Trends")
    # ── Granularity selector ───────────────────────────────────────────────────
    gran_col, space_col = st.columns([2, 5])
    with gran_col:
        granularity = st.radio(
            "Aggregation Level",
            options=["Hourly", "Daily", "Weekly"],
            horizontal=True
        )

    # Map label to pandas resample frequency
    gran_map = {"Hourly": "h", "Daily": "D", "Weekly": "W"}
    freq = gran_map[granularity]

    # ── Resample data ──────────────────────────────────────────────────────────
    trend_data = (filtered_df
        .set_index("timestamp")
        [["utilization_rate", "ports_available",
          "estimated_wait_time_mins"]]
        .resample(freq)
        .mean()
        # resample(freq): group timestamps into fixed-size time bins
        # .mean(): average all values within each bin
        .reset_index())

    # ── Two-panel chart ────────────────────────────────────────────────────────
    fig_trend = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        # shared_xaxes: both subplots share the same x-axis
        # Zooming on one zooms both simultaneously
        subplot_titles=("Utilization Rate", "Ports Available"),
        vertical_spacing=0.12,
        row_heights=[0.6, 0.4]
    )

    # ── Top panel: Utilization rate ────────────────────────────────────────────
    fig_trend.add_trace(
        go.Scatter(
            x=trend_data["timestamp"],
            y=trend_data["utilization_rate"],
            mode="lines",
            name="Utilization",
            line=dict(color="#7c83e0", width=2),
            fill="tozeroy",
            # fill="tozeroy": shade area between line and y=0
            fillcolor="rgba(124,131,224,0.12)"
        ),
        row=1, col=1
    )

    # ── Bottom panel: Ports available ──────────────────────────────────────────
    fig_trend.add_trace(
        go.Scatter(
            x=trend_data["timestamp"],
            y=trend_data["ports_available"],
            mode="lines",
            name="Ports Available",
            line=dict(color="#4dde9d", width=2),
            fill="tozeroy",
            fillcolor="rgba(77,222,157,0.10)"
        ),
        row=2, col=1
    )

    fig_trend.update_layout(
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8eaf0", size=12),
        hovermode="x unified",
        # hovermode="x unified": hovering shows all traces at that x
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            bgcolor="rgba(0,0,0,0)"
        ),
        showlegend=True
    )
    fig_trend.update_xaxes(gridcolor="#1e2130", color="#8b92a5")
    fig_trend.update_yaxes(gridcolor="#1e2130", color="#8b92a5")
    fig_trend.update_yaxes(tickformat=".0%", row=1)
    st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("---")

    st.markdown("### ⚡ Peak vs Off-Peak Breakdown")

    filtered_df["hour"] = pd.to_datetime(filtered_df["timestamp"]).dt.hour
    filtered_df["is_peak_hour"] = filtered_df["hour"].apply(
        lambda x: 1 if 18 <= x <= 22 else 0)

    peak_data = (filtered_df
        .groupby("is_peak_hour")
        .agg(
            avg_util     = ("utilization_rate",          "mean"),
            avg_wait     = ("estimated_wait_time_mins",  "mean"),
            pct_occupied = ("ports_available",lambda x: (x == 0).mean()))
        .reset_index())
    
    peak_data["Period"] = peak_data["is_peak_hour"].map(
        {0: "Off-Peak", 1: "Peak Hour"}
    )

    pk1, pk2, pk3 = st.columns(3)
    for col_w, metric, label, fmt in [
        (pk1, "avg_util",     "Avg Utilization",     ".1%"),
        (pk2, "avg_wait",     "Avg Wait (min)",      ".1f"),
        (pk3, "pct_occupied", "% Fully Occupied",    ".1%"),
    ]:
        fig_pk = px.bar(
            peak_data,
            x="Period",
            y=metric,
            color="Period",
            color_discrete_map={
                "Off-Peak":  "#2166ac",
                "Peak Hour": "#e63946"
            },
            text=peak_data[metric].map(
                lambda v: f"{v:{fmt}}"
            )
        )
        fig_pk.update_traces(textposition="outside",
                              textfont_color="#e8eaf0")
        fig_pk.update_layout(
            title=dict(text=label,
                       font=dict(color="#e8eaf0", size=13)),
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e8eaf0", size=11),
            showlegend=False,
            margin=dict(l=10, r=10, t=40, b=20)
        )
        fig_pk.update_yaxes(
            tickformat=fmt if fmt != ".1f" else None,
            gridcolor="#1e2130", color="#8b92a5"
        )
        fig_pk.update_xaxes(color="#8b92a5")
        col_w.plotly_chart(fig_pk, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# TAB 3 — FORECAST (Regression Model)
# ═══════════════════════════════════════════════════════════

with tab3:

    st.markdown("#### 🔮 Utilization Forecast — 30 Minutes Ahead")

    # ── Station selector ───────────────────────────────────────────────────────
    fc_col1, fc_col2 = st.columns([2, 1])

    with fc_col1:
        forecast_station = st.selectbox(
            "🏪 Select Station to Forecast",
            options=filtered_df["station_id"].unique().tolist(),
            key="forecast_station"
        )

    with fc_col2:
        n_hours = st.slider(
            "⏱️ Hours to Display",
            min_value=24,
            max_value=168,
            value=24,
            step=24,
            key="fc_hours"
        )
        # 168 hours = 7 days maximum

    #n_rows = n_hours * 2

    time_diff = filtered_df["timestamp"].diff().dropna().median()

    rows_per_hour = int(pd.Timedelta(hours=1) / time_diff)

    n_rows = int(n_hours * rows_per_hour)
    # 2 rows per hour (30-min intervals)

    # ── Filter to selected station ─────────────────────────────────────────────
    stn_df = (filtered_df[filtered_df["station_id"] == forecast_station]
              .sort_values("timestamp")
              .reset_index(drop=True))

   # ── Run regression model ───────────────────────────────────────────────────
    # Step 1: Select only the features the model was trained on
    avail_rg = [f for f in rg_features if f in stn_df.columns]
    
    # Step 2: Fill any NaN values
    X_fc = stn_df[avail_rg].fillna(0)

    # Step 3: Run prediction
    preds_reg = rfg_model.predict(X_fc)

    # Step 4: Add predictions to dataframe
    stn_df["predicted_util"] = preds_reg

    # ── Slice to display window ────────────────────────────────────────────────
    display_fc = stn_df.tail(n_rows)
    # .tail(n): last n rows = most recent observations

    # ── Forecast chart ─────────────────────────────────────────────────────────
    fig_fc = go.Figure()
    fig_fc.add_trace(
        go.Scatter(
            x=display_fc["timestamp"],
            y=display_fc["utilization_rate"],
            mode="lines+markers",
            name="Actual Utilization",
            line=dict(color="#7c83e0", width=2),
            marker=dict(size=4, color="#7c83e0")
        )
    )
    fig_fc.add_trace(
        go.Scatter(
            x=display_fc["timestamp"],
            y=display_fc["predicted_util"],
            mode="lines+markers",
            name="Predicted Utilization",
            line=dict(color="#e63946", width=2),
            marker=dict(size=4, color="#e63946")
        )
    )
    fig_fc.update_layout(
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8eaf0", size=12),
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            bgcolor="rgba(0,0,0,0)"
        ),
        showlegend=True
    )
    fig_fc.update_xaxes(gridcolor="#1e2130", color="#8b92a5")
    fig_fc.update_yaxes(gridcolor="#1e2130", color="#8b92a5", tickformat=".0%")
    st.plotly_chart(fig_fc, use_container_width=True)

    # Shade peak hour periods
    

    #peak_rows = display_fc[display_fc["is_peak_hour"] == 1]
    display_fc["peak_group"] = (
    display_fc["is_peak_hour"] != display_fc["is_peak_hour"].shift()
).cumsum()

    #for _, grp in display_fc.groupby("peak_group"):
    #    if grp["is_peak_hour"].iloc[0] == 1:
    #        fig_fc.add_vrect(
    #            x0=grp["timestamp"].min(),
    #         x1=grp["timestamp"].max(),
    #            fillcolor="rgba(230,57,70,0.08)",
    #            line_width=0
    #    )

    # fig_fc.update_layout(
    #     height=420,
    #     paper_bgcolor="rgba(0,0,0,0)",
    #     plot_bgcolor="rgba(0,0,0,0)",
    #     font=dict(color="#e8eaf0", size=12),
    #     hovermode="x unified",
    #     legend=dict(orientation="h", y=1.05,
    #                 bgcolor="rgba(0,0,0,0)"),
    #     margin=dict(l=20, r=20, t=20, b=40),
    #     yaxis=dict(
    #         tickformat=".0%",
    #         range=[0, 1.05],
    #         gridcolor="#1e2130",
    #         color="#8b92a5"
    #     ),
    #     xaxis=dict(gridcolor="#1e2130", color="#8b92a5")
    # )
    # st.plotly_chart(fig_fc, use_container_width=True)

    # ── Station metrics ────────────────────────────────────────────────────────
    #valid = display_fc.dropna(subset=["targets_utilization_t+1","predicted_util"])
    #mae  = np.abs(valid["targets_utilization_t+1"] -
   #               valid["predicted_util"]).mean()
    #rmse = np.sqrt(((valid["targets_utilization_t+1"] -
    #                 valid["predicted_util"])**2).mean())


    max_util = display_fc["predicted_util"].max()

    if max_util > 0.85:
       st.error("⚠️ High congestion expectedv , Expect long wait times")
    elif max_util > 0.6:
       st.warning("⚡ Moderate demand expected , Some delay possible")
    else:
       st.success("✅ Station likely available , Stations mostly free")

    m1, m2, m3 = st.columns(3)
    m1.metric("📊 Avg Predicted", f"{display_fc['predicted_util'].mean():.1%}")
    m2.metric("📈 Peak Predicted", f"{display_fc['predicted_util'].max():.1%}")
    m3.metric("⚡ Min Availability", f"{1 - display_fc['predicted_util'].max():.1%}")

# ═════════════════════════════════════════════════════════════
# TAB 4 — AVAILABILITY (Classification Model)
# ═══════════════════════════════════════════════════════════

with tab4:
    st.markdown("### 🎯 Port Availability Prediction")
    st.markdown(
        f'<div class="info-box">Target: will a port be free at t+1?'
        f'· </div>',
        unsafe_allow_html=True
    )
    # ── Station selector ───────────────────────────────────────────────────────
    av_col1, av_col2 = st.columns([2, 1])
    with av_col1:
        avail_station = st.selectbox(
            "🏪 Select Station to Predict",
            options=filtered_df["station_id"].unique().tolist(),
            key="avail_station"
        )
    with av_col2:
        avail_hours  = st.slider(
            "⏱️ Hours to Display",
            min_value=24,
            max_value=168,
            value=48,
            step=24,
            key="avail_hours"
        )
    # ── Filter station data ─────────────────────────────────────────────────────
    av_df = (filtered_df[filtered_df["station_id"] == avail_station]
             .sort_values("timestamp")
             .reset_index(drop=True))
    # ── Run classification model ───────────────────────────────────────────────
    avail_clf = [f for f in clf_features if f in av_df.columns]
    X_av = av_df[avail_clf].fillna(0)
     # Get probability of port being free (class 1)
    probs  = clf_model.predict_proba(X_av)[:, 1]

    # Apply optimal threshold
    labels = (probs >= THRESHOLD).astype(int)
    # 1 = predicted available, 0 = predicted occupied

    av_df["avail_prob"]  = probs
    av_df["avail_pred"]  = labels

     # Slice to display window
    display_av = av_df.tail(avail_hours * 2)

    # ── Probability chart ───────────────────────────────────────────────────────
    fig_av = go.Figure()

    # Probability area fill
    fig_av.add_trace(go.Scatter(
        x=display_av["timestamp"],
        y=display_av["avail_prob"],
        mode="lines",
        fill="tozeroy",
        fillcolor="rgba(77,222,157,0.12)",
        line=dict(color="#4dde9d", width=2),
        name="P(port available)"
    ))

    # Actual availability (0 or 1)
    fig_av.add_trace(go.Scatter(
        x=display_av["timestamp"],
        y=display_av["target_avail_t1"],
        mode="lines",
        line=dict(color="#7c83e0", width=1, dash="dot"),
        opacity=0.6,
        name="Actual (0/1)"
    ))

    # Threshold line
    fig_av.add_hline(
        y=THRESHOLD,
        line_dash="dash",
        line_color="#e63946",
        line_width=1.5,
        annotation_text=f"Threshold ({THRESHOLD:.2f})",
        annotation_position="bottom right",
        annotation_font_color="#e63946"
    )

    fig_av.update_layout(
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8eaf0", size=12),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.05,
                    bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=20, r=20, t=20, b=40),
        yaxis=dict(range=[-0.05, 1.1],
                   gridcolor="#1e2130", color="#8b92a5"),
        xaxis=dict(gridcolor="#1e2130", color="#8b92a5")
    )
    st.plotly_chart(fig_av, use_container_width=True)

    # ── Metrics + Confusion Matrix ─────────────────────────────────────────────
    # left_col, right_col = st.columns([1, 1])

    valid_av = display_av.dropna(
         subset=["target_avail_t1", "avail_pred"])

    # with left_col:
    #     st.markdown("#### 📊 Prediction Metrics")
    av_f1   = f1_score(valid_av["target_avail_t1"],
                             valid_av["avail_pred"],
                             zero_division=0)
    av_prec = precision_score(valid_av["target_avail_t1"],
                          valid_av["avail_pred"],
                          zero_division=0)

    av_rec  = recall_score(valid_av["target_avail_t1"],
                       valid_av["avail_pred"],
                       zero_division=0)

    av_acc  = (valid_av["target_avail_t1"] ==
           valid_av["avail_pred"]).mean()

    # st.write(f"Precision: {av_prec:.3f}")
    # st.write(f"Recall: {av_rec:.3f}")
    # st.write(f"Accuracy: {av_acc:.3f}")

    # Colour-coded status
    if av_f1 >= 0.85:
        status_class = "success-box"
        status_text  = f"✅ Excellent F1 = {av_f1:.3f}"

    elif av_f1 >= 0.70:
        status_class = "info-box"
        status_text  = f"ℹ️ Good F1 = {av_f1:.3f}"

    else:
        status_class = "warning-box"
        status_text  = f"⚠️ Low F1 = {av_f1:.3f}"

    # ✅ Always render
    st.markdown(
        f'<div class="{status_class}">{status_text}</div>',unsafe_allow_html=True)

    
    # st.markdown("#### 🎯 Confusion Matrix")
    # from sklearn.metrics import confusion_matrix as cm_fn
    # cm_vals = cm_fn(valid_av["target_avail_t1"],
    #                     valid_av["avail_pred"])
    # fig_cm = px.imshow(
    #         cm_vals,
    #         text_auto=True,
    #         color_continuous_scale=["#0d1117","#2166ac","#7c83e0"],
    #         labels=dict(x="Predicted", y="Actual"),
    #         x=["Occupied", "Available"],
    #         y=["Occupied", "Available"]
    #     )
    # fig_cm.update_layout(
    #         height=270,
    #         paper_bgcolor="rgba(0,0,0,0)",
    #         plot_bgcolor="rgba(0,0,0,0)",
    #         font=dict(color="#e8eaf0", size=12),
    #         margin=dict(l=20, r=20, t=20, b=20),
    #         coloraxis_showscale=False
    #     )
    # st.plotly_chart(fig_cm, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# TAB 5 — STATION MAP
# ═══════════════════════════════════════════════════════════

with tab5:

    st.markdown("### 🗺️ Station Performance Map")

    # ── Build station summary ──────────────────────────────────────────────────
    stn_summary = (filtered_df
        .groupby(["station_id", "latitude", "longitude",
                  "city", "network", "location_type"])
        .agg(
            avg_util     = ("utilization_rate",         "mean"),
            pct_occupied = ("ports_available",
                             lambda x: (x == 0).mean()),
            avg_wait     = ("estimated_wait_time_mins", "mean"),
            n_obs        = ("utilization_rate",         "count")
        )
        .reset_index())

    # ── Colour metric selector ─────────────────────────────────────────────────
    color_col, _, _2 = st.columns([2, 2, 3])
    with color_col:
        color_by = st.selectbox(
            "🎨 Colour stations by",
            options=["avg_util", "pct_occupied", "avg_wait"],
            format_func=lambda x: {
                "avg_util"    : "Avg Utilization",
                "pct_occupied": "% Fully Occupied",
                "avg_wait"    : "Avg Wait Time (min)"
            }[x]
            # format_func: display readable label, store raw column name
        )

    # ── Bubble map ─────────────────────────────────────────────────────────────
    fig_map = px.scatter_mapbox(
        stn_summary,
        lat="latitude",
        lon="longitude",
        color=color_by,
        size="avg_util",
        # Bubble size = utilization → busier stations are bigger
        size_max=22,
        color_continuous_scale=[
            [0.0, "#1a6e1a"],
            [0.4, "#2166ac"],
            [0.7, "#f4a261"],
            [1.0, "#e63946"]
        ],
        # Green=low, Blue=medium, Orange=high, Red=very high
        hover_name="station_id",
        hover_data={
            "city"        : True,
            "network"     : True,
            "avg_util"    : ":.1%",
            "avg_wait"    : ":.1f",
            "latitude"    : False,
            "longitude"   : False
        },
        mapbox_style="carto-darkmatter",
        # Dark map tiles — matches our dark theme
        zoom=3,
        height=520
    )
    fig_map.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar=dict(
            title=dict(
                text=color_by.replace("_", " ").title(),
                font=dict(color="#8b92a5", size=11)
            ),
            tickfont=dict(color="#8b92a5", size=10)
        )
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("---")

    # ── Station leaderboard ────────────────────────────────────────────────────
    st.markdown("### 🏆 Station Leaderboard")

    sort_by = st.selectbox(
        "Sort by",
        options=["avg_util", "pct_occupied", "avg_wait"],
        format_func=lambda x: {
            "avg_util"    : "Avg Utilization",
            "pct_occupied": "% Fully Occupied",
            "avg_wait"    : "Avg Wait (min)"
        }[x],
        key="sort_leaderboard"
    )

    leaderboard = (stn_summary
        .sort_values(sort_by, ascending=False)
        [["station_id","city","network",
          "location_type","avg_util",
          "pct_occupied","avg_wait"]]
        .rename(columns={
            "station_id"   : "Station",
            "city"         : "City",
            "network"      : "Network",
            "location_type": "Location",
            "avg_util"     : "Avg Util",
            "pct_occupied" : "% Occupied",
            "avg_wait"     : "Avg Wait (min)"
        })
        .reset_index(drop=True)
    )

    st.dataframe(
        leaderboard.style
        .format({
            "Avg Util"     : "{:.1%}",
            "% Occupied"   : "{:.1%}",
            "Avg Wait (min)": "{:.1f}"
        })
        .background_gradient(
            subset=["Avg Util"],
            cmap="RdYlGn_r"
            # Red=high utilization, Green=low utilization
        )
        .set_properties(**{
            "background-color": "#1a1d2e",
            "color"           : "#e8eaf0",
            "border-color"    : "#2d3250"
        }),
        use_container_width=True,
        height=400
    )
    

with tab6:
    st.markdown("### 🤖 AI Assistant - Chatbot")
    st.markdown(
        f'<div class="info-box">Ask questions about EV charging trends, station performance, or get personalized recommendations based on the data.'
        f'· </div>',
        unsafe_allow_html=True
    )
    # How it works explainer
#     with st.expander("ℹ️ How does this chatbot work?"):
#         e1, e2 = st.columns(2)
#         e1.markdown("""
# **RAG — Retrieval Augmented Generation:**
# 1. You type a question
# 2. Keywords in your question are matched to real statistics
# 3. Matching statistics are attached to your question
# 4. Claude reads those real numbers and answers from them
# 5. Follow-up questions remember earlier context
# """)
#         e2.markdown("""
# **Topics this chatbot knows:**
# - City & network utilization rates
# - Peak hour demand patterns
# - Weather & traffic impact
# - Pricing strategy insights
# - ML model performance metrics
# - Operational recommendations
# - Maintenance windows
# """)

#     st.markdown("---")
#      # Claude system prompt
#     SYSTEM_PROMPT = """You are an expert EV charging network data analyst.
#         You have real statistics from 150 stations across 15 US cities, Jul-Dec 2025.
#      Rules:
# 1. Answer ONLY using the DATA CONTEXT provided — cite real numbers
# 2. Never invent statistics not in the context
# 3. 3-5 sentences unless the user asks for more detail
# 4. Professional, direct tone — no filler phrases
# 5. Ground all recommendations in actual data patterns
# """

#         # Session state — persists across Streamlit reruns
    if "chat_display" not in st.session_state:
        st.session_state.chat_display = []
    # chat_display: what the USER sees on screen

    if "chat_api" not in st.session_state:
        st.session_state.chat_api = []
    # chat_api: full augmented messages sent to Claude
    # includes data context prepended to each question

    # Suggested questions (only shown when chat is empty)
    if not st.session_state.chat_display:
       
        # st.markdown(
        #     "<p style=\\"color:#8b92a5;font-size:13px\\">"
        #     "💡 <b>Click a question to get started:</b></p>",
        #     unsafe_allow_html=True)
        suggestions = [
            "Which city has the highest utilization?",
            "When are stations most congested?",
            "How does rain affect charging demand?",
            "Which network performs best?",
            "What pricing strategy do you recommend?",
            "When is the best time for maintenance?",
            "How do weekdays and weekends compare?",
        ]
        sg1, sg2 = st.columns(2)
        for i, sug in enumerate(suggestions):
            if (sg1 if i%2==0 else sg2).button(
                    sug, key=f"sg_{i}",
                    use_container_width=True):
                st.session_state.pending_q = sug

    # Display all previous chat messages
    for msg in st.session_state.chat_display:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input bar at the bottom
    user_input = st.chat_input(
        "Ask about demand, availability, pricing, or model results..."
    )

    # Handle suggestion button clicks
    if "pending_q" in st.session_state:
        user_input = st.session_state.pop("pending_q")

    # Process new question
    if user_input:

        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.chat_display.append(
            {"role": "user", "content": user_input})

        # Get Claude answer with spinner
        with st.chat_message("assistant"):
            with st.spinner("🔍 Retrieving data and generating answer..."):
                try:
                    # Step 1: Retrieve relevant data context
                    ctx = rag_retrieve(user_input, max_chunks=3)
                    # rag_retrieve: defined at top of file
                    # Returns text blocks matching the question keywords

                    # Step 2: Build augmented prompt (data + question)
                    augmented = (
                        f"DATA CONTEXT (real stats from EV dataset):\\n"
                        f"{ctx}\\n\\n"
                        f"QUESTION: {user_input}\\n\\n"
                        f"Answer using only the data context above. "
                        f"Cite specific numbers."
                    )

                    # Step 3: Add to full conversation history
                    messages = st.session_state.chat_api + [
                        {"role": "user", "content": augmented}
                    ]

                    # Step 4: Call Claude API
                    # ai_client = anthropic.Anthropic(
                    #     api_key=os.environ.get("","")
                    # )
                    # response  = ai_client.messages.create(
                    #     model      = "claude-opus-4-5",
                    #     max_tokens = 1024,
                    #     system     = SYSTEM_PROMPT,
                    #     messages   = messages
                    # )

                    # Alternative: Google Gemini API
                    genai.configure(api_key="AIzaSyDj95CBwQ5_dOky-T8YzkiaEUgZfkbFwMA")
                    model = genai.GenerativeModel("gemini-1.0")
                    response = model.generate_content("Explain EV charging demand prediction")
                    print(response.text)

                    # Step 5: Extract answer text
                    answer = response.text
                except Exception as e:
                    answer = f"⚠️ **Error:** {str(e)}"

            st.markdown(answer)

        # Update display history
        st.session_state.chat_display.append(
            {"role": "assistant", "content": answer})

        # Update API history with augmented question
        st.session_state.chat_api.extend([
            {"role": "user",
             "content": (f"DATA CONTEXT:\\n"
                         f"{rag_retrieve(user_input)}\\n\\n"
                         f"QUESTION: {user_input}")},
            {"role": "assistant", "content": answer}
        ])

        # Rerun to scroll to new message
        st.rerun()

    # Footer controls
    if st.session_state.chat_display:
        st.markdown("---")
        fc1, fc2, fc3 = st.columns([1,1,4])
        if fc1.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_display = []
            st.session_state.chat_api     = []
            st.rerun()
        n = len(st.session_state.chat_display) // 2
        fc3.caption(
            f"💬 {n} turn(s) · "
            f"{len(context_lib)} data chunks · "
            f"Model: Gemini 1.0")