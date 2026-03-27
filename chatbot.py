import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT LIBRARY = pre-computed real statistics stored as text chunks
# Used by the RAG chatbot to ground answers in actual data
# Each chunk covers one topic (cities, peak hours, weather...)
# ─────────────────────────────────────────────────────────────────────────────

def build_context_library(df):
    """
    Compute statistics from the real dataset.
    Store as a dictionary of text blocks.
    Each block covers one topic the chatbot might be asked about.
    """
    ctx = {}

    # ── Chunk 1: Overall summary ───────────────────────────────────────────────
    ctx['overall'] = f"""
DATASET OVERVIEW:
- Stations : {df['station_id'].nunique()}
- Cities   : {df['city'].nunique()} — {', '.join(sorted(df['city'].unique()))}
- Networks : {df['network'].nunique()} — {', '.join(sorted(df['network'].unique()))}
- Period   : {df['timestamp'].min().date()} to {df['timestamp'].max().date()}
- Records  : {len(df):,} (30-min intervals)
- Avg utilization  : {df['utilization_rate'].mean():.1%}
- Fully occupied % : {(df['ports_available']==0).mean():.1%}
- Avg wait time    : {df['estimated_wait_time_mins'].mean():.1f} min
- Max wait time    : {df['estimated_wait_time_mins'].max():.0f} min
"""

    # ── Chunk 2: Peak hours ────────────────────────────────────────────────────
    peak    = df[df['is_peak_hour']==1]['utilization_rate']
    offpeak = df[df['is_peak_hour']==0]['utilization_rate']
    hourly  = df.groupby('hour_of_day')['utilization_rate'].mean()

    ctx['peak_hours'] = f"""
PEAK vs OFF-PEAK:
- Peak hours   : 7-9 AM and 5-8 PM
- Peak avg util    : {peak.mean():.1%}
- Off-peak avg util: {offpeak.mean():.1%}
- Difference       : {peak.mean()-offpeak.mean():.1%} higher during peak
- Peak % full      : {(df[df['is_peak_hour']==1]['ports_available']==0).mean():.1%}
- Off-peak % full  : {(df[df['is_peak_hour']==0]['ports_available']==0).mean():.1%}
- Busiest hour     : {hourly.idxmax()}:00 ({hourly.max():.1%})
- Quietest hour    : {hourly.idxmin()}:00 ({hourly.min():.1%})
- Weekend peak util: {df[(df['is_weekend']==1)&(df['is_peak_hour']==1)]['utilization_rate'].mean():.1%}
- Weekday peak util: {df[(df['is_weekend']==0)&(df['is_peak_hour']==1)]['utilization_rate'].mean():.1%}
"""

    # ── Chunk 3: Cities ────────────────────────────────────────────────────────
    city_stats = (df.groupby('city')['utilization_rate']
                    .agg(['mean','median','std'])
                    .sort_values('mean', ascending=False))
    city_lines = '\n'.join([
        f"  {c:22s}: avg={r['mean']:.1%} median={r['median']:.1%}"
        for c, r in city_stats.iterrows()
    ])
    ctx['cities'] = f"""
UTILIZATION BY CITY (highest to lowest):
{city_lines}
- Busiest city : {city_stats['mean'].idxmax()} ({city_stats['mean'].max():.1%})
- Quietest city: {city_stats['mean'].idxmin()} ({city_stats['mean'].min():.1%})
"""

    # ── Chunk 4: Networks ──────────────────────────────────────────────────────
    net_stats = (df.groupby('network')
                   .agg(avg_util=('utilization_rate','mean'),
                        avg_wait=('estimated_wait_time_mins','mean'),
                        pct_full=('ports_available', lambda x:(x==0).mean()))
                   .sort_values('avg_util', ascending=False))
    net_lines = '\n'.join([
        f"  {n:20s}: util={r['avg_util']:.1%} wait={r['avg_wait']:.1f}min"
        for n, r in net_stats.iterrows()
    ])
    ctx['networks'] = f"""
NETWORK PERFORMANCE:
{net_lines}
- Best network : {net_stats['avg_util'].idxmax()} ({net_stats['avg_util'].max():.1%})
- Worst network: {net_stats['avg_util'].idxmin()} ({net_stats['avg_util'].min():.1%})
"""

    # ── Chunk 5: Location type ─────────────────────────────────────────────────
    loc_stats = (df.groupby('location_type')['utilization_rate']
                   .mean().sort_values(ascending=False))
    loc_lines = '\n'.join([f"  {l:25s}: {v:.1%}"
                            for l,v in loc_stats.items()])
    ctx['location_type'] = f"""
UTILIZATION BY LOCATION TYPE:
{loc_lines}
- Busiest location : {loc_stats.idxmax()} ({loc_stats.max():.1%})
- Quietest location: {loc_stats.idxmin()} ({loc_stats.min():.1%})
"""

    # ── Chunk 6: Weather ───────────────────────────────────────────────────────
    if 'weather_condition' in df.columns:
        weather_stats = (df.groupby('weather_condition')['utilization_rate']
                           .mean().sort_values(ascending=False))
        weather_lines = '\n'.join([f"  {w:20s}: {v:.1%}"
                                    for w,v in weather_stats.items()])
    else:
        wc_cols = sorted([c for c in df.columns if c.startswith('weather_condition_')])
        if wc_cols:
            weather_stats = {}
            for c in wc_cols:
                cond = c.replace('weather_condition_', '')
                vals = df.loc[df[c] == 1, 'utilization_rate']
                weather_stats[cond] = vals.mean() if len(vals) else float('nan')
            weather_stats = dict(sorted(weather_stats.items(), key=lambda x: x[1], reverse=True))
            weather_lines = '\n'.join([f"  {w:20s}: {v:.1%}" if pd.notna(v) else f"  {w:20s}: N/A"
                                        for w,v in weather_stats.items()])
        else:
            weather_lines = "  No weather_condition data available."

    temp_corr = df['temperature_f'].corr(df['utilization_rate']) if 'temperature_f' in df.columns else None
    precip_corr = df['precipitation_mm'].corr(df['utilization_rate']) if 'precipitation_mm' in df.columns else None

    temp_corr_str = f"{temp_corr:+.3f}" if temp_corr is not None else "N/A"
    precip_corr_str = f"{precip_corr:+.3f}" if precip_corr is not None else "N/A"

    ctx['weather'] = f"""
WEATHER IMPACT:
{weather_lines}
- Temperature correlation: {temp_corr_str}
- Precipitation correlation: {precip_corr_str}
"""

    # ── Chunk 7: Pricing & traffic ─────────────────────────────────────────────
    ctx['pricing_traffic'] = f"""
PRICING AND TRAFFIC:
- Avg price ($/kWh): ${df['current_price'].mean():.3f}
- Price range: ${df['current_price'].min():.3f} to ${df['current_price'].max():.3f}
- Price vs utilization correlation: {df['current_price'].corr(df['utilization_rate']):+.3f}
- Avg traffic congestion index: {df['traffic_congestion_index'].mean():.2f}
- Traffic vs utilization correlation: {df['traffic_congestion_index'].corr(df['utilization_rate']):+.3f}
- Avg gas price: ${df['gas_price_per_gallon'].mean():.3f}/gallon
- EV demand is relatively price-inelastic (captive market)
"""

    # ── Chunk 8: Charger types ─────────────────────────────────────────────────
    charger_stats = (df.groupby('charger_type')
                       .agg(avg_util=('utilization_rate','mean'),
                            avg_power=('power_output_kw','mean'),
                            avg_wait=('estimated_wait_time_mins','mean'))
                       .sort_values('avg_util', ascending=False))
    charger_lines = '\n'.join([
        f"  {c:15s}: util={r['avg_util']:.1%} power={r['avg_power']:.0f}kW wait={r['avg_wait']:.1f}min"
        for c,r in charger_stats.iterrows()
    ])
    ctx['charger_type'] = f"""
CHARGER TYPE PERFORMANCE:
{charger_lines}
"""

    # ── Chunk 9: ML model results ──────────────────────────────────────────────

    thresh = 0.50

    ctx['ml_models'] = f"""
ML MODEL PERFORMANCE:

REGRESSION (predict utilization at t+1):
- Target   : utilization_rate 30 min ahead
- Best model: Random Forest Regressor (tuned)
- Key features: lag_1_utilization, rolling_3h, station_hour_baseline

CLASSIFICATION (predict port free at t+1):
- Target   : 1=port free, 0=fully occupied
- Best model: Random Forest Classifier (tuned in Task 4)
- Threshold : {thresh:.2f} (optimised for max F1)
- Key features: lag_1_ports_available, lag_1_utilization, traffic

VALIDATION:
- Method: Walk-forward TimeSeriesSplit (5 folds, 24h gap)
- Train: Jul-Sep | Val: Oct-Nov | Test: Dec 2025
- No leakage: all targets shifted to t+1
"""

    # ── Chunk 10: Operations ───────────────────────────────────────────────────
    top_wait = (df.groupby('station_id')['estimated_wait_time_mins']
                  .mean().nlargest(3))
    hourly   = df.groupby('hour_of_day')['utilization_rate'].mean()
    q_hour   = hourly.idxmin()

    ctx['operations'] = f"""
OPERATIONAL INSIGHTS:

HIGHEST WAIT TIME STATIONS:
{chr(10).join([f"  {s}: {w:.1f} min avg" for s,w in top_wait.items()])}

MAINTENANCE WINDOW:
- Best time: {q_hour}:00-{(q_hour+2)%24}:00
- Utilization at this hour: {hourly[q_hour]:.1%} (lowest demand)

RECOMMENDATIONS:
- Surge pricing when utilization > 85%
- Off-peak discount when utilization < 30%
- Add stations in top 3 busiest cities
- Prioritise DC fast chargers at highway locations
"""

    return ctx


# ── Context library builder helper (used by app.py) ───────────────────────────
# Note: do not execute at import time, because app.py owns the dataset (df).
# In app.py, do:
#     from chatbot import build_context_library
#     context_lib = build_context_library(df)
