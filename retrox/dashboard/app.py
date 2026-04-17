"""
RetroX — Dengue Early Warning & Data Intelligence Platform v3
=============================================================
Production-ready · Premium Dark UI · Multi-model Forecasting · Universal Data Lab
"""
from __future__ import annotations

import json
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrox.data.dengai import dengai_files_available, load_dengai_train, write_demo_dataset
from retrox.features.engineering import FeatureParams, last_complete_feature_row
from retrox.features.ers import classify_ers
from retrox.models.inference import load_latest_artifact, predict_next
from retrox.models.registry import default_model_dir
from retrox.models.training import train_model
from retrox.dashboard.lab_pipeline import (
    render_smart_eda,
    render_preprocessing_controls,
    render_tsne,
    render_automl,
    render_suggestions,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config — MUST be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RetroX — Dengue Intelligence Platform",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Premium CSS — injected via st.html for reliability, not st.markdown
# ─────────────────────────────────────────────────────────────────────────────
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* ── Core background ── */
.stApp {
  background: #070b14 !important;
  background-image:
    radial-gradient(ellipse at 20% 50%, rgba(79,70,229,0.06) 0%, transparent 60%),
    radial-gradient(ellipse at 80% 20%, rgba(124,58,237,0.05) 0%, transparent 50%) !important;
}
section[data-testid="stSidebar"] {
  background: #0a0e1c !important;
  border-right: 1px solid #1a2744 !important;
}
.main .block-container {
  padding-top: 1.5rem !important;
  max-width: 1400px !important;
}

/* ── Typography ── */
h1, .st-emotion-cache-10trblm {
  background: linear-gradient(135deg, #818cf8 0%, #c084fc 45%, #38bdf8 100%) !important;
  -webkit-background-clip: text !important;
  -webkit-text-fill-color: transparent !important;
  background-clip: text !important;
  font-weight: 800 !important;
  font-size: 2rem !important;
  letter-spacing: -0.02em !important;
}
h2 { color: #93c5fd !important; font-weight: 700 !important; font-size: 1.3rem !important; margin-top: 0.5rem !important; }
h3 { color: #a5b4fc !important; font-weight: 600 !important; }
p, span, li, label, div { color: #c8d4e8 !important; }

/* ── Sidebar text ── */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div { color: #94a3b8 !important; }

/* ── Metric cards ── */
div[data-testid="stMetric"] {
  background: linear-gradient(135deg, #111827 0%, #1a2540 100%) !important;
  border: 1px solid #1f3050 !important;
  border-radius: 16px !important;
  padding: 1.2rem 1.4rem !important;
  transition: transform 0.18s ease, box-shadow 0.18s ease !important;
  box-shadow: 0 4px 24px rgba(0,0,0,0.35) !important;
}
div[data-testid="stMetric"]:hover {
  transform: translateY(-3px) !important;
  box-shadow: 0 8px 36px rgba(129,140,248,0.18) !important;
  border-color: #3b4f80 !important;
}
div[data-testid="stMetricLabel"] > div { color: #64748b !important; font-size: 0.72rem !important; font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: 0.06em !important; }
div[data-testid="stMetricValue"] > div { color: #f1f5f9 !important; font-size: 1.7rem !important; font-weight: 700 !important; }
div[data-testid="stMetricDelta"] > div { font-size: 0.8rem !important; }

/* ── Navigation tabs ── */
div[data-baseweb="tab-list"] {
  background: #0c1120 !important;
  border-radius: 14px !important;
  padding: 5px !important;
  gap: 4px !important;
  border: 1px solid #1a2744 !important;
}
button[data-baseweb="tab"] {
  color: #475569 !important;
  font-weight: 500 !important;
  border-radius: 10px !important;
  padding: 10px 22px !important;
  font-size: 0.875rem !important;
  transition: all 0.15s !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
  background: linear-gradient(135deg, #1e3560 0%, #1a2d50 100%) !important;
  color: #818cf8 !important;
  font-weight: 700 !important;
  border-bottom: 2px solid #818cf8 !important;
  box-shadow: 0 2px 12px rgba(129,140,248,0.2) !important;
}

/* ── Buttons ── */
button[data-testid="baseButton-secondary"],
.stButton > button {
  background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 10px !important;
  padding: 10px 24px !important;
  font-weight: 600 !important;
  font-size: 0.88rem !important;
  letter-spacing: 0.02em !important;
  box-shadow: 0 4px 16px rgba(79,70,229,0.4) !important;
  transition: all 0.18s ease !important;
}
.stButton > button:hover {
  background: linear-gradient(135deg, #4338ca 0%, #6d28d9 100%) !important;
  box-shadow: 0 6px 24px rgba(79,70,229,0.55) !important;
  transform: translateY(-1px) !important;
}
button[data-testid="stDownloadButton"] {
  background: linear-gradient(135deg, #0f766e 0%, #065f46 100%) !important;
  box-shadow: 0 4px 16px rgba(5,150,105,0.3) !important;
}

/* ── Selectbox / inputs ── */
div[data-baseweb="select"] > div {
  background: #111827 !important;
  border: 1px solid #1f3050 !important;
  border-radius: 8px !important;
  color: #e2e8f0 !important;
}
div[data-baseweb="select"] * { color: #e2e8f0 !important; }
input, textarea { background: #111827 !important; color: #e2e8f0 !important; border-color: #1f3050 !important; border-radius: 8px !important; }

/* ── Sliders ── */
div[data-testid="stSlider"] div[role="slider"] { background: #818cf8 !important; }

/* ── Progress bar ── */
div[data-testid="stProgress"] div {
  background: linear-gradient(90deg, #4f46e5, #7c3aed, #c084fc) !important;
  border-radius: 4px !important;
}

/* ── Expanders ── */
div[data-testid="stExpander"] {
  background: #0d1120 !important;
  border: 1px solid #1a2744 !important;
  border-radius: 12px !important;
  margin-bottom: 8px !important;
  overflow: hidden !important;
}
summary span { color: #93c5fd !important; font-weight: 600 !important; }
summary:hover { background: rgba(129,140,248,0.05) !important; }

/* ── Alert boxes ── */
div[data-testid="stAlert"] {
  border-radius: 10px !important;
}
div.stInfo {
  background: rgba(79,70,229,0.09) !important;
  border: 1px solid rgba(129,140,248,0.25) !important;
}
div.stInfo * { color: #a5b4fc !important; }
div.stWarning {
  background: rgba(245,158,11,0.09) !important;
  border: 1px solid rgba(251,191,36,0.25) !important;
}
div.stWarning * { color: #fcd34d !important; }
div.stSuccess {
  background: rgba(16,185,129,0.09) !important;
  border: 1px solid rgba(52,211,153,0.25) !important;
}
div.stSuccess * { color: #6ee7b7 !important; }
div.stError {
  background: rgba(239,68,68,0.09) !important;
  border: 1px solid rgba(252,165,165,0.25) !important;
}
div.stError * { color: #fca5a5 !important; }

/* ── Dataframe ── */
div[data-testid="stDataFrame"] {
  border: 1px solid #1a2744 !important;
  border-radius: 10px !important;
  overflow: hidden !important;
}

/* ── File uploader ── */
section[data-testid="stFileUploaderDropzone"] {
  background: #0d1120 !important;
  border: 2px dashed #1f3050 !important;
  border-radius: 14px !important;
}
section[data-testid="stFileUploaderDropzone"]:hover {
  border-color: #818cf8 !important;
  background: rgba(79,70,229,0.04) !important;
}

/* ── Radio buttons ── */
div[data-testid="stRadio"] label { color: #94a3b8 !important; }
div[data-testid="stRadio"] div[role="radio"][aria-checked="true"] ~ div { color: #818cf8 !important; }

/* ── Checkboxes ── */
div[data-testid="stCheckbox"] label { color: #94a3b8 !important; }

/* ── Divider ── */
hr { border-color: #1a2744 !important; margin: 1.5rem 0 !important; }

/* ── Spinner ── */
div[data-testid="stSpinner"] > div {
  border-top-color: #818cf8 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0a0e1c; }
::-webkit-scrollbar-thumb { background: #1f3050; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #4f46e5; }

/* ── Caption ── */
small, .stCaption, div[data-testid="stCaptionContainer"] { color: #475569 !important; }

/* ── Code blocks ── */
code, pre { background: #0c1524 !important; border: 1px solid #1a2744 !important; border-radius: 6px !important; color: #7dd3fc !important; }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _read_file(f) -> pd.DataFrame:
    f.seek(0)
    name = f.name.lower()
    if name.endswith(".csv"):
        for enc in ("utf-8", "utf-8-sig", "latin1", "cp1252"):
            try:
                f.seek(0)
                return pd.read_csv(f, encoding=enc)
            except Exception:
                continue
        f.seek(0)
        return pd.read_csv(f, engine="python", sep=None)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(f)
    if name.endswith(".json"):
        return pd.read_json(f)
    if name.endswith(".parquet"):
        return pd.read_parquet(f)
    raise ValueError(f"Unsupported format: {name}")


def _artifact_safe(*, city: str, horizon_weeks: int):
    try:
        return load_latest_artifact(default_model_dir(city=city, horizon_weeks=horizon_weeks),
                                     name="random_forest"), None
    except FileNotFoundError as e:
        return None, str(e)
    except Exception as e:
        return None, f"Error: {e}"


def _ensure_demo():
    if not dengai_files_available():
        with st.spinner("Initialising DengAI demo dataset …"):
            write_demo_dataset()


def _dark_chart(fig):
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#07090f",
        paper_bgcolor="#07090f",
        font=dict(color="#c8d4e8", family="Inter"),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:16px 0 24px;">
      <div style="font-size:3rem;line-height:1;margin-bottom:8px;">🦟</div>
      <div style="font-size:1.3rem;font-weight:800;background:linear-gradient(135deg,#818cf8,#c084fc);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent;">RetroX</div>
      <div style="font-size:0.75rem;color:#475569;margin-top:4px;letter-spacing:0.05em;">
        DENGUE INTELLIGENCE PLATFORM
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    city = st.selectbox(
        "🌆 City",
        ["sj", "iq"], index=0,
        format_func=lambda x: "🇵🇷 San Juan (sj)" if x == "sj" else "🇵🇪 Iquitos (iq)",
    )
    horizon = st.slider("📅 Forecast Horizon (weeks)", 1, 12, 4)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.76rem;color:#334155;line-height:1.7;">
      <strong style="color:#475569;">Tip:</strong> Models auto-select the best of
      HistGBM · GradBoost · Random Forest · ExtraTrees · Ridge via time-series CV.
      <br><br>
      <strong style="color:#475569;">Data Lab:</strong> Upload any tabular dataset
      (CSV, Excel, JSON, Parquet) to run the full AI pipeline.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Built with Streamlit · scikit-learn · Plotly")

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:1.5rem;">
  <h1 style="margin:0;padding:0;">🦟 RetroX — Dengue Intelligence Platform</h1>
  <p style="color:#475569;margin:4px 0 0;font-size:0.88rem;letter-spacing:0.01em;">
    AI-powered epidemiological forecasting · Self-service analytics · Production-ready
  </p>
</div>
""", unsafe_allow_html=True)

tab_fc, tab_lab = st.tabs([
    "📡  Forecast & Alerting",
    "🔬  Data Lab  ·  EDA · t-SNE · AutoML · Insights",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Forecast & Alerting
# ─────────────────────────────────────────────────────────────────────────────
with tab_fc:
    st.markdown("### 📥 Data Source")

    mode = st.radio(
        "data_source", ["🦟 DengAI built-in data (last 60 weeks)", "📁 Upload any CSV / Excel"],
        horizontal=True, label_visibility="collapsed",
    )

    df: pd.DataFrame | None = None

    # ── DengAI built-in ───────────────────────────────────────────────────────
    if mode.startswith("🦟"):
        _ensure_demo()
        try:
            full_df = load_dengai_train(city=city)
            df      = full_df.tail(60).reset_index(drop=True)
            st.success(f"✅ DengAI training data loaded for **{city.upper()}** — {len(df):,} rows")
        except Exception as e:
            st.error(f"Could not load DengAI data: {e}")

    # ── Upload any file ───────────────────────────────────────────────────────
    else:
        up = st.file_uploader(
            "Upload any tabular time-series (CSV/Excel). You can map your target column below.",
            type=["csv", "xlsx", "xls"],
            key="fc_upload",
        )
        if not up:
            st.info("⬆️ Upload a file to continue.")
        else:
            try:
                df = _read_file(up)
                st.success(f"✅ `{up.name}` — {len(df):,} rows × {df.shape[1]} columns")
            except Exception as e:
                st.error(f"Read error: {e}")

    # ── Universal Mapping & Configuration ────────────────────────────────────
    target_col = "total_cases"
    date_col   = "week_start_date"

    if df is not None:
        with st.expander("⚙️ Forecasting Configuration & Column Mapping", expanded=True):
            all_cols = df.columns.tolist()
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            c1, c2 = st.columns(2)
            with c1:
                if not num_cols:
                    st.error("No numeric columns found in this dataset. Forecasting requires a numeric target.")
                    st.stop()
                y_col = st.selectbox(
                    "🎯 Target Column (what to forecast)", 
                    num_cols, 
                    index=num_cols.index("total_cases") if "total_cases" in num_cols else 0
                )
            with c2:
                # Try to find date columns
                date_candidates = [c for c in all_cols if any(k in c.lower() for k in ["date", "time", "week", "year"])]
                d_col = st.selectbox(
                    "📅 Time/Date Column (optional)", 
                    ["None"] + all_cols,
                    index=all_cols.index(date_candidates[0]) + 1 if date_candidates else 0
                )
                date_col = d_col if d_col != "None" else None

        with st.expander("📋 Data Preview (last 10 rows)", expanded=False):
            st.dataframe(df.tail(10), width="stretch")

        if y_col:
            fig_hist = go.Figure()
            # Simple line chart for history
            fig_hist.add_trace(go.Scatter(
                y=df[y_col].values, mode="lines+markers",
                fill="tozeroy", name=y_col,
                line=dict(color="#818cf8", width=2),
                marker=dict(size=3, color="#818cf8"),
                fillcolor="rgba(129,140,248,0.1)",
            ))
            fig_hist.update_layout(
                title=f"Historical Series — {y_col}",
                xaxis_title="Timeline index", yaxis_title=y_col,
                **dict(template="plotly_dark", plot_bgcolor="#07090f", paper_bgcolor="#07090f",
                       font=dict(color="#c8d4e8"), margin=dict(l=10, r=10, t=40, b=10)),
            )
            st.plotly_chart(fig_hist, width="stretch")

        # ── Forecast section ──────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🔮 AI Forecast")

        artifact, err = _artifact_safe(city=city, horizon_weeks=int(horizon))
        
        # Check if artifact is compatible with the selected Target
        is_compatible = False
        if artifact:
            art_meta = artifact.load_meta()
            art_target = art_meta.get("target_column")
            expected_target = f"target_{y_col}_t_plus_{horizon}"
            
            if art_target == expected_target:
                is_compatible = True
            elif art_target == y_col:  # Legacy support
                is_compatible = True
            elif mode.startswith("🦟") and (art_target == f"target_total_cases_t_plus_{horizon}" or art_target == "total_cases"):
                is_compatible = True

            # If target matches, ensure features match too
            if is_compatible:
                f_params = FeatureParams(
                    lags=(1, 2, 4, 8, 12, 26, 52), rolling_windows=(4, 8, 12, 26),
                    include_autoregressive=True, target_column=str(y_col), date_column=date_col
                )
                try:
                    feat_last = last_complete_feature_row(df, horizon_weeks=int(horizon), params=f_params)
                    required_cols = art_meta.get("feature_columns", [])
                    if required_cols and any(c for c in required_cols if c not in feat_last.columns):
                        is_compatible = False
                except Exception:
                    is_compatible = False

        if artifact is not None and is_compatible:
            try:
                # ── Robust Prediction initialization ──────────────────────────
                f_params = FeatureParams(
                    lags=(1, 2, 4, 8, 12, 26, 52), rolling_windows=(4, 8, 12, 26),
                    include_autoregressive=True, target_column=str(y_col), date_column=date_col
                )

                pred = predict_next(
                    history_df=df, artifact=artifact,
                    horizon_weeks=int(horizon), params=f_params
                )
                feat_last = last_complete_feature_row(df, horizon_weeks=int(horizon), params=f_params)
                
                # ── Dynamic Features based on Dataset ──
                is_dengue = mode.startswith("🦟") or y_col == "total_cases"
                
                if is_dengue:
                    eri = float(feat_last["eri"].iloc[0]) if "eri" in feat_last.columns else None
                    ers = classify_ers(pred)
                    
                    box_2_title = "ALERT LEVEL"
                    box_2_val = ers
                    box_3_title = "ERI RISK INDEX"
                    box_3_val = f"{eri:.3f}" if eri is not None else "—"
                    box_3_sub = "environmental risk"
                else:
                    # Universal Features: Anomaly Score and Trend Velocity
                    hist_vals = df[y_col].dropna().tail(20)
                    hist_mean = hist_vals.mean()
                    hist_std = hist_vals.std()
                    z_score = (pred - hist_mean) / (hist_std + 1e-9)
                    
                    if abs(z_score) > 2.0:
                        ers = "High"
                    elif abs(z_score) > 1.0:
                        ers = "Elevated"
                    else:
                        ers = "Normal"
                        
                    velocity = (pred - df[y_col].dropna().iloc[-1]) / int(horizon)
                    vel_sign = "+" if velocity > 0 else ""
                    
                    box_2_title = "ANOMALY LEVEL"
                    box_2_val = ers
                    box_3_title = "TREND VELOCITY"
                    box_3_val = f"{vel_sign}{velocity:.2f}"
                    box_3_sub = f"per week (Z: {z_score:.1f})"

                # ── UI Render ──
                ers_icon  = {"Normal": "🟢", "Elevated": "🟠", "High": "🔴"}.get(ers, "⚪")
                ers_color = {"Normal": "#22c55e", "Elevated": "#f59e0b", "High": "#ef4444"}.get(ers, "#94a3b8")
                ers_bg    = {"Normal": "rgba(34,197,94,0.08)", "Elevated": "rgba(245,158,11,0.08)",
                              "High": "rgba(239,68,68,0.1)"}.get(ers, "rgba(129,140,248,0.08)")

                st.markdown(f"""
                <div style="background:{ers_bg};border:1px solid {ers_color}55;
                     border-radius:16px;padding:28px 32px;margin:12px 0;
                     box-shadow:0 4px 30px rgba(0,0,0,0.3);">
                  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:20px;text-align:center;">
                    <div>
                      <div style="color:#64748b;font-size:0.7rem;font-weight:600;letter-spacing:0.08em;margin-bottom:8px;">PREDICTED {str(y_col).upper()}</div>
                      <div style="color:#f1f5f9;font-size:2.8rem;font-weight:800;line-height:1;">{pred:.1f}</div>
                      <div style="color:#475569;font-size:0.72rem;margin-top:4px;">in +{horizon} weeks</div>
                    </div>
                    <div>
                      <div style="color:#64748b;font-size:0.7rem;font-weight:600;letter-spacing:0.08em;margin-bottom:8px;">{box_2_title}</div>
                      <div style="color:{ers_color};font-size:2rem;font-weight:800;line-height:1;">{ers_icon}</div>
                      <div style="color:{ers_color};font-size:1rem;font-weight:700;margin-top:4px;">{box_2_val}</div>
                    </div>
                    <div>
                      <div style="color:#64748b;font-size:0.7rem;font-weight:600;letter-spacing:0.08em;margin-bottom:8px;">{box_3_title}</div>
                      <div style="color:#f1f5f9;font-size:2.8rem;font-weight:800;line-height:1;">{box_3_val}</div>
                      <div style="color:#475569;font-size:0.72rem;margin-top:4px;">{box_3_sub}</div>
                    </div>
                    <div>
                      <div style="color:#64748b;font-size:0.7rem;font-weight:600;letter-spacing:0.08em;margin-bottom:8px;">HORIZON</div>
                      <div style="color:#818cf8;font-size:2.8rem;font-weight:800;line-height:1;">{horizon}</div>
                      <div style="color:#475569;font-size:0.72rem;margin-top:4px;">weeks ahead</div>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Explainability (SHAP) ─────────────────────────────────────
                from retrox.models.inference import explain_prediction
                st.markdown("### 🔍 Why this forecast? (Explainability)")
                expl = explain_prediction(
                    history_df=df, artifact=artifact,
                    horizon_weeks=int(horizon), params=f_params
                )
                
                if "error" not in expl:
                    import matplotlib.pyplot as plt
                    import shap
                    shap_values = expl["shap_values"]
                    
                    c1, c2 = st.columns([1.5, 1])
                    with c1:
                        st.markdown("#### Primary Drivers (Waterfall)")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        plt.style.use('dark_background')
                        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
                        fig.patch.set_facecolor('#07090f')
                        ax.set_facecolor('#07090f')
                        st.pyplot(fig, clear_figure=True)
                    with c2:
                        st.markdown("#### Evidence Summary")
                        st.info(f"""
                        The **SHAP Waterfall** plot shows how each feature pushed the prediction away from the model's base average.
                        - **Target**: {y_col}
                        - **Red bars**: Positive contribution.
                        - **Blue bars**: Negative contribution.
                        """)
                else:
                    st.caption(f"SHAP explanation unavailable: {expl['error']}")

                # ── Actual vs Predicted trend ─────────────────────────────────
                if y_col and y_col in df.columns and len(df) > int(horizon) + 5:
                    st.markdown("---")
                    st.markdown("### 📉 Historical Backtesting")
                    try:
                        from retrox.features.engineering import build_supervised_frame
                        from sklearn.metrics import r2_score
                        
                        sup = build_supervised_frame(df, horizon_weeks=int(horizon), params=f_params)
                        sup_f   = sup.frame
                        feat_cols = art_meta.get("feature_columns", [])
                        
                        if feat_cols and all(c in sup_f.columns for c in feat_cols):
                            X_all  = sup_f[feat_cols]
                            y_all  = sup_f[sup.target_column].astype(float).values
                            pipe   = artifact.load_model()
                            fitted = pipe.predict(X_all)
                            
                            idx = sup_f.index
                            date_vals = sup_f[date_col] if date_col and date_col in sup_f.columns else idx

                            fig_avp = go.Figure()
                            fig_avp.add_trace(go.Scatter(x=date_vals, y=y_all, name="Actual (Historical)",
                                                          line=dict(color="#34d399", width=2)))
                            fig_avp.add_trace(go.Scatter(x=date_vals, y=fitted, name=f"{horizon}w Lead Fit",
                                                          line=dict(color="#f59e0b", width=1.5, dash="dot")))
                            
                            last_date = date_vals.iloc[-1]
                            try:
                                future_date = last_date + pd.Timedelta(weeks=horizon)
                            except:
                                future_date = len(df) + horizon

                            fig_avp.add_trace(go.Scatter(x=[future_date], y=[pred],
                                                          name=f"Current +{horizon}w Forecast",
                                                          mode="markers+text",
                                                          text=[f"{pred:.1f}"], textposition="top center",
                                                          marker=dict(size=14, color="#ef4444",
                                                                       symbol="star", line=dict(width=2, color="white"))))
                            fig_avp.update_layout(
                                title=f"Backtesting Overview — {y_col}",
                                xaxis_title="Timeline", yaxis_title=y_col,
                                template="plotly_dark", plot_bgcolor="#07090f", paper_bgcolor="#07090f",
                                font=dict(color="#c8d4e8"), margin=dict(l=10, r=10, t=40, b=10),
                                legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor="#1a2744", borderwidth=1),
                            )
                            st.plotly_chart(fig_avp, width="stretch")

                            cols = st.columns(3)
                            mae_val = np.mean(np.abs(y_all - fitted))
                            r2_val = r2_score(y_all, fitted)
                            cols[0].metric("Backtest MAE", f"{mae_val:.3f}")
                            cols[1].metric("Backtest R²", f"{r2_val:.3f}")
                            cols[2].caption("Note: Metrics computed on historical lead-shifted sequence.")

                    except Exception as trend_e:
                        st.caption(f"Historical chart unavailable: {trend_e}")

            except Exception as pe:
                st.error(f"Intelligence Pipeline Error: {pe}")
                import traceback
                st.code(traceback.format_exc(), language="python")

        else:
            # ── Missing or Incompatible model ──────────────────────────────
            status_msg = "No trained model found" if not artifact else f"Incompatible model found (Trained for: {art_meta.get('target_column')})"
            st.markdown(f"""
            <div style="background:rgba(245,158,11,0.07);border:1px solid rgba(245,158,11,0.25);
                 border-radius:14px;padding:24px;margin:12px 0;">
              <div style="color:#fcd34d;font-weight:700;font-size:1.1rem;margin-bottom:10px;">
                ⚠️ {status_msg}
              </div>
              <div style="color:#64748b;font-size:0.84rem;">
                Configuration: city=<code>{city}</code>, horizon=<code>{horizon}w</code>, target=<code>{y_col}</code>
                <br><br>
                A specialized model must be trained specifically for this dataset and target column to enable forecasting.
              </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button(f"🚀 Initiate Targeted Model Training for '{y_col}'",
                          key=f"train_btn_{city}_{horizon}_{y_col}"):
                try:
                    with st.spinner(f"Benchmarking best architecture for {y_col} …"):
                        art = train_model(
                            df, city=city, horizon_weeks=int(horizon),
                            params=FeatureParams(lags=(1, 2, 4, 8, 12, 26, 52), rolling_windows=(4, 8, 12, 26), include_autoregressive=True, target_column=str(y_col), date_column=date_col),
                            target_column=y_col,
                            fast_mode=not mode.startswith("🦟")
                        )
                    st.success(f"✅ Intelligence Artifact Generated! Ready to forecast.")
                    st.rerun()
                except Exception as te:
                    st.error(f"Training error: {te}")
                    import traceback
                    st.code(traceback.format_exc())
            else:
                st.success("✅ Model was trained this session. Reload data source above to forecast.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Data Lab
# ─────────────────────────────────────────────────────────────────────────────
with tab_lab:

    # ── Hero pipeline card ────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(79,70,229,0.08),rgba(124,58,237,0.06));
         border:1px solid rgba(129,140,248,0.15);border-radius:18px;padding:24px;margin-bottom:24px;">
      <div style="color:#a5b4fc;font-weight:700;font-size:1rem;margin-bottom:16px;letter-spacing:0.02em;">
        🔬 Automated Analysis Pipeline — Upload any tabular dataset to start
      </div>
      <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:10px;text-align:center;">
        <div style="background:rgba(0,0,0,0.25);border:1px solid rgba(129,140,248,0.12);border-radius:10px;padding:14px;">
          <div style="font-size:1.6rem;margin-bottom:6px;">🔍</div>
          <div style="color:#818cf8;font-size:0.72rem;font-weight:700;letter-spacing:0.04em;">STAGE 1</div>
          <div style="color:#64748b;font-size:0.68rem;margin-top:2px;">Smart EDA</div>
        </div>
        <div style="background:rgba(0,0,0,0.25);border:1px solid rgba(129,140,248,0.12);border-radius:10px;padding:14px;">
          <div style="font-size:1.6rem;margin-bottom:6px;">⚙️</div>
          <div style="color:#818cf8;font-size:0.72rem;font-weight:700;letter-spacing:0.04em;">STAGE 2</div>
          <div style="color:#64748b;font-size:0.68rem;margin-top:2px;">Preprocessing</div>
        </div>
        <div style="background:rgba(0,0,0,0.25);border:1px solid rgba(129,140,248,0.12);border-radius:10px;padding:14px;">
          <div style="font-size:1.6rem;margin-bottom:6px;">🌌</div>
          <div style="color:#818cf8;font-size:0.72rem;font-weight:700;letter-spacing:0.04em;">STAGE 3</div>
          <div style="color:#64748b;font-size:0.68rem;margin-top:2px;">t-SNE / PCA</div>
        </div>
        <div style="background:rgba(0,0,0,0.25);border:1px solid rgba(129,140,248,0.12);border-radius:10px;padding:14px;">
          <div style="font-size:1.6rem;margin-bottom:6px;">🤖</div>
          <div style="color:#818cf8;font-size:0.72rem;font-weight:700;letter-spacing:0.04em;">STAGE 4</div>
          <div style="color:#64748b;font-size:0.68rem;margin-top:2px;">AutoML · 5 Models</div>
        </div>
        <div style="background:rgba(0,0,0,0.25);border:1px solid rgba(129,140,248,0.12);border-radius:10px;padding:14px;">
          <div style="font-size:1.6rem;margin-bottom:6px;">💡</div>
          <div style="color:#818cf8;font-size:0.72rem;font-weight:700;letter-spacing:0.04em;">STAGE 5</div>
          <div style="color:#64748b;font-size:0.68rem;margin-top:2px;">AI Insights</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Upload row ────────────────────────────────────────────────────────────
    upcol, btncol = st.columns([4, 1])
    with upcol:
        uploaded = st.file_uploader(
            "📂 Upload dataset (CSV, Excel, JSON, Parquet — any tabular data)",
            type=["csv", "xlsx", "xls", "json", "parquet"],
            key="lab_upload",
        )
    with btncol:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if st.button("🦟 Load DengAI", key="lab_dengai_btn"):
            _ensure_demo()
            try:
                st.session_state["lab_df"]     = load_dengai_train(city="sj")
                st.session_state["lab_source"] = "DengAI training data (San Juan)"
            except Exception as e:
                st.error(str(e))

    # Resolve dataframe
    raw_df: pd.DataFrame | None = st.session_state.get("lab_df")
    if uploaded is not None:
        try:
            raw_df = _read_file(uploaded)
            st.session_state["lab_df"]     = raw_df
            st.session_state["lab_source"] = uploaded.name
        except Exception as e:
            st.error(f"Could not read file: {e}")
            raw_df = None

    if raw_df is None:
        st.markdown("""
        <div style="text-align:center;padding:80px 40px;border:2px dashed #1a2744;border-radius:20px;margin-top:24px;">
          <div style="font-size:4rem;margin-bottom:16px;opacity:0.5;">📂</div>
          <div style="font-size:1.15rem;color:#334155;font-weight:600;">Drop any dataset to begin the AI pipeline</div>
          <div style="font-size:0.85rem;color:#1f2d45;margin-top:8px;">
            CSV · Excel (.xlsx/.xls) · JSON · Parquet
          </div>
          <div style="font-size:0.8rem;color:#1a2338;margin-top:6px;">
            Census · Medical · Financial · Epidemiological · Any tabular data
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        src = st.session_state.get("lab_source", "dataset")
        st.markdown(f"""
        <div style="display:inline-flex;align-items:center;gap:12px;background:rgba(16,185,129,0.08);
             border:1px solid rgba(52,211,153,0.2);border-radius:10px;padding:10px 18px;margin-bottom:16px;">
          <span style="color:#34d399;font-size:1.1rem;">✅</span>
          <strong style="color:#34d399;">{src}</strong>
          <span style="color:#334155;font-size:0.82rem;">{raw_df.shape[0]:,} rows × {raw_df.shape[1]} columns</span>
        </div>
        """, unsafe_allow_html=True)

        # ── Stage 1 ───────────────────────────────────────────────────────────
        render_smart_eda(raw_df)

        # ── Stage 2 ───────────────────────────────────────────────────────────
        processed_df = render_preprocessing_controls(raw_df)

        # ── Download processed ────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📥 Export Processed Data")
        dc1, dc2 = st.columns(2)
        with dc1:
            st.download_button("⬇️ Download as CSV",
                                processed_df.to_csv(index=False).encode(),
                                "processed_data.csv", "text/csv")
        with dc2:
            buf = BytesIO()
            try:
                processed_df.to_parquet(buf, index=False)
                st.download_button("⬇️ Download as Parquet", buf.getvalue(),
                                    "processed_data.parquet", "application/octet-stream")
            except Exception:
                st.caption("Parquet not available; download CSV instead.")

        # ── Stage 3 ───────────────────────────────────────────────────────────
        num_check = processed_df.select_dtypes(include=[np.number]).columns.tolist()
        st.markdown("---")
        if len(num_check) >= 2 and len(processed_df) >= 20:
            render_tsne(processed_df, raw_df)
        else:
            st.warning("⚠️ t-SNE requires ≥2 numeric columns and ≥20 rows after preprocessing.")

        # ── Stage 4 ───────────────────────────────────────────────────────────
        aml = render_automl(processed_df, raw_df)

        # ── Stage 5 ───────────────────────────────────────────────────────────
        if aml:
            render_suggestions(raw_df, processed_df, aml)
        else:
            st.info("💡 Complete Stage 4 (select a target column and run AutoML) to see AI suggestions.")
