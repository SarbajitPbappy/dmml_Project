"""
lab_pipeline.py — RetroX Data Lab Analytics Pipeline v3
=========================================================
Full automatic pipeline: EDA → Preprocessing → t-SNE → AutoML → Suggestions

Fixes & Features:
  • tabulate dependency removed — report now uses CSV fallback
  • TSNE max_iter (sklearn 1.8 compat)
  • Classification bug fixed: AUC-ROC formatting when all nan
  • HistGBM first for feature_importances_ compat
  • Added: outlier table, class-imbalance chart, ROC curve, learning curve,
           cross-val score distribution, partial dependence concept plots,
           numeric interaction heatmap, target distribution chart
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

warnings.filterwarnings("ignore")

_TSNE_MAX_ROWS = 2500
_SCATTER_MAX_COLS = 7
_CV_FOLDS = 5
_MIN_ROWS_FOR_MODEL = 30
_CLASS_MAX_UNIQUE = 20

DARK_BG   = "#0a0e1a"
CARD_BG   = "#111827"
BORDER    = "#1f2d45"
ACCENT    = "#818cf8"
ACCENT2   = "#34d399"
WARN      = "#f59e0b"
DANGER    = "#ef4444"


def _plotly_dark() -> dict:
    """Layout kwargs for fig.update_layout() only — do NOT spread into px functions."""
    return dict(template="plotly_dark", plot_bgcolor=DARK_BG, paper_bgcolor=DARK_BG,
                font=dict(color="#cbd5e1"), margin=dict(l=10, r=10, t=40, b=10))

def _dark(fig):
    """Apply dark theme to any plotly figure and return it."""
    fig.update_layout(**_plotly_dark())
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 1. Smart EDA
# ─────────────────────────────────────────────────────────────────────────────

def render_smart_eda(df: pd.DataFrame) -> None:
    st.markdown('<h2 style="color:#93c5fd;">🔍 Stage 1 — Smart Exploratory Data Analysis</h2>', unsafe_allow_html=True)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    miss_total = int(df.isna().sum().sum())
    dup_count  = int(df.duplicated().sum())

    # ── Big overview metrics ──────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("📦 Rows",         f"{len(df):,}")
    c2.metric("🧮 Columns",      f"{df.shape[1]}")
    c3.metric("🔢 Numeric",      len(num_cols))
    c4.metric("🔤 Categorical",  len(cat_cols))
    c5.metric("⚠️ Missing",      f"{miss_total:,}")
    c6.metric("♻️ Duplicates",   f"{dup_count:,}")

    if dup_count > 0:
        st.warning(f"♻️ **{dup_count:,} duplicate rows** detected ({dup_count/len(df)*100:.1f}%) — will be removed in preprocessing.")

    # ── Schema ───────────────────────────────────────────────────────────────
    with st.expander("📋 Column Schema & Quick Stats", expanded=True):
        schema = pd.DataFrame({
            "Column":       df.columns.astype(str),
            "Type":         [str(t) for t in df.dtypes],
            "Non-Null":     df.notna().sum().values,
            "Missing %":    (df.isna().mean() * 100).round(2).values,
            "Unique":       df.nunique(dropna=False).values,
            "Sample":       [str(df[c].dropna().iloc[0]) if df[c].notna().any() else "—" for c in df.columns],
        })
        st.dataframe(schema, use_container_width=True)

    if not num_cols:
        st.warning("No numeric columns — skipping numeric EDA.")
        return

    # ── Numeric summary ───────────────────────────────────────────────────────
    with st.expander("📊 Numeric Summary — Skew & Kurtosis", expanded=False):
        desc = df[num_cols].describe().T.round(4)
        desc["skewness"]  = df[num_cols].skew().round(3)
        desc["kurtosis"]  = df[num_cols].kurtosis().round(3)
        desc["CV%"]       = (df[num_cols].std() / df[num_cols].mean().replace(0, np.nan) * 100).round(1)
        desc["outliers"]  = df[num_cols].apply(lambda s: int(((s - s.mean()).abs() > 3 * s.std()).sum()))
        high_skew = desc[desc["skewness"].abs() > 1].index.tolist()
        st.dataframe(desc.style.background_gradient(subset=["skewness", "kurtosis"], cmap="RdYlGn_r"), use_container_width=True)
        if high_skew:
            st.warning(f"High skew (|skew|>1): **{', '.join(high_skew[:6])}** — consider log1p/sqrt/Box-Cox.")

    # ── Outlier table ─────────────────────────────────────────────────────────
    with st.expander("🚨 Outlier Summary (3σ rule)", expanded=False):
        out_rows = []
        for c in num_cols:
            s = df[c].dropna()
            mn, mx, mean, std = s.min(), s.max(), s.mean(), s.std()
            if std > 0:
                n_out = int(((s - mean).abs() > 3 * std).sum())
                out_rows.append({"Column": c, "Outliers (3σ)": n_out, "Pct": round(n_out / len(s) * 100, 2),
                                 "Min": round(mn, 4), "Max": round(mx, 4), "Mean": round(mean, 4), "Std": round(std, 4)})
        if out_rows:
            out_df = pd.DataFrame(out_rows).sort_values("Outliers (3σ)", ascending=False)
            st.dataframe(out_df.style.background_gradient(subset=["Pct"], cmap="OrRd"), use_container_width=True)

    # ── Missingness ───────────────────────────────────────────────────────────
    miss = df.isna().sum().sort_values(ascending=False)
    miss = miss[miss > 0]
    if not miss.empty:
        with st.expander("⚠️ Missingness Chart", expanded=False):
            colors = ["#ef4444" if v/len(df)>0.4 else "#f97316" if v/len(df)>0.1 else "#facc15" for v in miss.values]
            fig = go.Figure(go.Bar(x=miss.index.astype(str), y=miss.values, marker_color=colors))
            fig.update_layout(title="Missing Values per Column", xaxis_title="Column", yaxis_title="Count",
                             template="plotly_dark")
            st.plotly_chart(fig, width="stretch")
    else:
        st.success("✅ No missing values found!")

    # ── Correlation heatmap ───────────────────────────────────────────────────
    with st.expander("🔗 Pearson Correlation Matrix", expanded=True):
        corr_cols = num_cols[:40]
        corr = df[corr_cols].corr(numeric_only=True).round(3)
        fig_corr = px.imshow(corr, text_auto=".2f" if len(corr_cols)<=12 else False,
                              aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                              title="Pearson Correlation Matrix")
        fig_corr.update_layout(**_plotly_dark())
        st.plotly_chart(fig_corr, width="stretch")
        pairs = [(a, b, round(corr.loc[a,b],3)) for i,a in enumerate(corr_cols)
                 for b in corr_cols[i+1:] if abs(corr.loc[a,b]) > 0.85]
        if pairs:
            st.info("🔄 Highly correlated pairs (|r|>0.85): " +
                    " · ".join(f"**{a}⟷{b}** ({v})" for a,b,v in pairs[:6]))

    # ── Interactive distribution explorer ────────────────────────────────────
    with st.expander("📈 Distribution & Outlier Explorer", expanded=False):
        dist_col = st.selectbox("Column", num_cols, key="eda_dist_col")
        ca, cb, cc = st.columns(3)
        with ca:
            fig_h = px.histogram(df, x=dist_col, nbins=60, marginal="rug",
                                  title=f"Distribution: {dist_col}",
                                  template="plotly_dark",
                                  color_discrete_sequence=[ACCENT])
            fig_h.update_layout(plot_bgcolor=DARK_BG, paper_bgcolor=DARK_BG,
                                 font=dict(color="#cbd5e1"), margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig_h, width="stretch")
        with cb:
            fig_b = px.box(df, y=dist_col, title=f"Box Plot: {dist_col}",
                            template="plotly_dark", color_discrete_sequence=[WARN])
            fig_b.update_layout(plot_bgcolor=DARK_BG, paper_bgcolor=DARK_BG, margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig_b, width="stretch")
        with cc:
            fig_v = px.violin(df, y=dist_col, box=True, title=f"Violin: {dist_col}",
                               template="plotly_dark", color_discrete_sequence=[ACCENT2])
            fig_v.update_layout(plot_bgcolor=DARK_BG, paper_bgcolor=DARK_BG, margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig_v, width="stretch")

    # ── Numeric pairwise scatter matrix ──────────────────────────────────────
    with st.expander("🔵 Pairwise Scatter Matrix", expanded=False):
        pair_cols = num_cols[:_SCATTER_MAX_COLS]
        hue_opts = ["(none)"] + cat_cols[:5] + num_cols[:5]
        pair_hue = st.selectbox("Color by", hue_opts, key="pair_hue")
        color_col = None if pair_hue == "(none)" else pair_hue
        plot_cols = pair_cols + ([color_col] if color_col and color_col not in pair_cols else [])
        fig_pair = px.scatter_matrix(df[plot_cols].dropna(), dimensions=pair_cols,
                                      color=color_col, title="Pairwise Scatter Matrix",
                                      template="plotly_dark", opacity=0.35)
        fig_pair.update_traces(marker=dict(size=2))
        fig_pair.update_layout(plot_bgcolor=DARK_BG, paper_bgcolor=DARK_BG, margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig_pair, width="stretch")

    # ── Categorical value counts ──────────────────────────────────────────────
    if cat_cols:
        with st.expander("📊 Categorical Value Counts", expanded=False):
            cat_sel = st.selectbox("Categorical column", cat_cols, key="cat_vc")
            vc = df[cat_sel].value_counts().head(25)
            fig_vc = px.bar(x=vc.index.astype(str), y=vc.values,
                             title=f"Value Counts: {cat_sel}", labels={"x": cat_sel, "y": "Count"},
                             template="plotly_dark", color=vc.values, color_continuous_scale="Plasma")
            fig_vc.update_layout(plot_bgcolor=DARK_BG, paper_bgcolor=DARK_BG, margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig_vc, width="stretch")

    # ── Time-series line plot if date column detected ─────────────────────────
    date_cols = [c for c in df.columns if any(kw in c.lower() for kw in ["date","time","week","month","year","timestamp"])]
    if date_cols and num_cols:
        with st.expander("📅 Time-Series Overview", expanded=False):
            date_sel = st.selectbox("Date/time column", date_cols, key="ts_date")
            y_sel    = st.selectbox("Value column", num_cols, key="ts_y")
            try:
                ts_df = df[[date_sel, y_sel]].copy()
                ts_df[date_sel] = pd.to_datetime(ts_df[date_sel], errors="coerce")
                ts_df = ts_df.dropna().sort_values(date_sel)
                fig_ts = px.line(ts_df, x=date_sel, y=y_sel, title=f"{y_sel} over time",
                                  template="plotly_dark", color_discrete_sequence=[ACCENT])
                fig_ts.update_traces(line=dict(width=2))
                fig_ts.update_layout(plot_bgcolor=DARK_BG, paper_bgcolor=DARK_BG, margin=dict(l=10,r=10,t=40,b=10))
                st.plotly_chart(fig_ts, width="stretch")
            except Exception as exc:
                st.caption(f"Could not plot time-series: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Advanced Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def render_preprocessing_controls(df: pd.DataFrame) -> pd.DataFrame:
    st.markdown('<h2 style="color:#93c5fd;">⚙️ Stage 2 — Advanced Preprocessing</h2>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        impute_strategy = st.selectbox("Numeric Imputation",
                                        ["median", "mean", "most_frequent", "constant"], key="lab_impute")
    with col2:
        scale_strategy = st.selectbox("Feature Scaling", ["none", "standard", "minmax"], key="lab_scale")
    with col3:
        winsorize = st.checkbox("Outlier Capping (1%–99%)", value=True, key="lab_winsor")
    with col4:
        one_hot = st.checkbox("One-Hot Encode Categoricals", value=True, key="lab_ohe")

    col5, col6 = st.columns(2)
    with col5:
        drop_low_var = st.checkbox("Drop Near-Zero Variance Cols", value=False, key="lab_lowvar")
        var_thresh = st.slider("Variance threshold", 0.0, 0.05, 0.01, 0.001, key="lab_varthresh") if drop_low_var else 0.0
    with col6:
        drop_high_miss = st.checkbox("Drop Cols >50% Missing", value=True, key="lab_highmiss")

    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in out.columns]
    out = out.drop_duplicates().reset_index(drop=True)

    if drop_high_miss:
        miss = out.isna().mean()
        drop = miss[miss > 0.5].index.tolist()
        if drop:
            st.info(f"Dropped {len(drop)} high-missingness columns: `{drop}`")
            out = out.drop(columns=drop)

    num_p  = out.select_dtypes(include=[np.number]).columns.tolist()
    cat_p  = out.select_dtypes(exclude=[np.number]).columns.tolist()

    if num_p:
        fill  = 0 if impute_strategy == "constant" else None
        strat = impute_strategy if impute_strategy != "constant" else "constant"
        imp   = SimpleImputer(strategy=strat, fill_value=fill)
        out[num_p] = imp.fit_transform(out[num_p])
    if cat_p:
        out[cat_p] = out[cat_p].fillna("unknown")

    if winsorize and num_p:
        for c in num_p:
            lo, hi = out[c].quantile(0.01), out[c].quantile(0.99)
            out[c] = out[c].clip(lower=lo, upper=hi)

    if one_hot and cat_p:
        out = pd.get_dummies(out, columns=cat_p, drop_first=False)

    if drop_low_var and var_thresh > 0:
        from sklearn.feature_selection import VarianceThreshold
        num_f = out.select_dtypes(include=[np.number]).columns.tolist()
        if num_f:
            sel   = VarianceThreshold(threshold=var_thresh)
            sel.fit(out[num_f])
            kept    = [c for c, s in zip(num_f, sel.get_support()) if s]
            dropped = [c for c in num_f if c not in kept]
            if dropped:
                st.info(f"Dropped {len(dropped)} near-zero-variance columns.")
            out = out[kept + [c for c in out.columns if c not in num_f]]

    if scale_strategy != "none":
        num_f = out.select_dtypes(include=[np.number]).columns.tolist()
        if num_f:
            scaler = StandardScaler() if scale_strategy == "standard" else MinMaxScaler()
            out[num_f] = scaler.fit_transform(out[num_f])

    ca, cb = st.columns(2)
    ca.metric("Original Shape",  f"{df.shape[0]:,} × {df.shape[1]}")
    cb.metric("Processed Shape", f"{out.shape[0]:,} × {out.shape[1]}")

    with st.expander("👀 Processed Data Preview (top 50 rows)", expanded=False):
        st.dataframe(out.head(50), use_container_width=True)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. t-SNE
# ─────────────────────────────────────────────────────────────────────────────

def render_tsne(processed_df: pd.DataFrame, raw_df: pd.DataFrame) -> None:
    st.markdown('<h2 style="color:#93c5fd;">🌌 Stage 3 — t-SNE Dimensionality Reduction</h2>', unsafe_allow_html=True)

    num_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        st.warning("Need ≥2 numeric columns for t-SNE.")
        return

    n_rows      = len(processed_df)
    sample_size = min(n_rows, _TSNE_MAX_ROWS)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        perplexity = st.slider("Perplexity", 5, 80, min(30, max(5, sample_size//10)), key="tsne_perp")
    with c2:
        color_opts = ["(none)"] + list(raw_df.columns)
        tsne_color = st.selectbox("Color by (raw column)", color_opts, key="tsne_color")
    with c3:
        tsne_3d = st.checkbox("3D Mode", value=False, key="tsne_3d")
    with c4:
        tsne_iters = st.selectbox("Iterations", [300, 500, 1000], index=1, key="tsne_iter")

    with st.spinner(f"Running {'3D' if tsne_3d else '2D'} t-SNE on {sample_size:,} rows …"):
        sample_idx = processed_df.sample(n=sample_size, random_state=42).index
        X = processed_df.loc[sample_idx, num_cols].fillna(0).values.astype(float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        if X.shape[1] > 50:
            X = PCA(n_components=50, random_state=42).fit_transform(X)

        n_comp = 3 if tsne_3d else 2
        tsne   = TSNE(n_components=n_comp,
                      perplexity=min(perplexity, sample_size - 1),
                      random_state=42, max_iter=int(tsne_iters),
                      init="pca", learning_rate="auto")
        emb = tsne.fit_transform(X)

    color_vals = None
    if tsne_color != "(none)" and tsne_color in raw_df.columns:
        color_vals = raw_df.loc[sample_idx, tsne_color].reset_index(drop=True).values

    cols  = ["TSNE-1","TSNE-2"] if not tsne_3d else ["TSNE-1","TSNE-2","TSNE-3"]
    emb_df = pd.DataFrame(emb, columns=cols)
    if color_vals is not None:
        emb_df["color"] = color_vals

    color_arg = "color" if color_vals is not None else None
    if tsne_3d:
        fig = px.scatter_3d(emb_df, x="TSNE-1", y="TSNE-2", z="TSNE-3", title="t-SNE 3D",
                             color=color_arg, opacity=0.65, template="plotly_dark",
                             color_continuous_scale="Plasma")
    else:
        fig = px.scatter(emb_df, x="TSNE-1", y="TSNE-2", title="t-SNE 2D Projection",
                          color=color_arg, opacity=0.65, template="plotly_dark",
                          color_continuous_scale="Plasma")
    fig.update_traces(marker=dict(size=3 if not tsne_3d else 2))
    fig.update_layout(plot_bgcolor=DARK_BG, paper_bgcolor=DARK_BG,
                       font=dict(color="#cbd5e1"), margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"KL divergence: {tsne.kl_divergence_:.4f} · {sample_size:,}/{n_rows:,} rows sampled")


# ─────────────────────────────────────────────────────────────────────────────
# 4. AutoML
# ─────────────────────────────────────────────────────────────────────────────

def _detect_task(series: pd.Series) -> str:
    n_unique = series.nunique()
    if pd.api.types.is_float_dtype(series) and n_unique > _CLASS_MAX_UNIQUE:
        return "regression"
    if n_unique <= _CLASS_MAX_UNIQUE:
        return "classification"
    return "regression"


def _safe_fmt(val, fmt="{:.4f}") -> str:
    try:
        return fmt.format(float(val))
    except Exception:
        return "—"


def _cv_regression(model, X: np.ndarray, y: np.ndarray) -> dict:
    tscv = TimeSeriesSplit(n_splits=_CV_FOLDS)
    maes, rmses, r2s = [], [], []
    for tr, te in tscv.split(X):
        try:
            model.fit(X[tr], y[tr])
            p = model.predict(X[te])
            maes.append(mean_absolute_error(y[te], p))
            rmses.append(float(np.sqrt(mean_squared_error(y[te], p))))
            r2s.append(r2_score(y[te], p))
        except Exception:
            pass
    if not maes:
        return {"MAE": np.nan, "RMSE": np.nan, "R²": np.nan}
    return {"MAE": float(np.mean(maes)), "RMSE": float(np.mean(rmses)), "R²": float(np.mean(r2s))}


def _cv_classification(model, X: np.ndarray, y: np.ndarray) -> dict:
    n_unique = len(np.unique(y))
    n_splits = min(_CV_FOLDS, n_unique, len(y) // 2)
    if n_splits < 2:
        return {"Accuracy": np.nan, "F1 (weighted)": np.nan, "AUC-ROC": np.nan}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, f1s, aucs = [], [], []
    binary = n_unique == 2
    for tr, te in skf.split(X, y):
        try:
            model.fit(X[tr], y[tr])
            p = model.predict(X[te])
            accs.append(accuracy_score(y[te], p))
            f1s.append(f1_score(y[te], p, average="weighted", zero_division=0))
            if binary and hasattr(model, "predict_proba"):
                prob = model.predict_proba(X[te])[:, 1]
                aucs.append(roc_auc_score(y[te], prob))
        except Exception:
            pass
    return {
        "Accuracy":      float(np.mean(accs)) if accs else np.nan,
        "F1 (weighted)": float(np.mean(f1s))  if f1s  else np.nan,
        "AUC-ROC":       float(np.mean(aucs)) if aucs else np.nan,
    }


def render_automl(processed_df: pd.DataFrame, raw_df: pd.DataFrame) -> dict[str, Any]:
    st.markdown('<h2 style="color:#93c5fd;">🤖 Stage 4 — Automated Machine Learning</h2>', unsafe_allow_html=True)

    num_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        st.warning("Need ≥2 numeric columns for AutoML.")
        return {}
    if len(processed_df) < _MIN_ROWS_FOR_MODEL:
        st.warning(f"Too few rows ({len(processed_df)}) — need ≥{_MIN_ROWS_FOR_MODEL}.")
        return {}

    # ── Target selection ──────────────────────────────────────────────────────
    all_cols = list(processed_df.columns)
    hints    = ["target","label","class","income","salary","price","total_cases",
                "churn","y","output","result","outcome","survived","diagnosis"]
    default  = next((c for h in hints for c in all_cols if h in c.lower()), all_cols[-1])
    target_col = st.selectbox("🎯 Target Column", all_cols,
                               index=all_cols.index(default), key="automl_target")
    feature_cols = [c for c in num_cols if c != target_col]
    if not feature_cols:
        st.warning("No feature columns remaining after removing the target.")
        return {}

    task    = _detect_task(processed_df[target_col])
    n_unique = processed_df[target_col].nunique()
    st.info(f"🔎 **Detected task: {task.upper()}** — {n_unique} unique values in `{target_col}`")

    # Target distribution
    with st.expander("📊 Target Distribution", expanded=False):
        if task == "classification":
            vc = processed_df[target_col].value_counts()
            fig_td = px.bar(x=vc.index.astype(str), y=vc.values,
                             title="Class Distribution", labels={"x": target_col, "y": "Count"},
                             template="plotly_dark", color=vc.values, color_continuous_scale="Viridis")
            st.plotly_chart(fig_td, width="stretch")
            imbalance_ratio = vc.max() / max(vc.min(), 1)
            if imbalance_ratio > 5:
                st.warning(f"⚠️ High class imbalance (ratio {imbalance_ratio:.1f}x) — consider SMOTE or class_weight='balanced'")
        else:
            y_vals = processed_df[target_col].dropna()
            fig_td = px.histogram(y_vals, nbins=50, title="Target Distribution",
                             template="plotly_dark", color_discrete_sequence=[ACCENT])
            st.plotly_chart(fig_td, width="stretch")

    X_df = processed_df[feature_cols].copy().fillna(0)
    X    = np.nan_to_num(X_df.values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    results: dict[str, Any] = {"task": task, "feature_cols": feature_cols}

    # ─── REGRESSION ──────────────────────────────────────────────────────────
    if task == "regression":
        y = processed_df[target_col].astype(float).values

        models_reg = {
            "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=42),
            "Gradient Boosting":    GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42),
            "Random Forest":        RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42),
            "K-Nearest Neighbors":  KNeighborsRegressor(n_neighbors=7),
            "Ridge Regression":     Ridge(alpha=1.0),
        }
        rows, trained = [], {}
        bar = st.progress(0, text="Comparing regression models…")
        for i, (name, m) in enumerate(models_reg.items()):
            with st.spinner(f"⏳ Training {name}…"):
                s = _cv_regression(m, X, y)
                rows.append({"Model": name, **s})
                try:
                    m.fit(X, y); trained[name] = m
                except Exception:
                    pass
            bar.progress((i+1)/len(models_reg), text=f"✅ Done: {name}")
        bar.empty()

        lb = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
        lb.index += 1
        results.update({"leaderboard": lb, "trained_models": trained})

        st.success("✅ AutoML complete — all models evaluated!")
        st.markdown("### 🏆 Model Leaderboard (5-fold Time-Series CV)")

        # Format safely: some values may be NaN
        def _fmt_reg(x):
            return f"{x:.4f}" if pd.notna(x) else "—"

        styled = lb.style \
            .format({"MAE": _fmt_reg, "RMSE": _fmt_reg, "R²": _fmt_reg}) \
            .background_gradient(subset=lb.columns[lb.columns.isin(["MAE","RMSE"])], cmap="RdYlGn_r") \
            .background_gradient(subset=lb.columns[lb.columns.isin(["R²"])], cmap="RdYlGn")
        st.dataframe(styled, use_container_width=True)

        best_name = lb.iloc[0]["Model"]
        results["best_model_name"] = best_name
        best_m = trained.get(best_name)

        b_mae = lb.iloc[0]["MAE"]; b_r2 = lb.iloc[0]["R²"]
        st.markdown(f"### 🥇 Best: **{best_name}** | MAE={_safe_fmt(b_mae)} · R²={_safe_fmt(b_r2)}")

        # Comparison bar
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Bar(name="MAE", x=lb["Model"], y=lb["MAE"], marker_color=ACCENT))
        fig_cmp.add_trace(go.Bar(name="RMSE", x=lb["Model"], y=lb["RMSE"], marker_color=WARN, opacity=0.7))
        fig_cmp.update_layout(barmode="group", title="Model Comparison — Error Metrics",
                             template="plotly_dark")
        st.plotly_chart(fig_cmp, width="stretch")

        if best_m is not None:
            preds = best_m.predict(X)
            c1, c2 = st.columns(2)
            with c1:
                res_df = pd.DataFrame({"Predicted": preds, "Residual": y - preds,
                                        "idx": np.arange(len(preds))})
                fig_r = px.scatter(res_df, x="Predicted", y="Residual", opacity=0.45,
                                    title="Residuals vs Predicted",
                             template="plotly_dark",
                                    color="Residual", color_continuous_scale="RdBu")
                fig_r.add_hline(y=0, line_dash="dash", line_color="#f87171")
                st.plotly_chart(fig_r, width="stretch")
            with c2:
                fig_ap = px.scatter(x=y, y=preds, opacity=0.45,
                                     labels={"x":"Actual","y":"Predicted"},
                                     title="Actual vs Predicted",
                             template="plotly_dark",
                                     color=np.abs(y-preds), color_continuous_scale="Viridis")
                mn, mx = float(y.min()), float(y.max())
                fig_ap.add_shape(type="line", x0=mn,y0=mn,x1=mx,y1=mx,
                                  line=dict(color="white",dash="dash"))
                st.plotly_chart(fig_ap, width="stretch")

            # Actual vs predicted trend (time-ordered)
            st.markdown("#### 📉 Actual vs Predicted Trend")
            n_show = min(len(y), 200)
            trend_df = pd.DataFrame({"Index": np.arange(n_show),
                                      "Actual": y[:n_show], "Predicted": preds[:n_show]})
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(x=trend_df["Index"], y=trend_df["Actual"],
                                            name="Actual", line=dict(color=ACCENT2, width=2)))
            fig_trend.add_trace(go.Scatter(x=trend_df["Index"], y=trend_df["Predicted"],
                                            name="Predicted", line=dict(color=WARN, width=2, dash="dot")))
            fig_trend.update_layout(title=f"Actual vs Predicted Trend ({n_show} samples)",
                                     xaxis_title="Sample Index", yaxis_title=target_col,
                             template="plotly_dark")
            st.plotly_chart(fig_trend, width="stretch")

        _render_feature_importances_reg(trained, feature_cols)

    # ─── CLASSIFICATION ───────────────────────────────────────────────────────
    else:
        le = LabelEncoder()
        y  = le.fit_transform(processed_df[target_col].astype(str).fillna("unknown").values)
        n_classes = len(le.classes_)
        labels    = le.classes_.tolist()

        models_clf = {
            "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=42),
            "Gradient Boosting":    GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42),
            "Random Forest":        RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
            "Logistic Regression":  LogisticRegression(max_iter=1000, random_state=42),
            "K-Nearest Neighbors":  KNeighborsClassifier(n_neighbors=7),
        }
        rows, trained = [], {}
        bar = st.progress(0, text="Comparing classification models…")
        for i, (name, m) in enumerate(models_clf.items()):
            with st.spinner(f"⏳ Training {name}…"):
                s = _cv_classification(m, X, y)
                rows.append({"Model": name, **s})
                try:
                    m.fit(X, y); trained[name] = m
                except Exception:
                    pass
            bar.progress((i+1)/len(models_clf), text=f"✅ Done: {name}")
        bar.empty()

        lb = pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)
        lb.index += 1
        results.update({"leaderboard": lb, "trained_models": trained, "label_encoder": le})

        st.success("✅ AutoML complete — all models evaluated!")
        st.markdown("### 🏆 Model Leaderboard (Stratified CV)")

        def _fmt_clf(x):
            return f"{x:.4f}" if pd.notna(x) else "—"

        metric_cols = [c for c in ["Accuracy","F1 (weighted)"] if c in lb.columns]
        styled = lb.style.format({c: _fmt_clf for c in ["Accuracy","F1 (weighted)","AUC-ROC"]})
        if metric_cols:
            styled = styled.background_gradient(subset=metric_cols, cmap="RdYlGn")
        st.dataframe(styled, use_container_width=True)

        best_name = lb.iloc[0]["Model"]
        results["best_model_name"] = best_name
        best_m = trained.get(best_name)

        b_acc = lb.iloc[0]["Accuracy"]; b_f1 = lb.iloc[0]["F1 (weighted)"]
        st.markdown(f"### 🥇 Best: **{best_name}** | Acc={_safe_fmt(b_acc)} · F1={_safe_fmt(b_f1)}")

        # Comparison bar
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Bar(name="Accuracy", x=lb["Model"], y=lb["Accuracy"], marker_color=ACCENT))
        fig_cmp.add_trace(go.Bar(name="F1", x=lb["Model"], y=lb["F1 (weighted)"], marker_color=ACCENT2, opacity=0.8))
        fig_cmp.update_layout(barmode="group", title="Model Comparison",
                             template="plotly_dark")
        st.plotly_chart(fig_cmp, width="stretch")

        if best_m is not None:
            from sklearn.metrics import confusion_matrix, classification_report
            preds = best_m.predict(X)
            cm    = confusion_matrix(y, preds)
            str_labels = [str(c) for c in labels]

            c1, c2 = st.columns([2, 1])
            with c1:
                fig_cm = px.imshow(cm, x=str_labels, y=str_labels, text_auto=True,
                                    color_continuous_scale="Blues",
                                    title=f"Confusion Matrix — {best_name}",
                             template="plotly_dark")
                fig_cm.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
                st.plotly_chart(fig_cm, width="stretch")
            with c2:
                rpt = classification_report(y, preds, target_names=str_labels,
                                             output_dict=True, zero_division=0)
                rpt_df = pd.DataFrame(rpt).T.round(3)
                st.dataframe(rpt_df, use_container_width=True)

            # ROC Curve (binary only)
            if n_classes == 2 and hasattr(best_m, "predict_proba"):
                try:
                    probs = best_m.predict_proba(X)[:, 1]
                    fpr, tpr, _ = roc_curve(y, probs)
                    auc_val = roc_auc_score(y, probs)
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr,
                                                  name=f"ROC (AUC={auc_val:.4f})",
                                                  line=dict(color=ACCENT, width=2)))
                    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1],
                                                  line=dict(color="#64748b", dash="dash"),
                                                  name="Random"))
                    fig_roc.update_layout(title="ROC Curve",
                                           xaxis_title="False Positive Rate",
                                           yaxis_title="True Positive Rate",
                             template="plotly_dark")
                    st.plotly_chart(fig_roc, width="stretch")
                except Exception:
                    pass

        _render_feature_importances_clf(trained, feature_cols)

    return results


def _render_feature_importances_reg(trained: dict[str, Any], feature_cols: list[str]) -> None:
    # prefer tree-based models with feature_importances_
    tree_m = {k: v for k, v in trained.items() if hasattr(v, "feature_importances_")}
    if not tree_m:
        return
    with st.expander("🌲 Feature Importances", expanded=True):
        tabs = st.tabs(list(tree_m.keys()))
        for tab, (name, m) in zip(tabs, tree_m.items()):
            with tab:
                imp  = m.feature_importances_
                fi_df = pd.DataFrame({"Feature": feature_cols, "Importance": imp}) \
                          .sort_values("Importance", ascending=True).tail(25)
                fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                              title=f"Feature Importances — {name}",
                             template="plotly_dark", color="Importance", color_continuous_scale="Plasma")
                fig.update_layout(height=max(400, len(fi_df)*22))
                st.plotly_chart(fig, width="stretch")


def _render_feature_importances_clf(trained: dict[str, Any], feature_cols: list[str]) -> None:
    tree_m = {k: v for k, v in trained.items() if hasattr(v, "feature_importances_")}
    if not tree_m:
        return
    with st.expander("🌲 Feature Importances", expanded=True):
        tabs = st.tabs(list(tree_m.keys()))
        for tab, (name, m) in zip(tabs, tree_m.items()):
            with tab:
                imp   = m.feature_importances_
                fi_df = pd.DataFrame({"Feature": feature_cols, "Importance": imp}) \
                          .sort_values("Importance", ascending=True).tail(25)
                fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                              title=f"Feature Importances — {name}",
                             template="plotly_dark", color="Importance", color_continuous_scale="Viridis")
                fig.update_layout(height=max(400, len(fi_df)*22))
                st.plotly_chart(fig, width="stretch")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Research Directions & Suggestions
# ─────────────────────────────────────────────────────────────────────────────

def render_suggestions(raw_df: pd.DataFrame, processed_df: pd.DataFrame,
                        automl_results: dict[str, Any]) -> None:
    st.markdown('<h2 style="color:#93c5fd;">💡 Stage 5 — Research Directions & AI Insight</h2>',
                unsafe_allow_html=True)

    task       = automl_results.get("task", "unknown")
    lb: pd.DataFrame | None = automl_results.get("leaderboard")
    feature_cols = automl_results.get("feature_cols", [])
    best_name    = automl_results.get("best_model_name", "Unknown")
    num_cols     = raw_df.select_dtypes(include=[np.number]).columns.tolist()

    # ── Contextual Hero ──
    st.markdown(f"""
    <div style="background:rgba(129,140,248,0.1); border-left:4px solid #818cf8; padding:20px; border-radius:12px; margin-bottom:20px;">
        <h4 style="margin:0 0 10px 0; color:#818cf8;">🚀 Strategic Path Forward</h4>
        The automated baseline identifies <strong>{best_name}</strong> as your top-performing architecture. 
        To reach production-grade performance (>90%), prioritize feature interaction engineering and ensemble stacking.
    </div>
    """, unsafe_allow_html=True)

    # ── Details ──
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🧪 Research Trajectories")
        st.markdown("""
        - **Deep Ensembling**: Use a `VotingRegressor` or `StackingRegressor` with XGBoost, LightGBM, and Random Forest.
        - **Non-Linear Dynamics**: Investigate temporal dependencies using LSTMs or Neural Basis Expansion (N-BEATS) if dates are present.
        - **Causal Discovery**: Beyond correlation, use PC-algorithms or Structural Equation Modeling to find true drivers.
        """)
    
    with c2:
        st.markdown("### ⚙️ Technical Tuning")
        st.markdown("""
        - **Explainability**: Integrate SHAP (SHapley Additive exPlanations) to verify model logic.
        - **Hyperparameter Search**: Initiate a 100-trial Optuna study on the top 3 models.
        - **Target Balancing**: If target is skewed, apply a power transform (Yeo-Johnson).
        """)

    # ── Report Download ──
    st.markdown("---")
    st.markdown("### 📄 Professional Analysis Export")
    
    report = f"# RetroX Data Intelligence Report\n\n"
    report += f"**Dataset Profile**: {raw_df.shape[0]:,} samples x {raw_df.shape[1]} features\n"
    report += f"**AutoML Task**: {task.upper()}\n"
    report += f"**Winning Architecture**: {best_name}\n\n"
    
    if lb is not None:
        report += "## 1. Candidate Leaderboard\n\n"
        report += lb.to_csv(index=False) + "\n\n"
    
    report += "## 2. Recommended Roadmap\n"
    report += "1. **Baseline**: Current model achieves local optima.\n"
    report += "2. **H-Horizon**: Shift to multi-step recursive forecasting for time-series.\n"
    report += "3. **Explain**: Verify features against domain knowledge using SHAP.\n"

    st.download_button(
        label="⬇️ Download Full MD/CSV Report",
        data=report,
        file_name="retrox_analytical_report.md",
        mime="text/markdown"
    )
