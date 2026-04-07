
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ── Optional ML dependencies ───────────────────────────────────────────────────
try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        precision_recall_fscore_support,
        roc_curve, precision_recall_curve,
        confusion_matrix,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

try:
    from xgboost import XGBClassifier
    XGB_OK = True
except Exception:
    XGB_OK = False

try:
    import shap
    SHAP_OK = True
except Exception:
    SHAP_OK = False

# ── Constants ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "synthetic_fraud_data.csv"

FEATURE_COLS = [
    "device_risk_score", "ip_risk_score", "chargeback_history", "failed_login_7d",
    "num_cards_linked", "tx_count_7d", "tx_amount_7d", "burstiness",
    "hours_to_first_tx", "kyc_passed", "member_level",
]
FEATURE_ZH = {
    "device_risk_score":  "设备风险分",
    "ip_risk_score":      "IP风险分",
    "chargeback_history": "历史拒付次数",
    "failed_login_7d":    "7天登录失败",
    "num_cards_linked":   "绑卡数量",
    "tx_count_7d":        "7天交易笔数",
    "tx_amount_7d":       "7天交易金额",
    "burstiness":         "交易爆发度",
    "hours_to_first_tx":  "首次交易时延(h)",
    "kyc_passed":         "KYC通过",
    "member_level":       "会员等级",
}
MODEL_COLORS = {
    "Logistic Regression": "#636EFA",
    "Random Forest":       "#00CC96",
    "XGBoost":             "#EF553B",
}
RISK_COLORS = {
    "Low": "#2ECC71", "Medium": "#F39C12",
    "High": "#E67E22", "Very High": "#E74C3C",
}

st.set_page_config(page_title="跨境支付反欺诈风控 Demo", layout="wide")

# ── Data ───────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv(DATA, parse_dates=["signup_date"])


# ── Rule-based transparent scorer ─────────────────────────────────────────────
def transparent_score(row: dict) -> float:
    """加权规则评分，完全可解释，输出 [0,1] 风险概率。"""
    device_risk = row["device_risk_score"] / 100.0
    ip_risk     = row["ip_risk_score"] / 100.0
    burst       = row["burstiness"] / 100.0
    fast_first  = max(0.0, 1.0 - row["hours_to_first_tx"] / 72.0)
    cards       = min(row["num_cards_linked"], 6) / 6.0
    fail_login  = min(row["failed_login_7d"], 25) / 25.0
    chb         = min(row["chargeback_history"], 6) / 6.0

    score = (
        0.18 * device_risk + 0.18 * ip_risk + 0.14 * burst +
        0.12 * fast_first  + 0.10 * cards   + 0.10 * fail_login + 0.10 * chb
    )
    if row["acq_channel"] == "Affiliate": score += 0.07
    elif row["acq_channel"] == "Ads":     score += 0.03
    if row["country"] in ["CN", "BR", "IN", "MX"]: score += 0.03
    if row["device_type"] == "Web":       score += 0.02
    if int(row["kyc_passed"]) == 0:       score += 0.06
    score -= 0.01 * int(row["member_level"])
    return float(min(max(score, 0.0), 1.0))


def risk_label(p: float) -> str:
    if p < 0.05: return "Low"
    if p < 0.15: return "Medium"
    if p < 0.35: return "High"
    return "Very High"


# ── Model training (all 3 models) ──────────────────────────────────────────────
@st.cache_resource
def train_models(_df: pd.DataFrame):
    if not SKLEARN_OK:
        return None

    X = _df[FEATURE_COLS].copy()
    y = _df["is_fraud"].astype(int).copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Logistic Regression needs standardised features
    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_train)
    X_te_sc  = scaler.transform(X_test)

    trained, probs = {}, {}

    # 1. Logistic Regression
    lr = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
    lr.fit(X_tr_sc, y_train)
    probs["Logistic Regression"] = lr.predict_proba(X_te_sc)[:, 1]
    trained["Logistic Regression"] = {"model": lr, "scaler": scaler, "type": "lr"}

    # 2. Random Forest
    rf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    probs["Random Forest"] = rf.predict_proba(X_test)[:, 1]
    trained["Random Forest"] = {"model": rf, "scaler": None, "type": "tree"}

    # 3. XGBoost (if installed)
    if XGB_OK:
        pos_w = max(1, int((y == 0).sum() / max((y == 1).sum(), 1)))
        xgb_m = XGBClassifier(
            n_estimators=200, scale_pos_weight=pos_w,
            random_state=42, eval_metric="logloss", verbosity=0,
        )
        xgb_m.fit(X_train, y_train)
        probs["XGBoost"] = xgb_m.predict_proba(X_test)[:, 1]
        trained["XGBoost"] = {"model": xgb_m, "scaler": None, "type": "tree"}

    # Compute metrics
    metrics = {}
    for name, p in probs.items():
        auc = roc_auc_score(y_test, p)
        ap  = average_precision_score(y_test, p)
        yh  = (p >= 0.5).astype(int)
        pr, rc, f1, _ = precision_recall_fscore_support(
            y_test, yh, average="binary", zero_division=0
        )
        metrics[name] = {
            "AUC-ROC":       round(float(auc), 4),
            "AP(PR)":        round(float(ap),  4),
            "Precision@0.5": round(float(pr),  4),
            "Recall@0.5":    round(float(rc),  4),
            "F1@0.5":        round(float(f1),  4),
        }

    return {
        "trained":  trained,
        "probs":    probs,
        "y_test":   y_test.values,
        "X_train":  X_train,
        "X_test":   X_test,
        "metrics":  metrics,
    }


# ── Bootstrap ──────────────────────────────────────────────────────────────────
df     = load_data()
bundle = train_models(df)

# ── Sidebar filters ────────────────────────────────────────────────────────────
st.sidebar.header("筛选器 Filters")
min_d, max_d = df["signup_date"].min(), df["signup_date"].max()
date_range = st.sidebar.date_input(
    "注册日期范围",
    value=(min_d.date(), max_d.date()),
    min_value=min_d.date(), max_value=max_d.date(),
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
else:
    start, end = min_d, max_d

country_sel = st.sidebar.multiselect("国家/地区", sorted(df["country"].unique()), default=sorted(df["country"].unique()))
channel_sel = st.sidebar.multiselect("获客渠道",  sorted(df["acq_channel"].unique()), default=sorted(df["acq_channel"].unique()))
device_sel  = st.sidebar.multiselect("设备类型",  sorted(df["device_type"].unique()), default=sorted(df["device_type"].unique()))
kyc_sel     = st.sidebar.multiselect("KYC是否通过", [0, 1], default=[0, 1])

df_f = df[
    (df["signup_date"] >= start) & (df["signup_date"] <= end) &
    df["country"].isin(country_sel) &
    df["acq_channel"].isin(channel_sel) &
    df["device_type"].isin(device_sel) &
    df["kyc_passed"].isin(kyc_sel)
].copy()

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("跨境支付反欺诈风控 Demo")
st.caption(
    "端到端项目：数据 → 用户画像 → 行为路径 → "
    "多模型对比（LR / Random Forest / XGBoost）→ SHAP 可解释风险评分。"
    "数据为模拟生成。"
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["① 总览KPI", "② 用户画像", "③ 行为路径", "④ 模型性能", "⑤ 风险评分/SHAP"]
)

# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 ── KPI Overview
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    total      = len(df_f)
    fraud_rate = df_f["is_fraud"].mean() if total else 0.0
    very_high  = df_f["risk_bucket"].eq("Very High").mean() if total else 0.0
    best_auc   = (
        max(bundle["metrics"].values(), key=lambda m: m["AUC-ROC"])["AUC-ROC"]
        if bundle else None
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("样本量", f"{total:,}")
    c2.metric("欺诈率（标签）", f"{fraud_rate*100:.2f}%")
    c3.metric("Very High 占比", f"{very_high*100:.2f}%")
    c4.metric("最佳模型 AUC", f"{best_auc:.3f}" if best_auc else "未启用")

    left, right = st.columns([1.2, 0.8])

    with left:
        st.subheader("欺诈趋势（按周）")
        trend = (
            df_f
            .assign(week=df_f["signup_date"].dt.to_period("W").dt.start_time)
            .groupby("week")["is_fraud"].mean()
            .reset_index(name="fraud_rate")
        )
        fig = px.line(
            trend, x="week", y="fraud_rate",
            markers=True, line_shape="spline",
            labels={"week": "注册周", "fraud_rate": "欺诈率"},
            color_discrete_sequence=["#EF553B"],
        )
        fig.update_yaxes(tickformat=".1%")
        fig.update_layout(height=330, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("风险分层分布")
        bucket = (
            df_f["risk_bucket"]
            .value_counts(normalize=True)
            .reindex(["Low", "Medium", "High", "Very High"])
            .fillna(0)
            .reset_index()
        )
        bucket.columns = ["risk_bucket", "share"]
        fig2 = px.bar(
            bucket, x="risk_bucket", y="share",
            color="risk_bucket",
            color_discrete_map=RISK_COLORS,
            text=bucket["share"].map("{:.1%}".format),
            labels={"risk_bucket": "风险等级", "share": "占比"},
        )
        fig2.update_traces(textposition="outside")
        fig2.update_layout(showlegend=False, height=330, margin=dict(t=20, b=20))
        fig2.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("数据预览（前30行）")
    st.dataframe(df_f.head(30), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 ── User Portraits
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("画像对比：欺诈 vs 正常")
    c1, c2 = st.columns(2)

    with c1:
        dev = (
            df_f.groupby(["device_type", "is_fraud"])
            .size().reset_index(name="count")
        )
        dev["群体"] = dev["is_fraud"].map({0: "正常", 1: "欺诈"})
        fig = px.bar(
            dev, x="device_type", y="count", color="群体", barmode="group",
            color_discrete_map={"正常": "#636EFA", "欺诈": "#EF553B"},
            labels={"device_type": "设备类型", "count": "用户数"},
            title="设备类型 × 欺诈分布",
        )
        fig.update_layout(height=300, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

        ch = (
            df_f.groupby(["acq_channel", "is_fraud"])
            .size().reset_index(name="count")
        )
        ch["群体"] = ch["is_fraud"].map({0: "正常", 1: "欺诈"})
        fig2 = px.bar(
            ch, x="acq_channel", y="count", color="群体", barmode="group",
            color_discrete_map={"正常": "#636EFA", "欺诈": "#EF553B"},
            labels={"acq_channel": "获客渠道", "count": "用户数"},
            title="获客渠道 × 欺诈分布",
        )
        fig2.update_layout(height=300, margin=dict(t=40, b=20))
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        cc = (
            df_f.groupby("country")["is_fraud"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "fraud_rate", "count": "n"})
            .query("n >= 80")
            .sort_values("fraud_rate", ascending=False)
            .head(12)
            .reset_index()
        )
        fig3 = px.bar(
            cc, x="fraud_rate", y="country", orientation="h",
            color="fraud_rate", color_continuous_scale="RdYlGn_r",
            text=cc["fraud_rate"].map("{:.1%}".format),
            labels={"fraud_rate": "欺诈率", "country": "国家"},
            title="各国欺诈率 Top 12（样本 ≥ 80）",
        )
        fig3.update_traces(textposition="outside")
        fig3.update_xaxes(tickformat=".0%")
        fig3.update_layout(
            height=380, margin=dict(t=40, b=20),
            yaxis={"categoryorder": "total ascending"},
        )
        st.plotly_chart(fig3, use_container_width=True)

        feats = [
            "device_risk_score", "ip_risk_score", "burstiness",
            "hours_to_first_tx", "failed_login_7d", "num_cards_linked", "chargeback_history",
        ]
        comp = df_f.groupby("is_fraud")[feats].mean().T.reset_index()
        comp.columns = ["feature", "正常(0)", "欺诈(1)"]
        comp["feature"] = comp["feature"].map(lambda f: FEATURE_ZH.get(f, f))
        comp_m = comp.melt("feature", var_name="群体", value_name="均值")
        fig4 = px.bar(
            comp_m, x="均值", y="feature", color="群体",
            barmode="group", orientation="h",
            color_discrete_map={"正常(0)": "#636EFA", "欺诈(1)": "#EF553B"},
            title="关键特征均值对比（欺诈 vs 正常）",
        )
        fig4.update_layout(height=380, margin=dict(t=40, b=20))
        st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 ── Behavior Path / Funnel
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("黑产路径：注册 → 激活 → 爆发交易")
    st.write("用可观测的代理指标将黑产行为拆成三个可量化阶段，帮助风控团队确定最优拦截点。")

    d = df_f.copy()
    d["stage_activate"] = d["activated"]
    d["stage_burst"] = (
        (d["tx_count_7d"] >= 12) &
        (d["burstiness"] >= 70) &
        (d["hours_to_first_tx"] <= 24)
    ).astype(int)

    reg   = len(d)
    act   = int(d["stage_activate"].sum())
    burst = int(d["stage_burst"].sum())

    funnel_df = pd.DataFrame({
        "阶段": ["注册", "激活", "爆发交易"],
        "人数": [reg, act, burst],
        "欺诈率": [
            d["is_fraud"].mean() if reg else 0.0,
            d.loc[d["stage_activate"] == 1, "is_fraud"].mean() if act else 0.0,
            d.loc[d["stage_burst"] == 1, "is_fraud"].mean() if burst else 0.0,
        ],
    })

    col_f, col_b = st.columns([1.2, 0.8])
    with col_f:
        fig_funnel = go.Figure(go.Funnel(
            y=funnel_df["阶段"],
            x=funnel_df["人数"],
            textinfo="value+percent initial",
            marker={"color": ["#636EFA", "#EF553B", "#FF6B6B"]},
            connector={"line": {"color": "#aaa", "width": 2}},
        ))
        fig_funnel.update_layout(title="漏斗：各阶段用户规模", height=360, margin=dict(t=50))
        st.plotly_chart(fig_funnel, use_container_width=True)

    with col_b:
        fig_fr = px.bar(
            funnel_df, x="阶段", y="欺诈率",
            color="欺诈率", color_continuous_scale="RdYlGn_r",
            text=funnel_df["欺诈率"].map("{:.1%}".format),
            title="各阶段欺诈率",
            range_y=[0, min(funnel_df["欺诈率"].max() * 1.4 + 0.01, 1.0)],
        )
        fig_fr.update_traces(textposition="outside")
        fig_fr.update_yaxes(tickformat=".0%")
        fig_fr.update_layout(showlegend=False, height=360, margin=dict(t=50))
        st.plotly_chart(fig_fr, use_container_width=True)

    if burst > 0:
        burst_ch = (
            d[d["stage_burst"] == 1]["acq_channel"]
            .value_counts(normalize=True).reset_index()
        )
        burst_ch.columns = ["channel", "share"]
        burst_ch["group"] = "爆发交易人群"
        all_ch = d["acq_channel"].value_counts(normalize=True).reset_index()
        all_ch.columns = ["channel", "share"]
        all_ch["group"] = "全量用户"
        cmp = pd.concat([burst_ch, all_ch])
        fig_ch = px.bar(
            cmp, x="channel", y="share", color="group", barmode="group",
            color_discrete_map={"爆发交易人群": "#EF553B", "全量用户": "#636EFA"},
            text=cmp["share"].map("{:.1%}".format),
            labels={"channel": "获客渠道", "share": "占比", "group": ""},
            title="爆发交易人群 vs 全量用户：获客渠道对比",
        )
        fig_ch.update_traces(textposition="outside")
        fig_ch.update_yaxes(tickformat=".0%")
        fig_ch.update_layout(height=360, margin=dict(t=50))
        st.plotly_chart(fig_ch, use_container_width=True)
    else:
        st.info("当前筛选条件下没有爆发交易样本。")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 4 ── Model Performance
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    if bundle is None:
        st.warning("scikit-learn 未安装，请运行：pip install scikit-learn xgboost shap")
    else:
        # ── Metrics comparison table ───────────────────────────────────────────
        st.subheader("多模型性能对比")

        metrics_df = (
            pd.DataFrame(bundle["metrics"]).T
            .reset_index().rename(columns={"index": "模型"})
        )

        def highlight_best(s):
            is_max = s == s.max()
            return [
                "background-color: #d4edda; font-weight: bold" if v else ""
                for v in is_max
            ]

        st.dataframe(
            metrics_df.set_index("模型").style.apply(highlight_best).format("{:.4f}"),
            use_container_width=True,
        )
        st.caption(
            "绿色底色 = 该列最优  ｜  "
            "AP(PR) = Average Precision，比 AUC-ROC 更关注欺诈少数类表现  ｜  "
            "class_weight='balanced' 已对样本不平衡做补偿"
        )

        # ── ROC & PR curves ────────────────────────────────────────────────────
        col_roc, col_pr = st.columns(2)

        with col_roc:
            st.subheader("ROC 曲线")
            fig_roc = go.Figure()
            fig_roc.add_shape(
                type="line", line=dict(dash="dash", color="lightgray", width=1),
                x0=0, x1=1, y0=0, y1=1,
            )
            for name, p in bundle["probs"].items():
                fpr, tpr, _ = roc_curve(bundle["y_test"], p)
                auc = bundle["metrics"][name]["AUC-ROC"]
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode="lines",
                    name=f"{name} (AUC={auc:.3f})",
                    line=dict(color=MODEL_COLORS.get(name, "#999"), width=2.5),
                ))
            fig_roc.update_layout(
                xaxis_title="False Positive Rate（误报率）",
                yaxis_title="True Positive Rate（召回率）",
                height=420,
                legend=dict(x=0.38, y=0.08),
                margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        with col_pr:
            st.subheader("PR 曲线（Precision-Recall）")
            fig_pr = go.Figure()
            baseline_pr = float(bundle["y_test"].mean())
            fig_pr.add_shape(
                type="line", line=dict(dash="dash", color="lightgray", width=1),
                x0=0, x1=1, y0=baseline_pr, y1=baseline_pr,
            )
            for name, p in bundle["probs"].items():
                prec_v, rec_v, _ = precision_recall_curve(bundle["y_test"], p)
                ap = bundle["metrics"][name]["AP(PR)"]
                fig_pr.add_trace(go.Scatter(
                    x=rec_v, y=prec_v, mode="lines",
                    name=f"{name} (AP={ap:.3f})",
                    line=dict(color=MODEL_COLORS.get(name, "#999"), width=2.5),
                ))
            fig_pr.update_layout(
                xaxis_title="Recall（召回率）",
                yaxis_title="Precision（精确率）",
                height=420,
                legend=dict(x=0.05, y=0.12),
                margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig_pr, use_container_width=True)

        # ── Feature Importance ─────────────────────────────────────────────────
        st.subheader("特征重要性")
        model_sel = st.selectbox("选择模型", list(bundle["trained"].keys()), key="fi_model")
        info = bundle["trained"][model_sel]

        if info["type"] == "lr":
            importances = np.abs(info["model"].coef_[0])
            imp_label = "｜系数绝对值｜（标准化特征）"
        else:
            importances = info["model"].feature_importances_
            imp_label = "特征重要性（Gini/增益）"

        fi_df = pd.DataFrame({
            "feature":    FEATURE_COLS,
            "feature_zh": [FEATURE_ZH.get(f, f) for f in FEATURE_COLS],
            "importance": importances,
        }).sort_values("importance")

        fig_fi = px.bar(
            fi_df, x="importance", y="feature_zh", orientation="h",
            color="importance", color_continuous_scale="Blues",
            text=fi_df["importance"].map("{:.4f}".format),
            labels={"importance": imp_label, "feature_zh": "特征"},
            title=f"{model_sel} — 特征重要性排序",
        )
        fig_fi.update_traces(textposition="outside")
        fig_fi.update_layout(height=420, showlegend=False, margin=dict(t=50, b=20, r=80))
        st.plotly_chart(fig_fi, use_container_width=True)

        # ── Confusion Matrix ───────────────────────────────────────────────────
        st.subheader("混淆矩阵")
        best_name = (
            "XGBoost" if "XGBoost" in bundle["trained"]
            else ("Random Forest" if "Random Forest" in bundle["trained"]
                  else "Logistic Regression")
        )
        cm_pred = (bundle["probs"][best_name] >= 0.5).astype(int)
        cm = confusion_matrix(bundle["y_test"], cm_pred)
        cm_df = pd.DataFrame(
            cm,
            index=["实际:正常(0)", "实际:欺诈(1)"],
            columns=["预测:正常(0)", "预测:欺诈(1)"],
        )
        fig_cm = px.imshow(
            cm_df, text_auto=True,
            color_continuous_scale="Blues",
            title=f"混淆矩阵（{best_name}，分类阈值=0.5）",
            aspect="auto",
        )
        fig_cm.update_layout(height=370, margin=dict(t=60, b=20))

        cm_col, cm_exp = st.columns([1, 1])
        with cm_col:
            st.plotly_chart(fig_cm, use_container_width=True)
        with cm_exp:
            st.markdown("""
**如何读混淆矩阵：**

| | 预测正常 | 预测欺诈 |
|---|---|---|
| **实际正常** | ✅ 真负例 TN | ⚠️ 假正例 FP（误报） |
| **实际欺诈** | ❌ 假负例 FN（**漏报**） | ✅ 真正例 TP（命中） |

风控中的核心权衡：
- **漏报(FN)** = 欺诈被放过，直接造成资损
- **误报(FP)** = 好用户被拦，体验受损、客诉增加

→ **调低阈值**：提高召回，减少漏报，但误报增加
→ **调高阈值**：减少误报，但漏报增加

→ 实际业务会根据**损失成本比**设定最优阈值
""")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 5 ── Real-time Risk Scoring + SHAP
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("单用户实时风险评分 + SHAP 可解释分析")

    # Input widgets
    c1, c2, c3 = st.columns(3)
    with c1:
        device_type        = st.selectbox("设备类型", ["iOS", "Android", "Web"])
        acq_channel        = st.selectbox("获客渠道", ["Organic", "Ads", "Affiliate", "Referral"])
        country            = st.selectbox("国家/地区", sorted(df["country"].unique().tolist()))
        kyc_passed         = st.selectbox("KYC是否通过", [0, 1], index=1)
    with c2:
        member_level       = st.selectbox("会员等级(0-3)", [0, 1, 2, 3])
        device_risk_score  = st.slider("设备风险分", 0.0, 100.0, 48.0, 0.5)
        ip_risk_score      = st.slider("IP风险分", 0.0, 100.0, 44.0, 0.5)
        burstiness         = st.slider("交易爆发度", 0.0, 100.0, 55.0, 0.5)
    with c3:
        num_cards_linked   = st.slider("绑卡数量", 0, 6, 1)
        failed_login_7d    = st.slider("7天登录失败", 0, 25, 2)
        chargeback_history = st.slider("历史拒付次数", 0, 6, 0)
        tx_count_7d        = st.slider("7天交易笔数", 0, 60, 6)
        tx_amount_7d       = st.slider("7天交易金额", 0.0, 4000.0, 380.0, 10.0)
        hours_to_first_tx  = st.slider("注册到首次交易(小时)", 0.2, 240.0, 18.0, 0.2)

    row = {
        "device_type": device_type, "acq_channel": acq_channel,
        "country": country, "kyc_passed": int(kyc_passed),
        "member_level": int(member_level),
        "device_risk_score": float(device_risk_score),
        "ip_risk_score": float(ip_risk_score),
        "burstiness": float(burstiness),
        "num_cards_linked": int(num_cards_linked),
        "failed_login_7d": int(failed_login_7d),
        "chargeback_history": int(chargeback_history),
        "tx_count_7d": int(tx_count_7d),
        "tx_amount_7d": float(tx_amount_7d),
        "hours_to_first_tx": float(hours_to_first_tx),
    }

    # Rule-based score
    baseline_p = transparent_score(row)

    # ML scores
    ml_scores: dict[str, float] = {}
    if bundle:
        X_inp = pd.DataFrame([row])[FEATURE_COLS]
        for name, info in bundle["trained"].items():
            X_s = info["scaler"].transform(X_inp) if info["scaler"] else X_inp
            ml_scores[name] = float(info["model"].predict_proba(X_s)[:, 1][0])

    all_scores = {"规则评分": baseline_p, **ml_scores}

    # Score comparison bar chart
    score_df = pd.DataFrame({
        "模型": list(all_scores.keys()),
        "风险概率": list(all_scores.values()),
        "风险等级": [risk_label(v) for v in all_scores.values()],
    })
    score_df["标签"] = score_df.apply(
        lambda r: f"{r['风险概率']:.3f}  ({r['风险等级']})", axis=1
    )

    fig_score = px.bar(
        score_df, x="模型", y="风险概率",
        color="风险概率",
        color_continuous_scale=[
            [0,    "#2ECC71"],
            [0.05, "#2ECC71"],
            [0.15, "#F39C12"],
            [0.35, "#E67E22"],
            [1,    "#E74C3C"],
        ],
        text="标签",
        title="各模型风险评分对比",
        range_y=[0, 1],
    )
    fig_score.add_hline(y=0.05, line_dash="dot", line_color="#27AE60",
                        annotation_text="Low / Medium 阈值", annotation_position="right")
    fig_score.add_hline(y=0.15, line_dash="dot", line_color="#F39C12",
                        annotation_text="Medium / High 阈值", annotation_position="right")
    fig_score.add_hline(y=0.35, line_dash="dot", line_color="#E67E22",
                        annotation_text="High / VeryHigh 阈值", annotation_position="right")
    fig_score.update_traces(textposition="outside")
    fig_score.update_layout(
        height=420, showlegend=False, margin=dict(t=60, b=20, r=160)
    )
    st.plotly_chart(fig_score, use_container_width=True)

    # ── SHAP explainability ────────────────────────────────────────────────────
    st.divider()
    st.subheader("SHAP 可解释性分析")
    st.write(
        "每个特征对该用户最终预测分数的**贡献方向与大小**。"
        "红色 = 推高风险，绿色 = 降低风险。"
    )

    if not SHAP_OK:
        st.warning("请安装 shap 库：`pip install shap`")
    elif bundle is None:
        st.warning("需要训练好的模型，请确保 scikit-learn 已安装。")
    else:
        shap_candidates = [n for n in ["XGBoost", "Random Forest"] if n in bundle["trained"]]
        if not shap_candidates:
            st.warning("SHAP TreeExplainer 需要 XGBoost 或 Random Forest 模型。")
        else:
            shap_name  = shap_candidates[0]
            shap_model = bundle["trained"][shap_name]["model"]
            X_inp      = pd.DataFrame([row])[FEATURE_COLS]

            with st.spinner("计算 SHAP 值中..."):
                explainer = shap.TreeExplainer(shap_model)
                shap_vals = explainer.shap_values(X_inp)

            # Normalise shap_vals and expected_value across different shap/model versions:
            # - RF (older shap):  shap_vals is list[class0, class1], expected_value is array[2]
            # - XGBoost:          shap_vals may be 2-D array (n_samples, n_features) or list
            ev = explainer.expected_value
            if isinstance(shap_vals, list):
                # list → take class-1 slice
                sv       = np.array(shap_vals[1][0])
                base_val = float(np.atleast_1d(ev)[1])
            elif np.ndim(shap_vals) == 3:
                # shape (n_samples, n_features, n_classes) → class-1
                sv       = np.array(shap_vals[0, :, 1])
                base_val = float(np.atleast_1d(ev)[1])
            else:
                # shape (n_samples, n_features) → already for positive class
                sv       = np.array(shap_vals[0])
                base_val = float(np.atleast_1d(ev)[0])

            shap_df = pd.DataFrame({
                "feature":    FEATURE_COLS,
                "feature_zh": [FEATURE_ZH.get(f, f) for f in FEATURE_COLS],
                "shap_value": sv,
                "feat_val":   [row[f] for f in FEATURE_COLS],
            })
            shap_df["label"] = shap_df.apply(
                lambda r: f"{r['feature_zh']} = {r['feat_val']:.1f}", axis=1
            )
            shap_df["方向"] = shap_df["shap_value"].map(
                lambda v: "增加风险 ↑" if v > 0 else "降低风险 ↓"
            )
            shap_df = shap_df.sort_values("shap_value")

            fig_shap = px.bar(
                shap_df, x="shap_value", y="label", orientation="h",
                color="方向",
                color_discrete_map={"增加风险 ↑": "#EF553B", "降低风险 ↓": "#00CC96"},
                text=shap_df["shap_value"].map("{:+.4f}".format),
                labels={"shap_value": "SHAP值（对预测分数的贡献）", "label": "特征 = 当前值"},
                title=f"SHAP 特征贡献瀑布图（{shap_name}）",
            )
            fig_shap.update_traces(textposition="outside")
            fig_shap.add_vline(x=0, line_dash="solid", line_color="black", line_width=1.5)
            fig_shap.update_layout(
                height=500, margin=dict(t=60, b=20, r=110),
                xaxis_title="SHAP值（正 = 推高风险，负 = 降低风险）",
            )
            st.plotly_chart(fig_shap, use_container_width=True)

            model_prob = ml_scores.get(shap_name, float("nan"))
            st.caption(
                f"**基准值**（全体用户平均预测）= {base_val:.4f}  ｜  "
                f"各特征 SHAP 之和 = {sv.sum():+.4f}  ｜  "
                f"**{shap_name} 实际输出概率** = {model_prob:.4f}"
            )
            st.info(
                "💡 面试解读：SHAP 值来自博弈论的 Shapley 值，能公平地将模型预测分解到每个特征。"
                "对风控而言，这解决了「为什么这个用户被标记为高风险」的可解释性问题，"
                "可用于申诉回溯、监管合规与规则优化。"
            )

    # ── Strategy recommendation ────────────────────────────────────────────────
    st.divider()
    st.subheader("策略建议")
    max_p  = max(all_scores.values())
    label  = risk_label(max_p)

    strategy_map = {
        "Very High": ("🔴 **极高风险**",  "#FADBD8",
                      "强制冻结账户 / 拒绝交易 / 触发人工审核；优先联合设备+IP规则"),
        "High":      ("🟠 **高风险**",    "#FDEBD0",
                      "要求短信/人脸二次验证；设置单笔与累计交易限额；对 Affiliate 渠道额外加严"),
        "Medium":    ("🟡 **中等风险**",  "#FEFDE7",
                      "灰度观察期；结合实时行为变化（爆发度上升、登录失败暴增）动态升级风险等级"),
        "Low":       ("🟢 **低风险**",    "#EAFAF1",
                      "保持流畅用户体验；依赖事后监控与定期抽样风控"),
    }
    title_str, bg, action = strategy_map[label]
    st.markdown(
        f'<div style="background:{bg};padding:16px;border-radius:8px;">'
        f'{title_str} — {action}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("**各模型评分汇总：**")
    summary = pd.DataFrame({
        "模型":   list(all_scores.keys()),
        "风险概率": [f"{v:.4f}" for v in all_scores.values()],
        "风险等级": [risk_label(v) for v in all_scores.values()],
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)
