
import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Optional ML: use sklearn if installed, otherwise fallback to transparent scoring
try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
    from sklearn.linear_model import LogisticRegression
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "synthetic_fraud_data.csv"

st.set_page_config(page_title="跨境支付反欺诈风控 Demo", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA, parse_dates=["signup_date"])
    return df

def transparent_score(row: dict) -> float:
    """
    A transparent, rule-ish score in [0, 1].
    This is used if sklearn is unavailable, and also as an explainable baseline.
    """
    # Normalize some inputs
    device_risk = row["device_risk_score"] / 100.0
    ip_risk = row["ip_risk_score"] / 100.0
    burst = row["burstiness"] / 100.0
    fast_first_tx = max(0.0, 1.0 - (row["hours_to_first_tx"] / 72.0))  # <=72h is riskier
    cards = min(row["num_cards_linked"], 6) / 6.0
    fail_login = min(row["failed_login_7d"], 25) / 25.0
    chb = min(row["chargeback_history"], 6) / 6.0

    channel = row["acq_channel"]
    country = row["country"]
    device = row["device_type"]

    # weights (sum ~ 1.0)
    score = (
        0.18*device_risk +
        0.18*ip_risk +
        0.14*burst +
        0.12*fast_first_tx +
        0.10*cards +
        0.10*fail_login +
        0.10*chb
    )

    if channel == "Affiliate":
        score += 0.07
    elif channel == "Ads":
        score += 0.03

    if country in ["CN","BR","IN","MX"]:
        score += 0.03

    if device == "Web":
        score += 0.02

    if int(row["kyc_passed"]) == 0:
        score += 0.06

    level = int(row["member_level"])
    score -= 0.01*level  # higher level slightly safer

    return float(min(max(score, 0.0), 1.0))

@st.cache_resource
def train_model(df: pd.DataFrame):
    """
    Train a quick logistic regression model on the synthetic dataset.
    Cached so it won't retrain on every rerun.
    """
    if not SKLEARN_OK:
        return None

    feature_cols = [
        "device_risk_score","ip_risk_score","chargeback_history","failed_login_7d",
        "num_cards_linked","tx_count_7d","tx_amount_7d","burstiness","hours_to_first_tx",
        "kyc_passed","member_level"
    ]
    X = df[feature_cols].copy()
    y = df["is_fraud"].astype(int).copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    pred = model.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, pred)
    # default threshold 0.5 for metrics
    y_hat = (pred >= 0.5).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_hat, average="binary", zero_division=0)

    return {
        "model": model,
        "feature_cols": feature_cols,
        "metrics": {"AUC": float(auc), "Precision": float(p), "Recall": float(r), "F1": float(f1)}
    }

def risk_bucket(p: float) -> str:
    if p < 0.05:
        return "Low"
    if p < 0.15:
        return "Medium"
    if p < 0.35:
        return "High"
    return "Very High"

df = load_data()
bundle = train_model(df)

# ---------------- Sidebar filters ----------------
st.sidebar.header("筛选器 Filters")
min_date, max_date = df["signup_date"].min(), df["signup_date"].max()
date_range = st.sidebar.date_input("注册日期范围", value=(min_date.date(), max_date.date()), min_value=min_date.date(), max_value=max_date.date())
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
else:
    start, end = min_date, max_date

country_sel = st.sidebar.multiselect("国家/地区", sorted(df["country"].unique().tolist()), default=sorted(df["country"].unique().tolist()))
channel_sel = st.sidebar.multiselect("获客渠道", sorted(df["acq_channel"].unique().tolist()), default=sorted(df["acq_channel"].unique().tolist()))
device_sel = st.sidebar.multiselect("设备类型", sorted(df["device_type"].unique().tolist()), default=sorted(df["device_type"].unique().tolist()))
kyc_sel = st.sidebar.multiselect("KYC 是否通过", [0,1], default=[0,1])

df_f = df[
    (df["signup_date"] >= start) & (df["signup_date"] <= end) &
    (df["country"].isin(country_sel)) &
    (df["acq_channel"].isin(channel_sel)) &
    (df["device_type"].isin(device_sel)) &
    (df["kyc_passed"].isin(kyc_sel))
].copy()

# ---------------- Header ----------------
st.title("跨境支付反欺诈风控 Demo（可交互 Dashboard）")
st.caption("这是一个“可展示在简历/面试”的端到端项目：数据 → 画像/路径分析 → 特征与模型 → 策略建议。数据为模拟生成，用于演示风控思路与可视化。")

# ---------------- Tabs ----------------
tab1, tab2, tab3, tab4 = st.tabs(["① 总览KPI", "② 用户画像", "③ 行为路径", "④ 风险评分/模型"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    total = len(df_f)
    fraud_rate = (df_f["is_fraud"].mean() if total else 0.0)
    very_high = (df_f["risk_bucket"].eq("Very High").mean() if total else 0.0)

    col1.metric("样本量", f"{total:,}")
    col2.metric("欺诈率(标签)", f"{fraud_rate*100:.2f}%")
    col3.metric("Very High 占比", f"{very_high*100:.2f}%")
    if bundle is not None:
        col4.metric("模型AUC(训练/测试)", f"{bundle['metrics']['AUC']:.3f}")
    else:
        col4.metric("模型", "未启用(无sklearn)")

    left, right = st.columns([1.1, 0.9])
    with left:
        st.subheader("欺诈趋势（按周）")
        trend = (df_f
                 .assign(week=df_f["signup_date"].dt.to_period("W").dt.start_time)
                 .groupby("week")["is_fraud"].mean()
                 .reset_index(name="fraud_rate"))
        st.line_chart(trend.set_index("week")["fraud_rate"])

    with right:
        st.subheader("风险分层占比")
        bucket = df_f["risk_bucket"].value_counts(normalize=True).reindex(["Low","Medium","High","Very High"]).fillna(0).reset_index()
        bucket.columns = ["risk_bucket", "share"]
        st.bar_chart(bucket.set_index("risk_bucket")["share"])

    st.subheader("数据预览")
    st.dataframe(df_f.head(30), use_container_width=True)

with tab2:
    st.subheader("画像对比：欺诈 vs 正常")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**设备类型分布**")
        dev = (df_f.groupby(["device_type","is_fraud"]).size().unstack(fill_value=0))
        dev_share = dev.div(dev.sum(axis=0), axis=1)
        st.bar_chart(dev_share)

        st.write("**获客渠道分布**")
        ch = (df_f.groupby(["acq_channel","is_fraud"]).size().unstack(fill_value=0))
        ch_share = ch.div(ch.sum(axis=0), axis=1)
        st.bar_chart(ch_share)

    with c2:
        st.write("**国家/地区 Top10（按欺诈率）**")
        cc = (df_f.groupby("country")["is_fraud"]
              .agg(["mean","count"])
              .rename(columns={"mean":"fraud_rate","count":"n"})
              .query("n >= 80")
              .sort_values("fraud_rate", ascending=False)
              .head(10))
        st.dataframe(cc.style.format({"fraud_rate":"{:.2%}"}), use_container_width=True)

        st.write("**关键连续特征对比（均值）**")
        feats = ["device_risk_score","ip_risk_score","burstiness","hours_to_first_tx","failed_login_7d","num_cards_linked","chargeback_history"]
        comp = df_f.groupby("is_fraud")[feats].mean().T
        comp.columns = ["正常(0)", "欺诈(1)"]
        st.dataframe(comp, use_container_width=True)

with tab3:
    st.subheader("黑产路径：注册 → 激活 → 爆发交易（概念演示）")
    st.write("这里用“代理指标”来模拟路径：**激活(activated)**、**首次交易时延(hours_to_first_tx)**、**爆发度(burstiness)**。你可以在简历/面试中讲清楚：我们如何将黑产行为拆成可观测的阶段与指标。")

    # Define stage flags
    d = df_f.copy()
    d["stage_register"] = 1
    d["stage_activate"] = d["activated"]
    d["stage_burst"] = ((d["tx_count_7d"] >= 12) & (d["burstiness"] >= 70) & (d["hours_to_first_tx"] <= 24)).astype(int)

    # Sankey-like counts (rendered as table + bars for streamlit simplicity)
    reg = len(d)
    act = int(d["stage_activate"].sum())
    burst = int(d["stage_burst"].sum())

    st.metric("注册人数", f"{reg:,}")
    cols = st.columns(3)
    cols[0].metric("激活人数", f"{act:,}", f"{act/reg*100:.1f}%")
    cols[1].metric("爆发交易人数", f"{burst:,}", f"{burst/reg*100:.1f}%")
    cols[2].metric("爆发交易欺诈率", f"{d.loc[d['stage_burst']==1,'is_fraud'].mean()*100:.1f}%")

    st.write("**阶段转化与风险**")
    table = pd.DataFrame({
        "Stage": ["Register", "Activate", "Burst"],
        "Users": [reg, act, burst],
        "Share of Register": [1.0, act/reg if reg else 0.0, burst/reg if reg else 0.0],
        "Fraud Rate": [
            d["is_fraud"].mean() if reg else 0.0,
            d.loc[d["stage_activate"]==1,"is_fraud"].mean() if act else 0.0,
            d.loc[d["stage_burst"]==1,"is_fraud"].mean() if burst else 0.0,
        ]
    })
    st.dataframe(table.style.format({"Share of Register":"{:.1%}","Fraud Rate":"{:.1%}"}), use_container_width=True)

    st.write("**爆发交易人群画像（对比全量）**")
    burst_df = d[d["stage_burst"]==1]
    if len(burst_df) > 0:
        colA, colB = st.columns(2)
        with colA:
            st.write("爆发交易：获客渠道占比")
            x = burst_df["acq_channel"].value_counts(normalize=True).reset_index()
            x.columns = ["acq_channel","share"]
            st.bar_chart(x.set_index("acq_channel")["share"])
        with colB:
            st.write("全量：获客渠道占比")
            y = d["acq_channel"].value_counts(normalize=True).reset_index()
            y.columns = ["acq_channel","share"]
            st.bar_chart(y.set_index("acq_channel")["share"])
    else:
        st.info("当前筛选条件下没有爆发交易样本。")

with tab4:
    st.subheader("单用户风险评分（可解释）")

    # Input widgets
    c1, c2, c3 = st.columns(3)
    with c1:
        device_type = st.selectbox("设备类型", ["iOS","Android","Web"])
        acq_channel = st.selectbox("获客渠道", ["Organic","Ads","Affiliate","Referral"])
        country = st.selectbox("国家/地区", sorted(df["country"].unique().tolist()))
        kyc_passed = st.selectbox("KYC 是否通过", [0,1], index=1)
    with c2:
        member_level = st.selectbox("会员等级(0-3)", [0,1,2,3], index=0)
        device_risk_score = st.slider("设备风险分(0-100)", 0.0, 100.0, 48.0, 0.5)
        ip_risk_score = st.slider("IP 风险分(0-100)", 0.0, 100.0, 44.0, 0.5)
        burstiness = st.slider("交易爆发度(0-100)", 0.0, 100.0, 55.0, 0.5)
    with c3:
        num_cards_linked = st.slider("绑卡数量(0-6)", 0, 6, 1, 1)
        failed_login_7d = st.slider("7天失败登录次数(0-25)", 0, 25, 2, 1)
        chargeback_history = st.slider("历史拒付/争议次数(0-6)", 0, 6, 0, 1)
        tx_count_7d = st.slider("7天交易笔数(0-60)", 0, 60, 6, 1)
        tx_amount_7d = st.slider("7天交易金额(0-4000)", 0.0, 4000.0, 380.0, 10.0)
        hours_to_first_tx = st.slider("注册到首次交易(小时)", 0.2, 240.0, 18.0, 0.2)

    row = {
        "device_type": device_type,
        "acq_channel": acq_channel,
        "country": country,
        "kyc_passed": int(kyc_passed),
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

    baseline_p = transparent_score(row)
    st.write("**透明规则评分（Explainable Baseline）**")
    st.progress(min(max(baseline_p, 0.0), 1.0))
    st.write(f"风险概率（规则）: **{baseline_p:.3f}**  ｜ 风险分层: **{risk_bucket(baseline_p)}**")

    if bundle is not None:
        st.write("---")
        st.write("**机器学习评分（Logistic Regression）**")
        X = pd.DataFrame([row])[bundle["feature_cols"]]
        ml_p = float(bundle["model"].predict_proba(X)[:,1][0])
        st.progress(min(max(ml_p, 0.0), 1.0))
        st.write(f"风险概率（模型）: **{ml_p:.3f}**  ｜ 风险分层: **{risk_bucket(ml_p)}**")
        st.caption("提示：模型只用数值特征训练（为了演示简洁），类别特征已通过规则层体现。")
        st.write("模型指标（测试集，阈值0.5）")
        st.json(bundle["metrics"])

    st.write("---")
    st.subheader("策略建议（示例）")
    st.markdown("""
- **Very High**：强制二次验证 / 限额 / 冻结可疑交易；优先触发设备+IP联合规则
- **High**：提高风控强度（更严格的频次与金额阈值）、加强KYC；对 Affiliate 渠道适度加严
- **Medium**：灰度观察，结合行为变化（爆发度上升、失败登录暴增）动态升级
- **Low**：保持顺畅体验；主要依赖事后监控与抽样风控
""")
