"""
    cd ~/Desktop/IA/demo/IA && unset PYTHONPATH && /opt/anaconda3/bin/python -m streamlit run dashboard.py
"""
import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score

# ═══ Load data ═══
df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "fall_risk_patients_2000_v2.csv"))
for col in ["high_risk_medication", "orthostatic_hypotension"]:
    df[col] = df[col].map({True: 1, False: 0})

RISK_ORDER = ["LOW", "MEDIUM", "HIGH"]
RISK_COLOR = {"LOW": "#4CAF50", "MEDIUM": "#FFC107", "HIGH": "#F44336"}

st.set_page_config(page_title="跌倒風險分析", layout="wide")
st.title(" 老年人跌倒風險分析")
st.caption("Group 3 · 2000 個病人 · 10 個臨床特徵")

# ═══ Sidebar ═══
with st.sidebar:
    st.header(" 概覽")
    total = len(df)
    c = df["fall_risk_level"].value_counts()
    st.metric("總病人", f"{total:,}")
    st.metric("HIGH 風險", f"{c.get('HIGH', 0):,} ({c.get('HIGH', 0)/total:.1%})")
    st.metric("MEDIUM 風險", f"{c.get('MEDIUM', 0):,} ({c.get('MEDIUM', 0)/total:.1%})")
    st.metric("LOW 風險", f"{c.get('LOW', 0):,} ({c.get('LOW', 0)/total:.1%})")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    " 風險分佈", " 特徵分佈", " 相關性", " 年齡分析", " Model 比較"
])

# ═══════════════════════════════════
# Tab 1: 風險分佈
# ═══════════════════════════════════
with tab1:
    st.subheader(" 風險級別分佈")

    col1, col2 = st.columns(2)

    with col1:
        # Pie chart
        counts = df["fall_risk_level"].value_counts().reindex(RISK_ORDER)
        fig = px.pie(
            values=counts.values, names=counts.index,
            color=counts.index, color_discrete_map=RISK_COLOR, hole=0.4,
            title="風險級別比例",
        )
        fig.update_layout(showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Bar chart
        fig = px.bar(
            counts, color=counts.index, color_discrete_map=RISK_COLOR,
            title="風險級別人數",
        )
        fig.update_layout(xaxis_title="風險級別", yaxis_title="人數")
        st.plotly_chart(fig, use_container_width=True)

    # 佔比
    st.write("#### 各風險級別人數")
    st.dataframe(
        pd.DataFrame({
            "風險級別": counts.index,
            "人數": counts.values,
            "佔比": [f"{v/total:.1%}" for v in counts.values],
        }),
        use_container_width=True, hide_index=True,
    )

# ═══════════════════════════════════
# Tab 2: 特徵分佈
# ═══════════════════════════════════
with tab2:
    st.subheader(" 特徵分佈（按風險級別）")

    feat_options = {
        "age": "年齡",
        "night_bed_exits": "夜間離床次數",
        "night_activity_duration_min": "夜間活動分鐘",
        "past_falls": "過去跌倒次數",
        "mobility_score": "行動能力分數",
        "polypharmacy_count": "多重用藥數量",
        "tug_seconds": "TUG 測試秒數",
    }

    sel = st.selectbox("選擇特徵", list(feat_options.keys()),
                       format_func=lambda k: feat_options[k])

    col1, col2 = st.columns(2)

    with col1:
        # Box plot
        fig = px.box(
            df, x="fall_risk_level", y=sel, color="fall_risk_level",
            color_discrete_map=RISK_COLOR, category_orders={"fall_risk_level": RISK_ORDER},
            title=f"{feat_options[sel]} 分佈 (Box Plot)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Histogram
        fig = px.histogram(
            df, x=sel, color="fall_risk_level", nbins=20,
            color_discrete_map=RISK_COLOR, marginal="box",
            category_orders={"fall_risk_level": RISK_ORDER},
            title=f"{feat_options[sel]} 直方圖",
        )
        st.plotly_chart(fig, use_container_width=True)

    # 統計摘要
    st.write(f"#### {feat_options[sel]} 各風險級別統計")
    stat = df.groupby("fall_risk_level")[sel].describe().loc[RISK_ORDER][["mean", "std", "min", "max"]]
    st.dataframe(stat.round(2), use_container_width=True)

# ═══════════════════════════════════
# Tab 3: 相關性
# ═══════════════════════════════════
with tab3:
    st.subheader(" 特徵相關性矩陣")

    num_cols = ["age", "night_bed_exits", "night_activity_duration_min",
                "past_falls", "mobility_score", "polypharmacy_count", "tug_seconds"]
    corr = df[num_cols].corr()

    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, aspect="auto", title="特徵相關性 (Pearson)",
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.write("**觀察：**")
    st.markdown("""
    - `tug_seconds`（TUG 測試）與 `mobility_score`（行動能力）呈**負相關** — TUG 越慢，行動分數越低
    - `night_bed_exits`（離床次數）與 `night_activity_duration_min`（活動時間）**正相關** — 離床越多，活動越久
    """)

# ═══════════════════════════════════
# Tab 4: 年齡分析
# ═══════════════════════════════════
with tab4:
    st.subheader(" 年齡分析")

    # Age distribution by risk
    fig = px.histogram(
        df, x="age", color="fall_risk_level", nbins=15,
        color_discrete_map=RISK_COLOR, marginal="violin",
        category_orders={"fall_risk_level": RISK_ORDER},
        title="年齡分佈（按風險級別）",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Age box by risk
    fig = px.box(
        df, x="fall_risk_level", y="age", color="fall_risk_level",
        color_discrete_map=RISK_COLOR, category_orders={"fall_risk_level": RISK_ORDER},
        title="各風險級別年齡範圍",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Age group table
    df["age_group"] = pd.cut(df["age"], bins=[0, 70, 80, 90, 200], labels=["<70", "70-79", "80-89", "90+"])
    age_risk = pd.crosstab(df["age_group"], df["fall_risk_level"]).reindex(["<70", "70-79", "80-89", "90+"])[RISK_ORDER]
    st.write("#### 年齡組 × 風險級別")
    st.dataframe(age_risk, use_container_width=True)

# ═══════════════════════════════════
# Tab 5: Model 比較 (Kero vs Alex)
# ═══════════════════════════════════
REQUIRED = ["age", "night_bed_exits", "night_activity_duration_min",
            "past_falls", "mobility_score", "high_risk_medication",
            "cognitive_impairment", "polypharmacy_count",
            "orthostatic_hypotension", "tug_seconds"]
y = df["fall_risk_level"]


def high_recall(y_true, y_pred):
    return recall_score(y_true, y_pred, average=None, zero_division=0)[0]


def run_alex():
    X = df[REQUIRED]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    lr = LogisticRegression(C=100, max_iter=5000, random_state=42)
    lr.fit(X_train_s, y_train)
    lr_pred = lr.predict(X_test_s)
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    y_num_train = y_train.map({"LOW": 0, "MEDIUM": 1, "HIGH": 2})
    xgb = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6,
                        random_state=42, eval_metric='mlogloss')
    xgb.fit(X_train, y_num_train)
    xgb_pred = pd.Series(xgb.predict(X_test)).map({0: "LOW", 1: "MEDIUM", 2: "HIGH"}).values
    return {"LR": (lr_pred, y_test), "RF": (rf_pred, y_test), "XGB": (xgb_pred, y_test)}


def run_kero():
    X = df[REQUIRED]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.333, random_state=42, stratify=y_temp)
    lr = LogisticRegression(max_iter=5000, class_weight='balanced', random_state=42)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_val)
    rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_val)
    le = LabelEncoder()
    yt = le.fit_transform(y_train)
    xgb = XGBClassifier(n_estimators=200, max_depth=6, random_state=42)
    xgb.fit(X_train, yt)
    xgb_pred = le.inverse_transform(xgb.predict(X_val))
    return {"LR": (lr_pred, y_val), "RF": (rf_pred, y_val), "XGB": (xgb_pred, y_val)}


with tab5:
    st.subheader(" Model 比較 — Kero vs Alex")

    with st.spinner("正在訓練 3 個 models (LR / RF / XGB)..."):
        alex = run_alex()
        kero = run_kero()

    # Collect results
    rows = []
    for model in ["LR", "RF", "XGB"]:
        a_acc = accuracy_score(alex[model][1], alex[model][0])
        a_rec = high_recall(alex[model][1], alex[model][0])
        k_acc = accuracy_score(kero[model][1], kero[model][0])
        k_rec = high_recall(kero[model][1], kero[model][0])
        rows.append({
            "Model": model,
            "Alex Acc": a_acc, "Alex HIGH recall": a_rec,
            "Kero Acc": k_acc, "Kero HIGH recall": k_rec,
        })

    cmp_df = pd.DataFrame(rows)

    # HIGH recall grouped bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Kero", x=cmp_df["Model"], y=cmp_df["Kero HIGH recall"],
                         marker_color="#2E7D32", text=[f"{v:.3f}" for v in cmp_df["Kero HIGH recall"]]))
    fig.add_trace(go.Bar(name="Alex", x=cmp_df["Model"], y=cmp_df["Alex HIGH recall"],
                         marker_color="#C62828", text=[f"{v:.3f}" for v in cmp_df["Alex HIGH recall"]]))
    fig.update_layout(barmode="group", yaxis=dict(range=[0, 1]),
                      title="HIGH Recall 比較 (越高越好)", yaxis_title="HIGH recall")
    st.plotly_chart(fig, use_container_width=True)

    # Accuracy bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Kero", x=cmp_df["Model"], y=cmp_df["Kero Acc"],
                         marker_color="#2E7D32", text=[f"{v:.1%}" for v in cmp_df["Kero Acc"]]))
    fig.add_trace(go.Bar(name="Alex", x=cmp_df["Model"], y=cmp_df["Alex Acc"],
                         marker_color="#C62828", text=[f"{v:.1%}" for v in cmp_df["Alex Acc"]]))
    fig.update_layout(barmode="group", yaxis=dict(range=[0, 1], tickformat=".0%"),
                      title="Accuracy 比較", yaxis_title="Accuracy")
    st.plotly_chart(fig, use_container_width=True)

    st.write("#### 詳細數據")
    st.dataframe(cmp_df.round(4), use_container_width=True, hide_index=True)

    # Verdict
    k_lr, a_lr = cmp_df.loc[cmp_df["Model"] == "LR", "Kero HIGH recall"].values[0], \
                 cmp_df.loc[cmp_df["Model"] == "LR", "Alex HIGH recall"].values[0]
    st.success(f"**結論：** class_weight='balanced' 對 HIGH recall 最有效 "
               f"(Kero LR {k_lr:.3f} vs Alex LR {a_lr:.3f})")
