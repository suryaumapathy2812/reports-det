"""
DET English Program — CEFR Progress Report
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DET Program — CEFR Progress Report",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Constants ────────────────────────────────────────────────────────────────
CEFR_ORDER = ["Pre-A1", "A1", "Strong A1", "A2", "Strong A2", "B1", "B2"]
CEFR_NUMERIC = {
    "Pre-A1": 0,
    "A1": 1,
    "Strong A1": 1.5,
    "A2": 2,
    "Strong A2": 2.5,
    "B1": 3,
    "B2": 4,
}
CEFR_COLORS = {
    "Pre-A1": "#ef4444",
    "A1": "#f97316",
    "Strong A1": "#f59e0b",
    "A2": "#eab308",
    "Strong A2": "#84cc16",
    "B1": "#22c55e",
    "B2": "#06b6d4",
}
MONTH_ORDER = ["December", "January", "February"]
DIMENSIONS = ["fluency", "accuracy", "range", "coherence"]


# ─── Data Loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    monthly_cefr = pd.read_csv("data/monthly_cefr.csv")
    weekly_cefr = pd.read_csv("data/weekly_cefr.csv")
    effort = pd.read_csv("data/student_effort.csv")

    for df in [monthly_cefr, weekly_cefr, effort]:
        df["month"] = pd.Categorical(df["month"], categories=MONTH_ORDER, ordered=True)

    weekly_cefr["overall_numeric"] = weekly_cefr["overall_level"].map(CEFR_NUMERIC)

    # Merge effort into monthly CEFR
    combined = monthly_cefr.merge(
        effort[
            [
                "student",
                "month",
                "session_count",
                "total_duration_mins",
                "active_days",
                "total_program_days",
                "consistency_pct",
                "avg_daily_mins",
            ]
        ],
        on=["student", "month"],
        how="left",
        suffixes=("", "_effort"),
    )

    # Student-level totals & engagement bucketing
    student_totals = (
        effort.groupby("student")
        .agg(
            total_sessions=("session_count", "sum"),
            total_mins=("total_duration_mins", "sum"),
            avg_consistency=("consistency_pct", "mean"),
        )
        .reset_index()
    )

    if len(student_totals) > 0:
        for col in ["total_sessions", "total_mins", "avg_consistency"]:
            mn, mx = student_totals[col].min(), student_totals[col].max()
            student_totals[f"{col}_norm"] = (student_totals[col] - mn) / (
                mx - mn + 1e-9
            )

        student_totals["engagement_score"] = (
            student_totals["total_sessions_norm"] * 0.3
            + student_totals["total_mins_norm"] * 0.4
            + student_totals["avg_consistency_norm"] * 0.3
        )
        p80 = student_totals["engagement_score"].quantile(0.80)
        p40 = student_totals["engagement_score"].quantile(0.40)
        student_totals["engagement"] = student_totals["engagement_score"].apply(
            lambda x: "High" if x >= p80 else ("Medium" if x >= p40 else "Low")
        )

    combined = combined.merge(
        student_totals[
            [
                "student",
                "engagement_score",
                "engagement",
                "total_sessions",
                "total_mins",
                "avg_consistency",
            ]
        ],
        on="student",
        how="left",
        suffixes=("", "_total"),
    )

    return monthly_cefr, weekly_cefr, effort, combined, student_totals


monthly_cefr, weekly_cefr, effort, combined, student_totals = load_data()


# ─── Compute per-student movement (dimension-level) ──────────────────────────
@st.cache_data
def compute_student_movements(_combined):
    results = []
    for student in _combined["student"].unique():
        sdata = _combined[_combined["student"] == student].sort_values("month")
        if len(sdata) < 2:
            row = sdata.iloc[0]
            entry = {
                "student": student,
                "engagement": row.get("engagement", "—"),
                "months_active": 1,
                "total_sessions": row.get("total_sessions", 0),
                "total_mins": row.get("total_mins", 0),
            }
            for dim in DIMENSIONS:
                entry[f"{dim}_first"] = row.get(f"{dim}_level", "—")
                entry[f"{dim}_last"] = row.get(f"{dim}_level", "—")
                entry[f"{dim}_movement"] = 0.0
                entry[f"{dim}_arrow"] = "→"
            entry["overall_first"] = row.get("overall_level", "—")
            entry["overall_last"] = row.get("overall_level", "—")
            entry["overall_movement"] = 0.0
            entry["any_improved"] = False
            entry["any_declined"] = False
            entry["dims_improved"] = ""
            entry["dims_declined"] = ""
            results.append(entry)
            continue

        first = sdata.iloc[0]
        last = sdata.iloc[-1]
        entry = {
            "student": student,
            "engagement": first.get("engagement", "—"),
            "months_active": len(sdata),
            "total_sessions": first.get("total_sessions", 0),
            "total_mins": first.get("total_mins", 0),
        }

        improved_dims = []
        declined_dims = []
        for dim in DIMENSIONS:
            f_val = pd.to_numeric(first.get(f"{dim}_numeric"), errors="coerce") or 0
            l_val = pd.to_numeric(last.get(f"{dim}_numeric"), errors="coerce") or 0
            mv = float(l_val) - float(f_val)
            entry[f"{dim}_first"] = first.get(f"{dim}_level", "—")
            entry[f"{dim}_last"] = last.get(f"{dim}_level", "—")
            entry[f"{dim}_movement"] = round(mv, 2)
            entry[f"{dim}_arrow"] = "↑" if mv > 0 else ("↓" if mv < 0 else "→")
            if mv > 0:
                improved_dims.append(dim.title())
            elif mv < 0:
                declined_dims.append(dim.title())

        f_overall = pd.to_numeric(first.get("overall_numeric"), errors="coerce") or 0
        l_overall = pd.to_numeric(last.get("overall_numeric"), errors="coerce") or 0
        entry["overall_first"] = first.get("overall_level", "—")
        entry["overall_last"] = last.get("overall_level", "—")
        entry["overall_movement"] = round(float(l_overall) - float(f_overall), 2)
        entry["any_improved"] = len(improved_dims) > 0
        entry["any_declined"] = len(declined_dims) > 0
        entry["dims_improved"] = ", ".join(improved_dims)
        entry["dims_declined"] = ", ".join(declined_dims)
        results.append(entry)

    return pd.DataFrame(results)


movements_df = compute_student_movements(combined)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("📊 DET Program Report")
st.sidebar.caption("CEFR Progress — Dec 2025 to Feb 2026")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["Program Overview", "Student Lookup", "Weekly Detail (Teachers)"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Data Summary**")
st.sidebar.metric("Total Students", len(student_totals))
st.sidebar.metric("Total Sessions", f"{int(effort['session_count'].sum()):,}")
st.sidebar.metric(
    "Total Practice", f"{effort['total_duration_mins'].sum() / 60:.0f} hours"
)
st.sidebar.markdown("---")
for month in MONTH_ORDER:
    count = effort[effort["month"] == month]["student"].nunique()
    st.sidebar.caption(f"{month}: {count} students")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: PROGRAM OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Program Overview":
    st.title("Program Overview")
    st.markdown("**Baseline (Dec) → Midline (Jan) → Current (Feb, partial)**")

    # ══════════════════════════════════════════════════════════════════════
    # MONTHLY SUMMARY CARDS
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("Monthly Summary")

    mc1, mc2, mc3 = st.columns(3)
    for col, month in zip([mc1, mc2, mc3], MONTH_ORDER):
        month_effort = effort[effort["month"] == month]
        m_students = month_effort["student"].nunique()
        m_sessions = int(month_effort["session_count"].sum())
        m_mins = month_effort["total_duration_mins"].sum()
        m_hours = m_mins / 60
        label = (
            "Baseline"
            if month == "December"
            else ("Midline" if month == "January" else "Current (partial)")
        )

        with col:
            st.markdown(f"### {month}")
            st.caption(label)
            st.metric("Students", m_students)
            st.metric("Sessions", f"{m_sessions:,}")
            st.metric("Practice Time", f"{m_hours:.1f} hrs")

    # ══════════════════════════════════════════════════════════════════════
    # CEFR COHORT SHIFT — ONE CHART
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("CEFR Distribution — How the Cohort is Moving")

    dist_data = []
    for month in MONTH_ORDER:
        month_data = combined[combined["month"] == month]
        if len(month_data) == 0:
            continue
        total = len(month_data)
        for level in CEFR_ORDER:
            count = len(month_data[month_data["overall_level"] == level])
            if count > 0:
                dist_data.append(
                    {
                        "Month": month,
                        "CEFR Level": level,
                        "Students": count,
                        "Percentage": round(count / total * 100, 1),
                    }
                )

    if dist_data:
        dist_df = pd.DataFrame(dist_data)
        fig = px.bar(
            dist_df,
            x="Month",
            y="Percentage",
            color="CEFR Level",
            text=dist_df.apply(lambda r: f"{r['CEFR Level']}: {r['Students']}", axis=1),
            color_discrete_map=CEFR_COLORS,
            category_orders={"Month": MONTH_ORDER, "CEFR Level": CEFR_ORDER},
            barmode="stack",
            height=420,
        )
        fig.update_traces(textposition="inside", textfont_size=13)
        fig.update_layout(
            xaxis_title="",
            yaxis_title="% of Students",
            font=dict(size=13),
            margin=dict(l=40, r=20, t=20, b=30),
            legend_title="CEFR Level",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Summary sentence
    trackable = movements_df[movements_df["months_active"] >= 2]
    improved_count = int(trackable["any_improved"].sum())
    total_trackable = len(trackable)
    stable_count = total_trackable - improved_count
    st.markdown(
        f"Of **{total_trackable}** students tracked across months: "
        f"**{improved_count}** ({improved_count * 100 // max(total_trackable, 1)}%) improved in at least one CEFR dimension, "
        f"**{stable_count}** remained stable."
    )

    # ══════════════════════════════════════════════════════════════════════
    # EFFORT → RESULTS: THE STORIES YOGESH WANTS
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("Effort vs Results")
    st.caption(
        "Who put in the work and improved? Who didn't practice enough? "
        "Engagement = High (top 20%), Medium (middle 40%), Low (bottom 40%) "
        "— based on sessions, practice time, and consistency."
    )

    # Build the effort-results table
    trackable_df = movements_df[movements_df["months_active"] >= 2].copy()
    trackable_df["total_sessions"] = trackable_df["total_sessions"].astype(int)
    trackable_df["total_hrs"] = (trackable_df["total_mins"] / 60).round(1)

    def classify_student(row):
        high_effort = row["engagement"] in ["High", "Medium"]
        improved = row["any_improved"]
        if row["engagement"] == "High" and improved:
            return "✅ High effort, improved"
        elif row["engagement"] == "High" and not improved:
            return "⚠️ High effort, no improvement"
        elif row["engagement"] == "Medium" and improved:
            return "✅ Medium effort, improved"
        elif row["engagement"] == "Medium" and not improved:
            return "➡️ Medium effort, stable"
        elif row["engagement"] == "Low" and improved:
            return "🌟 Low effort, but improved"
        else:
            return "🔴 Low effort, no improvement"

    trackable_df["Category"] = trackable_df.apply(classify_student, axis=1)

    # ── Success stories: High effort + improved ───────────────────────
    st.markdown("#### ✅ Success Stories — High Effort & Improved")
    st.caption("Students who practiced consistently and showed CEFR improvement")

    success = trackable_df[
        (trackable_df["engagement"] == "High") & (trackable_df["any_improved"])
    ].sort_values("total_mins", ascending=False)

    if len(success) > 0:
        success_rows = []
        for _, row in success.iterrows():
            success_rows.append(
                {
                    "Student": row["student"],
                    "Sessions": int(row["total_sessions"]),
                    "Practice": f"{row['total_hrs']} hrs",
                    "CEFR": f"{row['overall_first']} → {row['overall_last']}",
                    "Improved In": row["dims_improved"],
                }
            )
        st.dataframe(
            pd.DataFrame(success_rows),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No high-engagement students with improvement found.")

    # ── Low effort, no improvement ────────────────────────────────────
    st.markdown("#### 🔴 Low Effort, No Improvement")
    st.caption("Students who haven't practiced enough and show no CEFR movement")

    low_no_improve = trackable_df[
        (trackable_df["engagement"] == "Low") & (~trackable_df["any_improved"])
    ].sort_values("total_mins", ascending=True)

    if len(low_no_improve) > 0:
        low_rows = []
        for _, row in low_no_improve.iterrows():
            low_rows.append(
                {
                    "Student": row["student"],
                    "Sessions": int(row["total_sessions"]),
                    "Practice": f"{row['total_hrs']} hrs",
                    "CEFR": f"{row['overall_first']} → {row['overall_last']}",
                    "Status": "No improvement",
                }
            )
        st.dataframe(
            pd.DataFrame(low_rows),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No low-engagement students without improvement.")

    # ── Full student list (collapsed) ─────────────────────────────────
    with st.expander("View all students — effort & CEFR movement"):
        all_rows = []
        for _, row in trackable_df.sort_values(
            "total_mins", ascending=False
        ).iterrows():
            all_rows.append(
                {
                    "Student": row["student"],
                    "Engagement": row["engagement"],
                    "Sessions": int(row["total_sessions"]),
                    "Practice": f"{row['total_hrs']} hrs",
                    "CEFR": f"{row['overall_first']} → {row['overall_last']}",
                    "Improved In": row["dims_improved"]
                    if row["dims_improved"]
                    else "—",
                    "Declined In": row["dims_declined"]
                    if row["dims_declined"]
                    else "—",
                }
            )
        st.dataframe(
            pd.DataFrame(all_rows),
            use_container_width=True,
            hide_index=True,
            height=500,
        )

        # Also show single-month students
        single_month = movements_df[movements_df["months_active"] < 2]
        if len(single_month) > 0:
            st.caption(
                f"**{len(single_month)}** students have only 1 month of data — not enough to track movement."
            )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: STUDENT LOOKUP
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Student Lookup":
    st.title("Student Lookup")

    selected = st.selectbox("Select Student", sorted(combined["student"].unique()))

    if selected:
        sdata = combined[combined["student"] == selected].sort_values("month")
        st_row = student_totals[student_totals["student"] == selected]
        mv_row = movements_df[movements_df["student"] == selected]

        engagement = st_row["engagement"].iloc[0] if len(st_row) > 0 else "—"
        sessions = int(st_row["total_sessions"].iloc[0]) if len(st_row) > 0 else 0
        mins = float(st_row["total_mins"].iloc[0]) if len(st_row) > 0 else 0
        consistency = float(st_row["avg_consistency"].iloc[0]) if len(st_row) > 0 else 0

        eng_emoji = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}

        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Engagement", f"{eng_emoji.get(engagement, '')} {engagement}")
        h2.metric("Total Sessions", sessions)
        h3.metric("Total Practice", f"{mins / 60:.1f} hrs")
        h4.metric("Avg Consistency", f"{consistency:.0f}%")

        # Movement summary
        if len(mv_row) > 0 and mv_row.iloc[0]["months_active"] >= 2:
            row = mv_row.iloc[0]
            dims_up = row["dims_improved"]
            dims_down = row["dims_declined"]
            if dims_up:
                st.success(f"**Improved in:** {dims_up}")
            elif dims_down:
                st.warning(f"**Declined in:** {dims_down}")
            else:
                st.info("**Stable** across all dimensions")

        st.markdown("---")
        st.subheader("Monthly CEFR Scores")
        detail = sdata[
            [
                "month",
                "overall_level",
                "fluency_level",
                "accuracy_level",
                "range_level",
                "coherence_level",
                "session_count",
                "total_duration_mins",
                "consistency_pct",
            ]
        ].copy()
        detail.columns = [
            "Month",
            "Overall",
            "Fluency",
            "Accuracy",
            "Range",
            "Coherence",
            "Sessions",
            "Mins",
            "Consistency %",
        ]
        st.dataframe(detail, use_container_width=True, hide_index=True)

        # Dimension progression chart
        if len(sdata) > 1:
            st.subheader("Progression")
            dim_data = []
            for _, row in sdata.iterrows():
                for dim in DIMENSIONS:
                    val = pd.to_numeric(row.get(f"{dim}_numeric"), errors="coerce")
                    if pd.notna(val):
                        dim_data.append(
                            {
                                "Month": str(row["month"]),
                                "Dimension": dim.title(),
                                "CEFR": float(val),
                            }
                        )
                oval = pd.to_numeric(row.get("overall_numeric"), errors="coerce")
                if pd.notna(oval):
                    dim_data.append(
                        {
                            "Month": str(row["month"]),
                            "Dimension": "Overall",
                            "CEFR": float(oval),
                        }
                    )

            if dim_data:
                fig = px.line(
                    pd.DataFrame(dim_data),
                    x="Month",
                    y="CEFR",
                    color="Dimension",
                    markers=True,
                    height=350,
                    category_orders={"Month": MONTH_ORDER},
                )
                fig.update_yaxes(
                    tickvals=[0, 1, 1.5, 2, 2.5, 3, 4],
                    ticktext=[
                        "Pre-A1",
                        "A1",
                        "Str A1",
                        "A2",
                        "Str A2",
                        "B1",
                        "B2",
                    ],
                    range=[0, 3.5],
                )
                fig.update_layout(
                    xaxis_title="",
                    font=dict(size=12),
                    margin=dict(l=40, r=20, t=20, b=30),
                )
                st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3: WEEKLY DETAIL (TEACHERS)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Weekly Detail (Teachers)":
    st.title("Weekly CEFR Detail")

    f1, f2 = st.columns(2)
    with f1:
        sel_student = st.selectbox(
            "Student", ["All"] + sorted(weekly_cefr["student"].unique().tolist())
        )
    with f2:
        sel_month = st.multiselect("Month", MONTH_ORDER, default=MONTH_ORDER)

    filtered = weekly_cefr[weekly_cefr["month"].isin(sel_month)]
    if sel_student != "All":
        filtered = filtered[filtered["student"] == sel_student]

    if sel_student != "All" and len(filtered) > 1:
        chart_data = []
        for _, row in filtered.iterrows():
            for dim in DIMENSIONS + ["overall_level"]:
                col = dim if dim != "overall_level" else "overall_level"
                label = dim.title() if dim != "overall_level" else "Overall"
                val = CEFR_NUMERIC.get(row.get(col, ""), None)
                if val is not None:
                    chart_data.append(
                        {"Week": row["week"], "Dimension": label, "CEFR": val}
                    )

        if chart_data:
            fig = px.line(
                pd.DataFrame(chart_data),
                x="Week",
                y="CEFR",
                color="Dimension",
                markers=True,
                height=380,
            )
            fig.update_yaxes(
                tickvals=[0, 1, 1.5, 2, 2.5, 3, 4],
                ticktext=["Pre-A1", "A1", "Str A1", "A2", "Str A2", "B1", "B2"],
                range=[0, 3.5],
            )
            fig.update_layout(
                xaxis_title="", font=dict(size=12), margin=dict(l=40, r=20, t=20, b=30)
            )
            st.plotly_chart(fig, use_container_width=True)

    display = filtered[
        [
            "student",
            "week",
            "month",
            "overall_level",
            "fluency",
            "accuracy",
            "range",
            "coherence",
            "session_count",
            "total_duration_mins",
        ]
    ].copy()
    display.columns = [
        "Student",
        "Week",
        "Month",
        "Overall",
        "Fluency",
        "Accuracy",
        "Range",
        "Coherence",
        "Sessions",
        "Mins",
    ]
    st.dataframe(display, use_container_width=True, height=500, hide_index=True)

    st.download_button(
        "Download CSV", display.to_csv(index=False), "weekly_cefr.csv", "text/csv"
    )

    if sel_student != "All" and len(filtered) > 0:
        st.markdown("---")
        st.subheader("Scoring Evidence")
        for _, row in filtered.iterrows():
            with st.expander(f"{row['week']} — Overall: {row['overall_level']}"):
                st.markdown(
                    f"**Fluency ({row['fluency']}):** {row.get('fluency_evidence', '—')}"
                )
                st.markdown(
                    f"**Accuracy ({row['accuracy']}):** {row.get('accuracy_errors', '—')}"
                )
                st.markdown(
                    f"**Range ({row['range']}):** {row.get('range_vocabulary', '—')}"
                )
                st.markdown(
                    f"**Coherence ({row['coherence']}):** {row.get('coherence_structure', '—')}"
                )


# ─── Footer ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.caption("Deshpande Education Trust")
st.sidebar.caption("Data: Dec 12, 2025 — Feb 19, 2026")
