"""
DET English Program — CEFR Progress Report
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DET Program — CEFR Progress Report",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Constants ────────────────────────────────────────────────────────────────
CEFR_ORDER = ["Pre-A1", "A1", "A2", "B1", "B2", "C1", "C2"]
CEFR_NUMERIC = {
    "Pre-A1": 0,
    "A1": 1,
    "A2": 2,
    "B1": 3,
    "B2": 4,
    "C1": 5,
    "C2": 6,
}
CEFR_COLORS = {
    "Pre-A1": "#ef4444",
    "A1": "#f97316",
    "A2": "#eab308",
    "B1": "#22c55e",
    "B2": "#06b6d4",
    "C1": "#8b5cf6",
    "C2": "#ec4899",
}

# Remap raw data levels → display labels
# A1 (raw) → Pre-A1, Strong A1 (raw) → A1, A2 (raw) → A2, Strong A2 (raw) → A2
LEVEL_REMAP = {
    "A1": "Pre-A1",
    "Strong A1": "A1",
    "A2": "A2",
    "Strong A2": "A2",
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

    # overall_numeric computed after remap (see below load_data call)

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
            student_totals["total_sessions_norm"] * 0.5
            + student_totals["total_mins_norm"] * 0.5
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

# ─── Exclude February from all charts (keep raw effort for Monthly Summary) ──
CHART_MONTHS = ["December", "January"]
combined = combined[combined["month"].isin(CHART_MONTHS)].copy()
weekly_cefr = weekly_cefr[weekly_cefr["month"].isin(CHART_MONTHS)].copy()
# Recompute student_totals from Dec+Jan effort only (for charts)
effort_chart = effort[effort["month"].isin(CHART_MONTHS)].copy()
student_totals_chart = (
    effort_chart.groupby("student")
    .agg(
        total_sessions=("session_count", "sum"),
        total_mins=("total_duration_mins", "sum"),
        avg_consistency=("consistency_pct", "mean"),
    )
    .reset_index()
)
if len(student_totals_chart) > 0:
    for col in ["total_sessions", "total_mins", "avg_consistency"]:
        mn, mx = student_totals_chart[col].min(), student_totals_chart[col].max()
        student_totals_chart[f"{col}_norm"] = (student_totals_chart[col] - mn) / (
            mx - mn + 1e-9
        )
    student_totals_chart["engagement_score"] = (
        student_totals_chart["total_sessions_norm"] * 0.5
        + student_totals_chart["total_mins_norm"] * 0.5
    )
    p80 = student_totals_chart["engagement_score"].quantile(0.80)
    p40 = student_totals_chart["engagement_score"].quantile(0.40)
    student_totals_chart["engagement"] = student_totals_chart["engagement_score"].apply(
        lambda x: "High" if x >= p80 else ("Medium" if x >= p40 else "Low")
    )
# Replace student_totals used by charts (Monthly Summary keeps original `effort`)
student_totals = student_totals_chart


# ─── Apply display-level remap to all level columns ──────────────────────────
def remap_levels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    level_cols = [
        c
        for c in df.columns
        if c.endswith("_level")
        or c == "overall_level"
        or c in ("fluency", "accuracy", "range", "coherence")
    ]
    for col in level_cols:
        if col in df.columns:
            df[col] = df[col].map(
                lambda v: LEVEL_REMAP.get(v, v) if isinstance(v, str) else v
            )
    return df


combined = remap_levels(combined)
weekly_cefr = remap_levels(weekly_cefr)

# Compute overall_numeric after remap so labels are already clean
weekly_cefr["overall_numeric"] = weekly_cefr["overall_level"].map(CEFR_NUMERIC)


# ─── Compute per-student movement (dimension-level) ──────────────────────────
@st.cache_data(show_spinner=False)
def compute_student_movements(_combined, _version=2):
    results = []
    for student in _combined["student"].unique():
        sdata = _combined[_combined["student"] == student].sort_values("month")
        if len(sdata) < 2:
            row = sdata.iloc[0]
            entry = {
                "student": student,
                "engagement": row.get("engagement", "—"),
                "engagement_score": row.get("engagement_score", 0),
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
            "engagement_score": first.get("engagement_score", 0),
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
        f_label = first.get("overall_level", "—")
        l_label = last.get("overall_level", "—")
        f_idx = CEFR_ORDER.index(f_label) if f_label in CEFR_ORDER else -1
        l_idx = CEFR_ORDER.index(l_label) if l_label in CEFR_ORDER else -1
        label_movement = l_idx - f_idx  # based on CEFR label rank, not numeric
        entry["overall_first"] = f_label
        entry["overall_last"] = l_label
        entry["overall_movement"] = (
            label_movement  # 0 = stable, >0 = improved, <0 = declined
        )
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
    ["Program Overview", "Student Wise Data", "Weekly Report"],
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
    # AVG DAILY PRACTICE DISTRIBUTION — BY MONTH
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("Average Daily Practice — Distribution by Month")
    st.caption(
        "How many minutes per day each student practiced on average. Shows spread across the cohort."
    )

    MONTH_COLORS = {
        "December": ("rgba(147,197,253,0.5)", "#3b82f6"),  # blue
        "January": ("rgba(167,243,208,0.5)", "#10b981"),  # green
    }

    fig_adp = go.Figure()
    x_max = 0

    for month in CHART_MONTHS:
        month_data = (
            effort[effort["month"] == month]["avg_daily_mins"].dropna().astype(float)
        )
        if month_data.empty:
            continue

        n = len(month_data)
        mean_val = month_data.mean()
        year = "2025" if month == "December" else "2026"
        bar_color, line_color = MONTH_COLORS[month]
        bin_width = 0.5
        x_max = max(x_max, month_data.max())

        # Histogram
        fig_adp.add_trace(
            go.Histogram(
                x=month_data,
                xbins=dict(start=0, end=x_max + 1, size=bin_width),
                marker_color=bar_color,
                marker_line=dict(color=line_color, width=1),
                name=f"{month} {year} (n={n}, mean={mean_val:.1f})",
                opacity=0.8,
            )
        )

        # KDE curve
        kde = gaussian_kde(month_data, bw_method=0.4)
        x_range = np.linspace(0, x_max + 1, 300)
        kde_y_scaled = kde(x_range) * n * bin_width
        fig_adp.add_trace(
            go.Scatter(
                x=x_range,
                y=kde_y_scaled,
                mode="lines",
                line=dict(color=line_color, width=2.5),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Mean line
        fig_adp.add_vline(
            x=mean_val,
            line_dash="dash",
            line_color=line_color,
            line_width=1.5,
            annotation_text=f"{month[:3]} mean {mean_val:.1f}",
            annotation_position="top",
            annotation_font_color=line_color,
            annotation_font_size=11,
        )

    fig_adp.update_layout(
        barmode="overlay",
        plot_bgcolor="white",
        xaxis=dict(
            title="Avg Daily Practice (min)",
            showgrid=False,
            zeroline=False,
            range=[0, x_max + 1],
        ),
        yaxis=dict(
            title="Students", showgrid=True, gridcolor="#e2e8f0", zeroline=False
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=40, b=40),
        height=360,
    )
    st.plotly_chart(fig_adp, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    # CEFR LEVEL DISTRIBUTION — WHERE STUDENTS STAND
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("Student Outcome (CEFR)")
    st.caption("Number of students at each CEFR level, by month.")

    MONTH_BAR_COLORS = {"December": "#93c5fd", "January": "#6ee7b7"}

    fig_cefr_dist = go.Figure()
    for month in CHART_MONTHS:
        month_combined = combined[combined["month"] == month]
        if month_combined.empty:
            continue
        year = "2025" if month == "December" else "2026"
        n = len(month_combined)
        label = "Baseline" if month == "December" else "Midline"
        level_counts = (
            month_combined["overall_level"]
            .value_counts()
            .reindex(CEFR_ORDER, fill_value=0)
            .reset_index()
        )
        level_counts.columns = ["Level", "Students"]
        fig_cefr_dist.add_trace(
            go.Bar(
                x=level_counts["Level"],
                y=level_counts["Students"],
                text=level_counts["Students"],
                textposition="outside",
                name=f"{month} {year} (n={n}, {label})",
                marker_color=MONTH_BAR_COLORS[month],
                hovertemplate=f"<b>%{{x}}</b><br>{month}: %{{y}} students<extra></extra>",
            )
        )

    fig_cefr_dist.update_layout(
        barmode="group",
        plot_bgcolor="white",
        xaxis=dict(
            title="CEFR Level",
            categoryorder="array",
            categoryarray=CEFR_ORDER,
            showgrid=False,
        ),
        yaxis=dict(
            title="Students",
            showgrid=True,
            gridcolor="#e2e8f0",
            zeroline=False,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=40, b=40),
        height=360,
        bargap=0.2,
        bargroupgap=0.05,
    )
    st.plotly_chart(fig_cefr_dist, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    # LEVEL TRANSITIONS
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("Level Transitions")
    st.caption(
        "Students who moved up at least one full CEFR level — "
        "Pre-A1 → A1, A1 → A2, or Pre-A1 → A2."
    )

    VALID_TRANSITIONS = {
        ("Pre-A1", "A1"),
        ("A1", "A2"),
        ("Pre-A1", "A2"),
    }

    transition_rows = []
    for _, row in movements_df[movements_df["months_active"] >= 2].iterrows():
        pair = (row["overall_first"], row["overall_last"])
        if pair in VALID_TRANSITIONS:
            transition_rows.append(
                {
                    "Student": row["student"],
                    "From": row["overall_first"],
                    "To": row["overall_last"],
                }
            )

    if transition_rows:
        trans_df = pd.DataFrame(transition_rows).sort_values(["From", "To"])
        # Summary counts
        t1, t2, t3 = st.columns(3)
        for col, (frm, to) in zip(
            [t1, t2, t3], [("Pre-A1", "A1"), ("A1", "A2"), ("Pre-A1", "A2")]
        ):
            n = int(((trans_df["From"] == frm) & (trans_df["To"] == to)).sum())
            col.metric(f"{frm} → {to}", n)
        st.dataframe(trans_df, use_container_width=True, hide_index=True)
    else:
        st.info("No level transitions found in the current data.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: STUDENT LOOKUP
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Student Wise Data":
    st.title("Student Wise Data")

    selected = st.selectbox("Select Student", sorted(combined["student"].unique()))

    if selected:
        sdata = combined[combined["student"] == selected].sort_values("month")
        st_row = student_totals[student_totals["student"] == selected]
        mv_row = movements_df[movements_df["student"] == selected]

        sessions = int(st_row["total_sessions"].iloc[0]) if len(st_row) > 0 else 0
        mins = float(st_row["total_mins"].iloc[0]) if len(st_row) > 0 else 0
        consistency = float(st_row["avg_consistency"].iloc[0]) if len(st_row) > 0 else 0

        h1, h2, h3 = st.columns(3)
        h1.metric("Total Sessions", sessions)
        h2.metric("Total Practice", f"{mins / 60:.1f} hrs")
        h3.metric("Avg Consistency", f"{consistency:.0f}%")

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
                fig = px.bar(
                    pd.DataFrame(dim_data),
                    x="Dimension",
                    y="CEFR",
                    color="Month",
                    barmode="group",
                    height=350,
                    category_orders={
                        "Month": MONTH_ORDER,
                        "Dimension": [
                            "Overall",
                            "Fluency",
                            "Accuracy",
                            "Range",
                            "Coherence",
                        ],
                    },
                    color_discrete_sequence=["#93c5fd", "#6ee7b7", "#fca5a5"],
                )
                fig.update_yaxes(
                    tickvals=[0, 1, 2, 3, 4, 5, 6],
                    ticktext=["Pre-A1", "A1", "A2", "B1", "B2", "C1", "C2"],
                    range=[0, 6.5],
                    title="",
                )
                fig.update_layout(
                    xaxis_title="",
                    legend_title="Month",
                    font=dict(size=12),
                    plot_bgcolor="white",
                    margin=dict(l=40, r=20, t=20, b=30),
                )
                st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3: WEEKLY DETAIL (TEACHERS)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Weekly Report":
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
