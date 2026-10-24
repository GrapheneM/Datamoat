"""Plotly figure builders. Every function returns a plotly Figure (or None if no data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

PRIMARY = "#1F4E79"
ACCENT = "#C0504D"
GOOD = "#2E7D5B"
GREY = "#8A94A3"
FONT = "Inter, Segoe UI, Helvetica, Arial, sans-serif"


def _style(fig: go.Figure, title: str = "", height: int = 360, legend: bool = False) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, x=0.0, xanchor="left", font=dict(size=16)),
        font=dict(family=FONT, size=13, color="#1F2933"),
        margin=dict(l=10, r=10, t=55 if title else 20, b=10),
        height=height,
        showlegend=legend,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        hoverlabel=dict(font_size=12),
    )
    return fig


def _text(values: np.ndarray, fmt: str) -> np.ndarray:
    def one(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        if fmt == "pct":
            return f"{v:.0%}"
        if fmt == "int":
            return f"{int(round(v))}"
        return f"{v:.1f}"
    return np.vectorize(one, otypes=[object])(values)


def heatmap(
    long: pd.DataFrame, x: str, y: str, value: str, *, title: str = "",
    colorscale: str = "RdYlGn", zmin: float | None = None, zmax: float | None = None,
    zmid: float | None = None, fmt: str = "pct", colorbar_title: str = "",
    annotate: bool = True, height: int | None = None,
) -> go.Figure | None:
    """Matrix heatmap from long-format data, preserving first-seen order of rows/columns."""
    if long.empty:
        return None
    xs = list(dict.fromkeys(long[x].astype(str)))
    ys = list(dict.fromkeys(long[y].astype(str)))
    frame = long.assign(**{x: long[x].astype(str), y: long[y].astype(str)})
    pivot = frame.pivot_table(index=y, columns=x, values=value, aggfunc="mean").reindex(index=ys, columns=xs)
    z = pivot.values.astype(float)
    kwargs: dict = {}
    if zmid is not None:
        kwargs["zmid"] = zmid
    fig = go.Figure(go.Heatmap(
        z=z, x=xs, y=ys, colorscale=colorscale, zmin=zmin, zmax=zmax, xgap=2, ygap=2,
        hoverongaps=False, colorbar=dict(title=colorbar_title, thickness=12, len=0.9),
        text=_text(z, fmt) if annotate and z.size <= 400 else None,
        texttemplate="%{text}" if annotate and z.size <= 400 else None,
        textfont=dict(size=11),
        hovertemplate=f"{y}: %{{y}}<br>{x}: %{{x}}<br>{colorbar_title or value}: %{{text}}<extra></extra>"
        if annotate and z.size <= 400 else None,
        **kwargs,
    ))
    fig.update_yaxes(autorange="reversed", title=None, automargin=True)
    fig.update_xaxes(title=None, side="top" if len(xs) <= 8 else "bottom", tickangle=-35 if len(xs) > 8 else 0, automargin=True)
    return _style(fig, title, height or int(min(720, max(200, 34 * len(ys) + 110))))


def attendance_heatmap(attendance: pd.DataFrame, students: pd.DataFrame) -> go.Figure | None:
    """Batch x week attendance rate (latest 16 weeks)."""
    if attendance.empty or students.empty:
        return None
    att = attendance.copy()
    att["student_id"] = att["student_id"].astype(str)
    att["_d"] = pd.to_datetime(att["date"], errors="coerce")
    att["present"] = att["present"].map(
        lambda v: float(v) if isinstance(v, (bool, np.bool_))
        else float(str(v).strip().casefold() in {"1", "true", "yes", "present"})
    )
    roster = students[["student_id", "batch_id"]].copy()
    roster["student_id"] = roster["student_id"].astype(str)
    att = att.dropna(subset=["_d"]).merge(roster, on="student_id", how="inner")
    if att.empty:
        return None
    att["week_start"] = att["_d"].dt.to_period("W").dt.start_time
    weeks = sorted(att["week_start"].unique())[-16:]
    grid = (att[att["week_start"].isin(weeks)].groupby(["batch_id", "week_start"], as_index=False)
            .agg(rate=("present", "mean")).sort_values(["batch_id", "week_start"]))
    grid["week"] = grid["week_start"].dt.strftime("%d %b")
    return heatmap(grid, "week", "batch_id", "rate", title="Attendance by batch and week",
                   zmin=0.4, zmax=1.0, colorbar_title="Attendance")


def mastery_heatmap(mastery: pd.DataFrame) -> go.Figure | None:
    if mastery.empty:
        return None
    grid = mastery.groupby(["batch_id", "topic"], as_index=False).agg(mastery=("mastery", "mean"))
    return heatmap(grid, "topic", "batch_id", "mastery", title="Topic mastery by batch",
                   zmin=0, zmax=1, colorbar_title="Mastery")


def student_signal_heatmap(summaries: dict, limit: int = 25) -> go.Figure | None:
    """Lowest students across test average, recent tests, attendance and homework."""
    cols = ["Test average", "Recent tests", "Attendance", "Homework"]
    rows = []
    for item in summaries.values():
        vals = [item.get("score_mean"), item.get("score_recent"),
                item.get("attendance_recent"), item.get("homework_rate")]
        if all(v is None for v in vals):
            continue
        rows.append(dict(zip(["student"] + cols, [item["name"]] + vals)))
    if not rows:
        return None
    wide = pd.DataFrame(rows)
    wide["_m"] = wide[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    wide = wide.sort_values("_m").head(limit)
    long = wide.melt(id_vars="student", value_vars=cols, var_name="signal", value_name="value")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    return heatmap(long, "signal", "student", "value",
                   title=f"Lowest {len(wide)} students across observed signals",
                   zmin=0, zmax=1, colorbar_title="Level",
                   height=int(min(760, max(220, 26 * len(wide) + 110))))


def risk_band_heatmap(risk: pd.DataFrame) -> go.Figure | None:
    if risk.empty:
        return None
    order = ["High support priority", "Needs a check-in", "Steady", "Insufficient data"]
    grid = risk.groupby(["batch_id", "band"], as_index=False).size().rename(columns={"size": "students"})
    grid["band"] = pd.Categorical(grid["band"], categories=order, ordered=True)
    grid = grid.sort_values(["batch_id", "band"])
    return heatmap(grid, "band", "batch_id", "students", title="Students by batch and support band",
                   colorscale="Blues", fmt="int", zmin=0, colorbar_title="Students")


def trend_chart(tests: pd.DataFrame) -> go.Figure | None:
    """Institute accuracy per test over time. `tests` has columns _date and accuracy."""
    if tests.empty:
        return None
    t = tests.sort_values("_date")
    fig = go.Figure(go.Scatter(
        x=t["_date"], y=t["accuracy"], mode="lines+markers",
        line=dict(color=PRIMARY, width=3), marker=dict(size=8, color=PRIMARY),
        fill="tozeroy", fillcolor="rgba(31,78,121,0.08)",
        hovertemplate="%{x|%d %b %Y}<br>Accuracy: %{y:.0%}<extra></extra>",
    ))
    fig.update_yaxes(tickformat=".0%", range=[0, 1], title=None, gridcolor="#E4E7EB")
    fig.update_xaxes(title=None)
    return _style(fig, "Institute test performance over time", 340)


def batch_topic_bars(grouped: pd.DataFrame, topic: str) -> go.Figure | None:
    """Mastery by batch for one topic. `grouped` has batch_id, topic, mastery, attempts."""
    d = grouped[grouped["topic"] == topic].sort_values("mastery")
    if d.empty:
        return None
    fig = go.Figure(go.Bar(
        x=d["mastery"], y=d["batch_id"].astype(str), orientation="h",
        marker=dict(color=d["mastery"], colorscale="RdYlGn", cmin=0, cmax=1),
        text=[f"{v:.0%}" for v in d["mastery"]], textposition="outside",
        customdata=d["attempts"], hovertemplate="%{y}<br>Mastery %{x:.0%}<br>%{customdata} attempts<extra></extra>",
    ))
    fig.update_xaxes(range=[0, 1.1], tickformat=".0%", title=None, gridcolor="#E4E7EB")
    fig.update_yaxes(title=None)
    return _style(fig, f"Mastery by batch: {topic}", max(220, 44 * len(d) + 90))


def calibration_chart(bins: pd.DataFrame) -> go.Figure | None:
    if bins.empty:
        return None
    top = float(max(bins["mean_predicted"].max(), bins["observed_rate"].max(), 0.1)) * 1.15
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", name="Perfect calibration",
                             line=dict(color=GREY, dash="dash")))
    fig.add_trace(go.Scatter(
        x=bins["mean_predicted"], y=bins["observed_rate"], mode="lines+markers", name="Model",
        line=dict(color=PRIMARY, width=3),
        marker=dict(size=np.clip(np.sqrt(bins["n"]) * 2.2, 8, 26), color=PRIMARY),
        customdata=bins["n"],
        hovertemplate="Predicted %{x:.1%}<br>Observed %{y:.1%}<br>n = %{customdata}<extra></extra>",
    ))
    fig.update_xaxes(range=[0, top], tickformat=".0%", title="Predicted probability", gridcolor="#E4E7EB")
    fig.update_yaxes(range=[0, top], tickformat=".0%", title="Observed departure rate", gridcolor="#E4E7EB")
    fig.update_layout(legend=dict(orientation="h", y=-0.25))
    return _style(fig, "Calibration on held-out test period", 360, legend=True)


def confusion_heatmap(conf: dict) -> go.Figure:
    z = np.array([[conf["tn"], conf["fp"]], [conf["fn"], conf["tp"]]], dtype=float)
    fig = go.Figure(go.Heatmap(
        z=z, x=["Not flagged", "Flagged"], y=["Stayed", "Left"], colorscale="Blues", xgap=3, ygap=3,
        text=_text(z, "int"), texttemplate="%{text}", textfont=dict(size=20), showscale=False,
        hovertemplate="Actual: %{y}<br>%{x}: %{z}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_xaxes(title=None, side="top")
    return _style(fig, "Confusion matrix at chosen threshold", 300)


def contribution_bars(coefs: pd.DataFrame, top: int = 10) -> go.Figure | None:
    if coefs.empty:
        return None
    d = coefs.head(top).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=d["coefficient"], y=d["label"], orientation="h",
        marker=dict(color=[ACCENT if c > 0 else GOOD for c in d["coefficient"]]),
        text=[f"{c:+.2f}" for c in d["coefficient"]], textposition="outside",
        hovertemplate="%{y}<br>coefficient %{x:+.3f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=GREY)
    fig.update_xaxes(title="Standardised coefficient (red = higher estimated probability)", gridcolor="#E4E7EB")
    fig.update_yaxes(title=None, automargin=True)
    return _style(fig, "What the model associates with departure", max(280, 34 * len(d) + 100))


def probability_histogram(preds: pd.DataFrame, threshold: float | None) -> go.Figure | None:
    if preds.empty:
        return None
    fig = go.Figure(go.Histogram(
        x=preds["probability"], nbinsx=20, marker=dict(color=PRIMARY, line=dict(color="white", width=1)),
        hovertemplate="Probability %{x}<br>%{y} students<extra></extra>",
    ))
    if threshold is not None:
        fig.add_vline(x=threshold, line_dash="dash", line_color=ACCENT,
                      annotation_text="Review threshold", annotation_position="top right")
    fig.update_xaxes(tickformat=".0%", title="Estimated probability", gridcolor="#E4E7EB")
    fig.update_yaxes(title="Students", gridcolor="#E4E7EB")
    fig.update_layout(bargap=0.05)
    return _style(fig, "Distribution of estimated probabilities", 340)


def feature_heatmap(preds: pd.DataFrame, features: list[str], labels: dict, limit: int = 20) -> go.Figure | None:
    """Show each risk feature's distance from the institute mean in standard deviations."""
    if preds.empty:
        return None
    short_labels = {
        "att_7": "Attend 7d",
        "att_14": "Attend 14d",
        "att_30": "Attend 30d",
        "att_trend": "Attend trend",
        "consec_absent": "Missed streak",
        "score_overall": "Test average",
        "score_recent3": "Recent tests",
        "score_slope": "Score trend",
        "tests_90d": "Tests 90d",
        "hw_30": "Homework",
        "days_since_activity": "Inactive days",
        "logins_30": "Logins 30d",
        "logins_change": "Login trend",
        "doubts_30": "Doubts 30d",
        "tenure_days": "Enrolled days",
        "fee_overdue_days": "Fee days late",
        "fee_overdue_amount": "Fee amount",
    }
    usable = [feature for feature in features if feature in preds.columns]
    if not usable:
        return None
    top = preds.head(limit).copy()
    z = {}
    full_names: dict[str, str] = {}
    for f in usable:
        col = pd.to_numeric(preds[f], errors="coerce")
        sd = col.std()
        display_name = short_labels.get(f, labels.get(f, f.replace("_", " ").title()))
        if not col.notna().any():
            z[display_name] = pd.Series(np.nan, index=top.index)
        elif pd.notna(sd) and sd > 0:
            z[display_name] = (
                (pd.to_numeric(top[f], errors="coerce") - col.mean()) / sd
            ).clip(-3, 3)
        else:
            z[display_name] = pd.Series(0.0, index=top.index)
        full_names[display_name] = labels.get(f, f.replace("_", " ").title())
    zf = pd.DataFrame(z)
    zf.insert(0, "student", top["name"].astype(str).values)
    long = zf.melt(id_vars="student", var_name="feature", value_name="z")
    fig = heatmap(
        long,
        "feature",
        "student",
        "z",
        title="Departure-risk signals vs institute average",
        colorscale="RdBu_r",
        zmin=-3,
        zmax=3,
        zmid=0,
        fmt="1f",
        colorbar_title="SD from average",
        annotate=False,
        height=int(min(650, max(260, 30 * len(top) + 130))),
    )
    if fig is None:
        return None
    x_order = list(dict.fromkeys(long["feature"].astype(str)))
    y_order = list(dict.fromkeys(long["student"].astype(str)))
    fig.data[0].customdata = [
        [full_names.get(feature, feature) for feature in x_order]
        for _ in y_order
    ]
    fig.data[0].hovertemplate = (
        "Student: %{y}<br>Signal: %{customdata}<br>"
        "%{z:+.2f} standard deviations from institute average"
        "<extra></extra>"
    )
    fig.update_xaxes(tickangle=0, side="top", tickfont=dict(size=10), automargin=True)
    fig.update_layout(margin=dict(l=10, r=10, t=65, b=20))
    return fig


def percentile_chart(frame: pd.DataFrame, limit: int = 40) -> go.Figure | None:
    """Percentile per student. Needs `name`, `mid`; optional `low`/`high` draw a range."""
    if frame.empty:
        return None
    d = frame.sort_values("mid", ascending=False).head(limit).iloc[::-1]
    fig = go.Figure()
    if {"low", "high"} <= set(d.columns) and d["low"].notna().any():
        fig.add_trace(go.Scatter(
            x=d["mid"], y=d["name"].astype(str), mode="markers", name="Estimate",
            marker=dict(size=9, color=PRIMARY),
            error_x=dict(type="data", symmetric=False, array=(d["high"] - d["mid"]).values,
                         arrayminus=(d["mid"] - d["low"]).values, color=GREY, thickness=2),
            hovertemplate="%{y}<br>Midpoint P%{x:.0f}<extra></extra>",
        ))
    else:
        fig.add_trace(go.Bar(
            x=d["mid"], y=d["name"].astype(str), orientation="h", name="Percentile",
            marker=dict(color=d["mid"], colorscale="RdYlGn", cmin=0, cmax=100),
            text=[f"P{v:.0f}" for v in d["mid"]], textposition="outside",
            hovertemplate="%{y}<br>Percentile %{x:.0f}<extra></extra>",
        ))
    fig.update_xaxes(range=[0, 108], title="Percentile", gridcolor="#E4E7EB")
    fig.update_yaxes(title=None, automargin=True)
    return _style(fig, "Percentile by student", int(min(900, max(260, 24 * len(d) + 110))))
