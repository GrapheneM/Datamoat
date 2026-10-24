from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Any

import pandas as pd
import streamlit as st
from sqlalchemy import select

from datamoat import auth, charts
from datamoat.attrition import AttritionConfig, run_attrition, FEATURE_LABELS
from datamoat.analytics import (
    compute_analytics,
    economic_summary,
    weak_and_strong_topics,
)
from datamoat.data_access import (
    clear_academic_data,
    dataset_counts,
    load_dataset,
    load_predictions,
    recent_audit,
    save_predictions,
    save_table,
    student_roster,
    validate_references,
)
from datamoat.db import check_database, session_scope
from datamoat.models import Institute, User
from datamoat.validation import TABLE_ORDER, TABLE_SPECS, parse_upload, template_csv, validate_upload


st.set_page_config(
    page_title="DataMoat | Institute Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

CORE_TABLES = ("students", "questions", "tests", "test_responses")
ROLE_PAGES = {
    "Owner": [
        "Overview",
        "Topic mastery",
        "Early warning",
        "Departure risk",
        "Percentile",
        "Parent reports",
        "Economics",
        "Data & imports",
        "User access",
        "Data validation",
        "Privacy",
        "Methodology",
    ],
    "Teacher": [
        "Overview",
        "Topic mastery",
        "Early warning",
        "Percentile",
        "Parent reports",
        "Data records",
        "Data validation",
        "Privacy",
        "Methodology",
    ],
    "Student": ["My progress", "Study plan", "Privacy", "Methodology"],
    "Parent": ["Child progress", "Reports", "Privacy", "Methodology"],
}

METHODOLOGY_TEXT = {
    "overview": (
        "## Overview measures\n"
        "- **Average test performance** is the unweighted mean of each student's average "
        "across their recorded tests. Students without test responses are excluded.\n"
        "- Counts describe the records visible under the selected institute, cohort, and cutoff."
    ),
    "mastery": (
        "## Topic mastery\n"
        "For each student and topic, DataMoat uses **(correct answers + 1) / "
        "(attempts + 2)**. This Beta(1,1) smoothing keeps a topic with one lucky answer "
        "from appearing fully mastered. Topics without attempts are not scored."
    ),
    "early_warning": (
        "## Early-warning support bands\n"
        "This is a transparent rule score, not a trained prediction. It adds points for "
        "observed low test performance, a recent drop, low attendance, and low homework "
        "completion. Missing data adds no points and is not treated as a negative result. "
        "The bands are prompts for a private supportive check-in, not labels or decisions."
    ),
    "percentile": (
        "## Percentiles and median\n"
        "**Current standing:** each student's per-test average is ranked within the selected "
        "cohort using average ranks; rank divided by cohort size is shown as a 0–100 percentile. "
        "This describes current relative standing, not a forecast.\n\n"
        "**Cohort median test average:** the median of the available student-level test averages; "
        "it is not weighted by how many responses each student has.\n\n"
        "**Forecast range:** shown only after at least five students have recorded final "
        "percentiles. A simple linear fit relates each student's observed test average to that "
        "historical outcome. Leave-one-out absolute errors set an 80th-percentile range, clipped "
        "to 0–100. It is an estimate, not a guarantee."
    ),
    "economics": (
        "## Fee totals\n"
        "**Fees due** sums instalments with a due date on or before the reporting cutoff. "
        "**Collected** counts those instalments with a paid date on or before the cutoff. "
        "**Past-due balance** includes instalments due by the cutoff with no payment recorded "
        "by then; the count is the number of such instalments. Partial payments, waivers, and "
        "other accounting adjustments are not represented, so this is a record summary—not a ledger."
    ),
    "departure": (
        "## Departure-risk model\n"
        "The model learns from this institute's recorded withdrawal dates. It builds dated "
        "snapshots from information available at that time, uses a chronological train/validation/"
        "test split, and calibrates probabilities on validation data when there are enough outcomes. "
        "It is marked validated only when minimum departure counts and evaluation checks are met; "
        "otherwise the app shows a rule-based support watchlist. Associations do not explain causes. "
        "Use results only to offer a private, supportive check-in—never for penalties or exclusion."
    ),
    "feature_chart": (
        "## Feature comparison chart\n"
        "Each cell is a feature's distance from the institute average in standard deviations. "
        "Red means the recorded feature value is above average and blue means below average; "
        "neither color means good, bad, or causal. Compare the underlying signal before deciding "
        "whether any follow-up is appropriate."
    ),
}


def _password_secret_is_set() -> bool:
    return bool(os.getenv("SESSION_SECRET", "").strip())


def _render_database_error(message: str) -> None:
    st.title("DataMoat")
    st.error(message)
    st.info(
        "Set DATABASE_URL to a PostgreSQL database, then run "
        "`python -m datamoat.migrate` once to create the tables."
    )
    st.stop()


def _render_auth() -> None:
    st.title("DataMoat")
    st.caption("Institute Intelligence Platform · Secure account access")

    sign_in, create_institute = st.tabs(["Sign in", "Create an institute"])

    with sign_in:
        login_col, help_col = st.columns([1.2, 0.8], gap="large")

        with login_col:
            st.subheader("Sign in to your institute")
            with st.form("sign_in_form"):
                institute_id = st.text_input("Institute ID")
                login_id = st.text_input("Staff ID, student ID, or email")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button(
                    "Sign in", type="primary", use_container_width=True
                )

            if submitted:
                success, result = auth.authenticate(institute_id, login_id, password)
                if success:
                    st.session_state["auth_user_id"] = result
                    st.session_state["view_role"] = None
                    st.session_state["preview_student_id"] = None
                    st.rerun()
                st.error(result)

        with help_col:
            st.subheader("Account access")
            st.write(
                "Use the account created for you by your institute. Your role is "
                "assigned to your account, so you do not need to select it here."
            )
            st.info(
                "Owners can create staff, student, and parent accounts after signing in."
            )
            st.caption(
                "For a forgotten password or an inactive account, contact your "
                "institute Owner."
            )

    with create_institute:
        st.subheader("Register an institute")
        st.write(
            "Create an institute and its first Owner account. Additional accounts "
            "can be created by an Owner after sign-in."
        )
        with st.form("create_institute_form"):
            name = st.text_input("Institute name")
            institute_id = st.text_input(
                "Institute ID", help="3–32 letters, numbers, or hyphens."
            )
            owner_name = st.text_input("Owner name")
            login_id = st.text_input("Owner email or account ID")
            password = st.text_input("Password", type="password")
            password_confirm = st.text_input("Confirm password", type="password")
            submitted = st.form_submit_button(
                "Create institute", type="primary", use_container_width=True
            )
        if submitted:
            if password != password_confirm:
                st.error("The passwords do not match.")
            else:
                success, result = auth.create_institute_owner(
                    institute_id, name, owner_name, login_id, password
                )
                if success:
                    st.session_state["auth_user_id"] = result
                    st.session_state["view_role"] = None
                    st.rerun()
                st.error(result)

    with st.expander("Why is there no demo login?"):
        st.write(
            "This version uses database-backed accounts and password verification. "
            "It does not include shared demo passwords or a public role selector."
        )

def _show(fig: Any) -> None:
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def _show_in(column: Any, fig: Any) -> None:
    if fig is not None:
        with column:
            _show(fig)


def _format_pct(value: Any) -> str:
    return f"{float(value):.0%}" if value is not None and pd.notna(value) else "—"


def _format_range(prediction: dict[str, float] | None) -> str:
    if not prediction:
        return "Not calibrated"
    return f"P{prediction['low']:.0f}–P{prediction['high']:.0f}"


def _metric_columns(metrics: list[tuple[str, str, str | None]]) -> None:
    cols = st.columns(len(metrics))
    for col, (label, value, help_text) in zip(cols, metrics):
        col.metric(label, value, help=help_text)


def _show_method_help(topic: str) -> None:
    with st.popover("ⓘ Calculation details", help="Open a plain-language explanation"):
        st.markdown(METHODOLOGY_TEXT[topic])
        if st.button("Read full methodology", key=f"methodology_link_{topic}"):
            st.session_state["page"] = "Methodology"
            st.rerun()


def _current_student(
    role: str,
    user: dict[str, Any],
    roster: pd.DataFrame,
    owner_preview_student_id: str | None,
) -> str | None:
    if role not in {"Student", "Parent"}:
        return None
    if user["role"] in {"Student", "Parent"}:
        return user.get("linked_student_id")
    if owner_preview_student_id:
        return owner_preview_student_id
    if roster.empty:
        return None
    return str(roster.iloc[0]["student_id"])


def _make_report(
    student: pd.Series,
    summary: dict[str, Any],
    weak: list[str],
    strong: list[str],
    prediction: dict[str, float] | None,
    institute_name: str,
    cutoff: date,
) -> str:
    trend = summary.get("score_recent")
    overall = summary.get("score_mean")
    if trend is None or overall is None or abs(trend - overall) < 0.03:
        trend_text = "steady"
    elif trend > overall:
        trend_text = "improving recently"
    else:
        trend_text = "a little lower recently; the teaching team can help"
    lines = [
        "Dear Parent,",
        "",
        f"Here is {student.get('name', 'the student')}'s progress summary through {cutoff:%d %b %Y}:",
        "",
    ]
    if strong:
        lines.append(f"Areas going well: {', '.join(strong)}.")
    if weak:
        lines.append(
            f"Topics to practise: {', '.join(weak)}. Short practice sessions and teacher support can help."
        )
    if summary.get("attendance_overall") is not None:
        lines.append(f"Attendance: {_format_pct(summary['attendance_overall'])}.")
    if overall is not None:
        lines.append(
            f"Average test performance: {_format_pct(overall)} ({trend_text})."
        )
    if prediction:
        lines.append(
            f"Estimated percentile range: P{prediction['low']:.0f}–P{prediction['high']:.0f}. "
            "This is an estimate based on historical institute outcomes, not a guarantee."
        )
    lines.extend(
        [
            "",
            "A steady study routine and encouragement are more helpful than comparing marks with other students.",
            "For questions, please contact the institute.",
            f"- {institute_name}",
        ]
    )
    return "\n".join(lines)


def _render_overview(
    data: dict[str, pd.DataFrame],
    analytics: dict[str, Any],
    cohort: str | None,
    user: dict[str, Any],
    cutoff: date,
    role: str,
) -> None:
    st.title("Institute overview")
    _show_method_help("overview")
    st.caption(
        f"{user['institute_name']} · {cohort or 'All cohorts'} · Data through {cutoff:%d %b %Y}"
    )
    counts = dataset_counts(user["institute_id"], role)
    ready = all(counts[name] > 0 for name in CORE_TABLES)
    if not ready:
        missing = [TABLE_SPECS[name]["title"] for name in CORE_TABLES if counts[name] == 0]
        st.warning("Analysis is waiting for: " + ", ".join(missing) + ".")
        if user["role"] == "Owner":
            st.info("Open Data & imports to upload the source files in the suggested order.")
        else:
            st.info("Ask an institute Owner to complete the required data imports.")
        status = pd.DataFrame(
            [
                {
                    "Dataset": TABLE_SPECS[name]["title"],
                    "Rows": counts[name],
                    "Required": "Yes" if name in CORE_TABLES else "Optional",
                    "Unlocks": TABLE_SPECS[name]["unlocks"],
                }
                for name in TABLE_ORDER
                if role == "Owner" or name != "fees"
            ]
        )
        st.dataframe(status, hide_index=True, use_container_width=True)
        return

    students = analytics["students"]
    summaries = analytics["summaries"]
    risk = analytics["risk"]
    score_values = [item["score_mean"] for item in summaries.values() if item["score_mean"] is not None]
    _metric_columns(
        [
            ("Students in view", f"{len(students):,}", "Students in the selected cohort with records visible to your role."),
            ("Questions tagged", f"{len(data['questions']):,}", "Question records used to group responses into topics."),
            ("Responses in view", f"{len(analytics['responses']):,}", "Scored student-question responses through the reporting cutoff."),
            ("Average test performance", _format_pct(np_mean(score_values)), "Unweighted average of each student's available test averages. Open Calculation details for the method."),
        ]
    )
    if not risk.empty:
        priority_count = int(
            risk["band"].isin(["High support priority", "Needs a check-in"]).sum()
        )
        st.metric("Students for a supportive check-in", priority_count)

    scores = analytics["responses"]
    if not scores.empty:
        tests = (
            scores.groupby(["test_id", "_date"], as_index=False)
            .agg(accuracy=("is_correct", "mean"))
            .sort_values("_date")
        )
        if not tests.empty:
            _show(charts.trend_chart(tests))
    if data["attendance"].empty:
        st.caption("Attendance signals will appear after the optional attendance file is imported.")
    elif role in {"Owner", "Teacher"}:
        att_chart = charts.attendance_heatmap(data["attendance"], analytics["students"])
        if att_chart is not None:
            st.subheader("Attendance by batch and week")
            _show(att_chart)


def np_mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _render_mastery(
    analytics: dict[str, Any], role: str, focus_student_id: str | None = None
) -> None:
    st.title("Topic mastery")
    _show_method_help("mastery")
    mastery = analytics["mastery"]
    if mastery.empty:
        st.info("No question-level responses are available for this selection yet.")
        return
    if role in {"Student", "Parent"} and focus_student_id:
        mastery = mastery[mastery["student_id"].astype(str) == focus_student_id]
    grouped = (
        mastery.groupby(["batch_id", "topic"], as_index=False)
        .agg(mastery=("mastery", "mean"), attempts=("attempts", "sum"))
        .sort_values(["batch_id", "topic"])
    )
    st.caption(
        "Mastery uses a small prior to avoid over-reading a few answers. Topics without attempts are not scored."
    )
    if role not in {"Student", "Parent"}:
        topic_options = sorted(grouped["topic"].dropna().astype(str).unique().tolist())
        if topic_options:
            selected_topic = st.selectbox("Compare batches by topic", topic_options)
            _show(charts.batch_topic_bars(grouped, selected_topic))
    if role not in {"Student", "Parent"}:
        heat = charts.mastery_heatmap(mastery)
        if heat is not None:
            _show(heat)
    pivot = grouped.pivot(index="batch_id", columns="topic", values="mastery").round(2)
    st.dataframe(pivot, use_container_width=True)
    details = mastery.rename(
        columns={
            "student_id": "Student ID",
            "batch_id": "Batch",
            "subject": "Subject",
            "topic": "Topic",
            "attempts": "Attempts",
            "correct": "Correct",
            "accuracy": "Raw accuracy",
            "mastery": "Smoothed mastery",
        }
    )
    if role in {"Owner", "Teacher"}:
        with st.expander("Student-level topic details"):
            st.dataframe(details, hide_index=True, use_container_width=True)


def _render_risk(analytics: dict[str, Any], role: str) -> None:
    st.title("Early warning and supportive follow-up")
    _show_method_help("early_warning")
    st.caption(
        "Signals use observed assessment, attendance, and homework data only. Missing data is not treated as a risk signal."
    )
    risk = analytics["risk"]
    if risk.empty:
        st.info("Load assessment data to calculate support indicators.")
        return
    if role in {"Student", "Parent"}:
        st.info(
            "This view is for private support planning. It does not show peer comparisons or public rankings."
        )
    if role in {"Owner", "Teacher"}:
        left, right = st.columns(2)
        band_chart = charts.risk_band_heatmap(risk)
        signal_chart = charts.student_signal_heatmap(analytics["summaries"])
        if band_chart is not None:
            _show_in(left, band_chart)
        if signal_chart is not None:
            _show_in(right, signal_chart)
    visible = risk.drop(columns=["student_id"]) if role in {"Student", "Parent"} else risk
    st.dataframe(visible, hide_index=True, use_container_width=True)
    if not analytics["responses"].empty and analytics["mastery"].empty:
        st.caption("Topic-level support details need question tags and response records.")


def _fmt(value: Any, pct: bool = True) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    return f"{value:.1%}" if pct else f"{value:.3f}"


@st.cache_data(ttl=600, show_spinner=False)
def _cached_attrition(institute_id: str, horizon: int, capacity: float, use_fees: bool,
                      fingerprint: tuple) -> Any:
    """Train and score automatically; cached until the data changes or ten minutes pass."""
    full = load_dataset(institute_id, "Owner", None)
    return run_attrition(full, AttritionConfig(horizon_days=horizon, max_alert_rate=capacity, use_fees=use_fees))


def _render_departure_risk(user: dict[str, Any]) -> None:
    st.title("Departure risk")
    _show_method_help("departure")
    st.caption(
        "Estimated probability that a student leaves within the chosen window, learned from this "
        "institute's own recorded withdrawals. It supports a human check-in; it is not a verdict."
    )
    with st.expander("Model settings", expanded=False):
        c1, c2, c3 = st.columns(3)
        horizon = c1.number_input("Prediction window (days)", 14, 180, 60, step=7)
        capacity = c2.slider("Share of students you can follow up", 5, 50, 20, step=5,
                             help="The review threshold is chosen so no more than this share is flagged.") / 100
        use_fees = c3.checkbox("Use fee timing as a feature", value=True,
                               help="Fee features only describe payment timing, not personal engagement.")
    counts = dataset_counts(user["institute_id"], "Owner")
    with st.spinner("Analysing departure risk..."):
        res = _cached_attrition(user["institute_id"], int(horizon), capacity, use_fees,
                                tuple(sorted(counts.items())))
    if st.button("Recalculate now", help="Results refresh automatically when data changes."):
        st.cache_data.clear()
        st.rerun()

    if res.status != "ok":
        st.info(res.message)
        summary = res.snapshot_summary
        if summary:
            seen = summary.get("positive_snapshots", 0)
            st.metric("Recorded departures", summary.get("recorded_departures", 0),
                      help="A trained model needs roughly 30 or more recorded withdrawals with dates.")
            st.progress(min(1.0, summary.get("recorded_departures", 0) / 30))
        st.subheader("Support watchlist")
        st.caption("Rule-based and not a trained prediction: it flags low attendance, falling scores and "
                   "low homework. It appears here until enough withdrawal history exists to train a model.")
        wl = res.watchlist
        if wl.empty:
            st.write("No active students with enough history to show.")
        else:
            band = charts.risk_band_heatmap(wl)
            _show(band)
            st.dataframe(
                wl.rename(columns={"name": "Student", "batch_id": "Batch", "rule_score": "Support score",
                                   "band": "Band", "att_30": "Attendance 30d", "score_overall": "Test average",
                                   "hw_30": "Homework 30d", "consec_absent": "Missed in a row",
                                   "days_since_activity": "Days since activity"}).drop(columns="student_id").round(2),
                hide_index=True, use_container_width=True)
        return

    if res.validated:
        st.success(res.message + " Model " + str(res.model_version) + ".")
    else:
        st.error(res.message + " Treat the numbers below as exploratory.")
    for w in res.warnings:
        st.caption(w)

    summary = res.snapshot_summary
    m, b = res.metrics, res.baseline_metrics
    _metric_columns([
        ("Test PR-AUC", _fmt(m["pr_auc"], False), f"Base departure rate in test period: {_fmt(m['base_rate'])}"),
        ("Test ROC-AUC", _fmt(m["roc_auc"], False), None),
        ("Precision at threshold", _fmt(m["precision"]), "Of flagged students, share who actually left."),
        ("Recall at threshold", _fmt(m["recall"]), "Of students who left, share that were flagged."),
    ])

    tab_students, tab_eval, tab_model, tab_store = st.tabs(
        ["Students to review", "Evaluation", "Model", "Stored predictions"]
    )
    with tab_students:
        preds = res.predictions
        if preds.empty:
            st.info("No active students to score.")
        else:
            st.caption(f"Review threshold {res.threshold:.1%} · window {res.horizon_days} days · "
                       f"data through {res.as_of:%d %b %Y} · model {res.model_version}")
            hist = charts.probability_histogram(preds, res.threshold)
            if hist is not None:
                _show(hist)
            heat = charts.feature_heatmap(preds, res.features, FEATURE_LABELS, limit=12)
            if heat is not None:
                _show_method_help("feature_chart")
                st.caption(
                    "Rows are listed by estimated departure probability. Red and blue show "
                    "above- or below-average recorded values, not positive or negative causes."
                )
                _show(heat)
            table = preds[["name", "batch_id", "probability", "flag", "top_factors", "rule_score"]].rename(columns={
                "name": "Student", "batch_id": "Batch", "probability": "Estimated probability",
                "flag": "Status", "top_factors": "Features contributing (not causes)", "rule_score": "Existing support score"})
            table["Estimated probability"] = (table["Estimated probability"] * 100).round(1)
            st.dataframe(
                table.rename(columns={"Estimated probability": "Estimated probability (%)"}),
                hide_index=True, use_container_width=True,
            )
            st.caption("Suggested action for flagged students: a private, supportive check-in about "
                       "schedule, workload or access. Do not use this list for any penalty or exclusion.")
    with tab_eval:
        st.write(f"Evaluated on the newest period: {summary['test_rows']:,} snapshots, "
                 f"{summary['test_positives']} departures. Training used {summary['train_rows']:,} snapshots "
                 f"({summary['train_positives']} departures) up to {summary['train_end']}.")
        rows = [
            {"Model": "Logistic regression (calibrated)", "PR-AUC": m["pr_auc"], "ROC-AUC": m["roc_auc"],
             "Brier": m["brier"], "Precision": m["precision"], "Recall": m["recall"]},
            {"Model": "Existing rule-based score", "PR-AUC": b["pr_auc"], "ROC-AUC": b["roc_auc"],
             "Brier": b["brier"], "Precision": b["precision"], "Recall": b["recall"]},
        ]
        if res.gb_metrics:
            g = res.gb_metrics
            rows.append({"Model": "Gradient boosting (comparison only)", "PR-AUC": g["pr_auc"], "ROC-AUC": g["roc_auc"],
                         "Brier": g["brier"], "Precision": g["precision"], "Recall": g["recall"]})
        st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, use_container_width=True)
        left, right = st.columns(2)
        cal = charts.calibration_chart(res.calibration_bins)
        if cal is not None:
            _show_in(left, cal)
        _show_in(right, charts.confusion_heatmap(m["confusion"]))
        st.caption(f"Brier score before calibration: {_fmt(m['brier_uncalibrated'], False)}; after: {_fmt(m['brier'], False)}. "
                   "Calibration is fitted on the validation period only.")
    with tab_model:
        bars = charts.contribution_bars(res.coefficients)
        if bars is not None:
            _show(bars)
        st.caption("Coefficients show statistical association in past data. They do not show why a student leaves.")
        st.write("Snapshots: " + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in summary.items()
                                           if k in {"students", "recorded_departures", "snapshots", "censored_snapshots"}))
    with tab_store:
        if st.button("Save these predictions", disabled=res.predictions.empty):
            ok, msg = save_predictions(user["id"], user["institute_id"], res.predictions,
                                       res.model_version or "", res.threshold, res.validated)
            (st.success if ok else st.error)(msg)
        history = load_predictions(user["institute_id"])
        if history.empty:
            st.caption("No predictions stored yet.")
        else:
            st.dataframe(history, hide_index=True, use_container_width=True)


def _current_standing(analytics: dict[str, Any]) -> pd.DataFrame:
    """Percentile rank of each student's observed test average within the selected cohort."""
    rows = [
        {"student_id": sid, "name": item["name"], "batch_id": item["batch_id"],
         "score": item["score_mean"], "tests": item["test_count"]}
        for sid, item in analytics["summaries"].items() if item.get("score_mean") is not None
    ]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["mid"] = (frame["score"].rank(pct=True, method="average") * 100).round(0)
    return frame.sort_values("mid", ascending=False)


def _render_predictions(analytics: dict[str, Any]) -> None:
    st.title("Percentile")
    _show_method_help("percentile")
    standing = _current_standing(analytics)
    count = analytics["calibration_count"]
    predictions = analytics["predictions"]
    if standing.empty:
        st.info("Percentiles need test responses. Import Tests and Test responses first.")
        return

    if count >= 5 and predictions:
        radius = analytics["calibration_radius"]
        st.caption(
            "Expected final percentile, calibrated on this institute's past final-percentile outcomes. "
            "The range is an estimate, not a promise."
        )
        _metric_columns([
            ("Past outcomes used", f"{count}", "Students with both a test average and recorded final percentile."),
            ("Typical error", f"±{radius:.0f} points", "Leave-one-out calibration radius; wide for small cohorts."),
            ("Students in view", f"{len(standing)}", "Students with a test average in this cohort."),
        ])
        rows = []
        for sid, pred in predictions.items():
            info = analytics["summaries"].get(sid, {})
            rows.append({"name": info.get("name", sid), "batch_id": info.get("batch_id", ""),
                         "low": pred["low"], "mid": pred["mid"], "high": pred["high"],
                         "score": info.get("score_mean")})
        view = pd.DataFrame(rows)
        _show(charts.percentile_chart(view))
        table = view.assign(**{"Expected range": view.apply(lambda r: f"P{r['low']:.0f} to P{r['high']:.0f}", axis=1),
                               "Test average": view["score"].map(_format_pct)})
        st.dataframe(table.rename(columns={"name": "Student", "batch_id": "Batch", "mid": "Midpoint"})
                     [["Student", "Batch", "Expected range", "Midpoint", "Test average"]]
                     .sort_values("Midpoint", ascending=False),
                     hide_index=True, use_container_width=True)
        return

    st.caption(
        "Current standing: each student's rank by observed test average within this cohort. "
        "This describes where students are now, not a forecast of the final result."
    )
    st.info(
        f"A forecast range needs at least 5 students with a recorded final_percentile; {count} qualify so far. "
        "Add past outcomes to the Students file to unlock it."
    )
    _metric_columns([
        ("Students ranked", f"{len(standing)}", "Students with at least one scored test."),
        ("Cohort median test average", _format_pct(standing["score"].median()), "Median of student-level average test scores; not weighted by number of responses."),
        ("Bottom quarter (below P25)", f"{int((standing['mid'] <= 25).sum())}", "Students who may benefit from extra support."),
    ])
    _show(charts.percentile_chart(standing))
    st.dataframe(
        standing.assign(**{"Test average": standing["score"].map(_format_pct)})
        .rename(columns={"name": "Student", "batch_id": "Batch", "mid": "Percentile", "tests": "Tests taken"})
        [["Student", "Batch", "Percentile", "Test average", "Tests taken"]],
        hide_index=True, use_container_width=True,
    )


def _render_parent_reports(
    analytics: dict[str, Any],
    roster: pd.DataFrame,
    role: str,
    user: dict[str, Any],
    focus_student_id: str | None,
    cutoff: date,
) -> None:
    st.title("Parent progress reports")
    if roster.empty:
        st.info("No student records are available yet.")
        return
    if role in {"Parent", "Student"}:
        student_id = focus_student_id
    else:
        options = roster["student_id"].astype(str).tolist()
        student_id = st.selectbox(
            "Select a student",
            options,
            format_func=lambda sid: (
                f"{roster.loc[roster['student_id'].astype(str) == sid, 'name'].iloc[0]} · {sid}"
            ),
        )
    if not student_id:
        st.info("This account is not linked to a student. Ask an institute Owner to update the account.")
        return
    selected = roster[roster["student_id"].astype(str) == str(student_id)]
    if selected.empty:
        st.info("This account is not linked to a student in this institute.")
        return
    student = selected.iloc[0]
    if not bool(student.get("parent_contact_consent", False)):
        st.warning(
            "A parent report is unavailable because verified parent-contact consent is not recorded for this student."
        )
        return
    summary = analytics["summaries"].get(str(student_id))
    if not summary:
        st.info("No academic results are available for this student yet.")
        return
    weak, strong = weak_and_strong_topics(analytics["mastery"], str(student_id))
    report = _make_report(
        student,
        summary,
        weak,
        strong,
        analytics["predictions"].get(str(student_id)),
        user["institute_name"],
        cutoff,
    )
    st.text_area("Report preview", report, height=260, disabled=True)
    st.download_button(
        "Download report",
        data=report.encode("utf-8"),
        file_name=f"progress-report-{student_id}.txt",
        mime="text/plain; charset=utf-8",
    )


def _render_student_portal(
    analytics: dict[str, Any], student_id: str | None, user: dict[str, Any], show_plan: bool
) -> None:
    title = "Study plan" if show_plan else "My progress"
    st.title(title)
    if not student_id:
        st.info("This account is not linked to a student record. Ask the institute Owner for help.")
        return
    summary = analytics["summaries"].get(str(student_id))
    if not summary:
        st.info("Your progress will appear after your test data is imported.")
        return
    st.subheader(summary["name"])
    _metric_columns(
        [
            ("Average test performance", _format_pct(summary.get("score_mean")), None),
            ("Recent test performance", _format_pct(summary.get("score_recent")), None),
            ("Attendance", _format_pct(summary.get("attendance_overall")), None),
        ]
    )
    weak, strong = weak_and_strong_topics(analytics["mastery"], str(student_id))
    if strong:
        st.success("Topics going well: " + ", ".join(strong))
    if weak:
        st.info("Topics to practise next: " + ", ".join(weak))
    if show_plan:
        plan = [
            f"Revise {topic}: spend two short sessions reviewing examples and solving practice questions."
            for topic in weak
        ]
        if strong:
            plan.append(f"Keep {strong[0]} fresh with a short mixed-practice set.")
        plan.append("Review mistakes from the latest test and write down what to try differently.")
        for number, step in enumerate(plan, start=1):
            st.write(f"{number}. {step}")
    st.caption("This private view shows only your own records, with no peer ranking.")


def _render_economics(
    data: dict[str, pd.DataFrame],
    analytics: dict[str, Any],
    cutoff: date,
) -> None:
    st.title("Fee collection overview")
    _show_method_help("economics")
    st.caption("Visible only to institute Owners. This is a record summary, not an accounting ledger.")
    if data["fees"].empty:
        st.info("No fee records are available. Add the optional Fee instalments file in Data & imports.")
        return
    summary, overdue = economic_summary(data["fees"], analytics["students"], cutoff)
    _metric_columns(
        [
            ("Fees due through cutoff", f"₹{summary['scheduled']:,.0f}", "Sum of instalments due on or before the reporting cutoff."),
            ("Collected", f"₹{summary['collected']:,.0f}", "Due instalments with a recorded paid date on or before the cutoff."),
            ("Past-due balance", f"₹{summary['overdue']:,.0f}", "Due instalments without a payment recorded by the cutoff."),
            ("Overdue instalments", f"{int(summary['overdue_count']):,}", "Number of unpaid instalments due by the cutoff."),
        ]
    )
    if not overdue.empty:
        st.subheader("Past-due instalments")
        st.dataframe(overdue, hide_index=True, use_container_width=True)
    st.caption("Use private, supportive follow-up; do not use fee status as a public student ranking.")


def _render_data_page(
    data: dict[str, pd.DataFrame],
    user: dict[str, Any],
    actual_role: str,
    effective_role: str,
) -> None:
    st.title("Data records and imports")
    counts = dataset_counts(user["institute_id"], effective_role)
    status = []
    for name in TABLE_ORDER:
        if name == "fees" and effective_role != "Owner":
            continue
        status.append(
            {
                "Dataset": TABLE_SPECS[name]["title"],
                "Rows": counts[name],
                "Status": "Loaded" if counts[name] else "Not loaded",
                "Unlocks": TABLE_SPECS[name]["unlocks"],
            }
        )
    st.dataframe(pd.DataFrame(status), hide_index=True, use_container_width=True)

    st.subheader("Review records")
    available_tables = [name for name in TABLE_ORDER if name != "fees" or effective_role == "Owner"]
    selected_table = st.selectbox(
        "Dataset",
        available_tables,
        format_func=lambda key: TABLE_SPECS[key]["title"],
    )
    table_data = data.get(selected_table, pd.DataFrame())
    if table_data.empty:
        st.info("No rows are stored for this dataset yet.")
    else:
        st.dataframe(table_data, hide_index=True, use_container_width=True)

    if actual_role != "Owner" or effective_role != "Owner":
        st.info("Uploads and data changes are restricted to the institute Owner.")
        return

    st.subheader("CSV / Excel imports")
    st.write(
        "Import students, questions, and tests first, then test responses. Optional attendance, "
        "fee, and engagement files unlock additional summaries. Passwords must never be uploaded."
    )
    template_cols = st.columns(4)
    for index, name in enumerate(TABLE_ORDER):
        template_cols[index % 4].download_button(
            f"{TABLE_SPECS[name]['title']} template",
            data=template_csv(name).encode("utf-8"),
            file_name=f"{name}.csv",
            mime="text/csv",
            key=f"template_{name}",
        )

    target = st.selectbox(
        "Choose a dataset to import",
        TABLE_ORDER,
        format_func=lambda key: TABLE_SPECS[key]["title"],
        key="import_table",
    )
    uploaded = st.file_uploader(
        "Choose a CSV or .xlsx file",
        type=["csv", "xlsx"],
        key=f"upload_{target}",
    )
    if uploaded is not None:
        try:
            raw = parse_upload(uploaded.name, uploaded.getvalue())
            cleaned, errors, warnings = validate_upload(target, raw)
            for warning in warnings:
                st.warning(warning)
            if errors:
                for issue in errors:
                    st.error(issue)
            else:
                ref_errors = validate_references(user["institute_id"], target, cleaned)
                if ref_errors:
                    for issue in ref_errors:
                        st.error(issue)
                else:
                    st.success(f"{len(cleaned):,} rows passed validation.")
                    st.dataframe(cleaned.head(12), hide_index=True, use_container_width=True)
                    save_mode = st.radio(
                        "Save behavior",
                        ["Add or update matching rows", "Replace this table"],
                        horizontal=True,
                        key=f"save_mode_{target}",
                        help="Replacing can be blocked when another dataset depends on current records.",
                    )
                    if st.button("Save validated rows", type="primary", key=f"save_{target}"):
                        mode = "append" if save_mode.startswith("Add") else "replace"
                        success, message = save_table(
                            user["id"],
                            user["institute_id"],
                            target,
                            cleaned,
                            mode,
                        )
                        if success:
                            st.cache_data.clear()
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
        except (ValueError, pd.errors.ParserError, OSError) as exc:
            st.error(str(exc))
        except Exception:
            st.error("The file could not be processed. Check the template and try again.")

    st.subheader("Clear academic data")
    st.warning(
        "This permanently deletes institute student, question, test, response, attendance, fee, and engagement records. "
        "Accounts and audit history are kept."
    )
    with st.form("clear_academic_data_form"):
        confirmation = st.text_input(
            f"Type {user['institute_id']} to confirm"
        )
        clear_button = st.form_submit_button("Clear all academic data")
    if clear_button:
        if confirmation.strip().upper() != user["institute_id"]:
            st.error("The Institute ID did not match. No data was changed.")
        else:
            success, message = clear_academic_data(user["id"], user["institute_id"])
            (st.success if success else st.error)(message)
            if success:
                st.cache_data.clear()
                st.rerun()


def _render_accounts(user: dict[str, Any]) -> None:
    st.title("Institute user access")
    st.caption(
        "Accounts are created by an Owner, and each account has one institute and one assigned role."
    )
    roster = student_roster(user["institute_id"])
    roles = ["Owner", "Teacher", "Student", "Parent"]
    with st.expander("Create an account", expanded=True):
        with st.form("create_account_form"):
            role = st.selectbox("Role", roles)
            display_name = st.text_input("Name")
            login_id = st.text_input("Email or account ID")
            linked_student_id = None
            if role in {"Student", "Parent"}:
                if roster.empty:
                    st.info("Import the Students dataset before creating student or parent accounts.")
                else:
                    linked_student_id = st.selectbox(
                        "Linked student",
                        roster["student_id"].astype(str).tolist(),
                        format_func=lambda sid: (
                            f"{roster.loc[roster['student_id'].astype(str) == sid, 'name'].iloc[0]} · {sid}"
                        ),
                    )
            password = st.text_input("Temporary password (12 or more characters)", type="password")
            submitted = st.form_submit_button("Create account", type="primary")
        if submitted:
            if role in {"Student", "Parent"} and roster.empty:
                st.error("Import student records first.")
            else:
                success, message = auth.create_account(
                    user["id"],
                    role,
                    display_name,
                    login_id,
                    password,
                    linked_student_id,
                )
                if success:
                    st.success("Account created. Share the temporary password privately.")
                    st.rerun()
                else:
                    st.error(message)

    with st.expander("Bulk create student accounts from CSV", expanded=False):
        st.write(
            "Create many Student login accounts in one operation. This uses the "
            "existing Students dataset as the source of truth: every uploaded "
            "student_id must already exist in this institute. The student's name "
            "and account link are taken from that roster, not typed again."
        )
        existing_student_account_ids = {
            str(item["linked_student_id"])
            for item in auth.list_accounts(user["institute_id"])
            if item["role"] == "Student" and item["linked_student_id"]
        }
        if roster.empty:
            st.warning("Import the Students dataset before creating student accounts.")
        else:
            pending_roster = roster[
                ~roster["student_id"].astype(str).isin(existing_student_account_ids)
            ].copy()
            template = pending_roster[["student_id"]].copy()
            # The canonical student ID is the default login ID; an Owner can replace
            # it with a unique email or another login ID in the CSV.
            template["login_id"] = template["student_id"].astype(str)
            st.download_button(
                "Download CSV template for students without accounts",
                data=template.to_csv(index=False).encode("utf-8-sig"),
                file_name="bulk_student_accounts.csv",
                mime="text/csv",
                disabled=template.empty,
                help="One row per student who does not yet have a Student account.",
            )
            st.caption(
                f"{len(roster):,} student records in the roster · "
                f"{len(existing_student_account_ids):,} already have Student accounts · "
                f"{len(pending_roster):,} available for bulk creation."
            )
            account_file = st.file_uploader(
                "Upload the completed bulk account CSV",
                type=["csv"],
                key="bulk_student_accounts_csv",
                help="Required column: student_id. Optional column: login_id. Blank login IDs default to student_id.",
            )
            if account_file is not None:
                try:
                    account_frame = pd.read_csv(
                        account_file, dtype=str, keep_default_na=False
                    )
                    account_frame.columns = [
                        str(column).strip().casefold().replace("-", "_").replace(" ", "_")
                        for column in account_frame.columns
                    ]
                    if account_frame.columns.duplicated().any():
                        st.error("The CSV contains duplicate column names.")
                    elif "student_id" not in account_frame.columns:
                        st.error("The CSV must contain a student_id column.")
                    elif set(account_frame.columns) - {"student_id", "login_id"}:
                        st.error(
                            "Use only these columns in the bulk account CSV: "
                            "student_id, login_id. Extra columns found: "
                            + ", ".join(sorted(set(account_frame.columns) - {"student_id", "login_id"}))
                        )
                    elif account_frame.empty:
                        st.error("The CSV has no student rows.")
                    else:
                        if "login_id" not in account_frame.columns:
                            account_frame["login_id"] = ""
                        account_frame = account_frame[["student_id", "login_id"]]
                        account_frame = account_frame.apply(
                            lambda column: column.map(lambda value: str(value).strip())
                        )
                        st.write(f"Rows to process: **{len(account_frame):,}**")
                        st.caption(
                            "No account is created until you press the button. The operation "
                            "is all-or-nothing: if a row is invalid, none of the accounts are created."
                        )
                        problems = auth.preview_student_accounts_bulk(
                            user["id"], account_frame.to_dict(orient="records")
                        )
                        if problems:
                            st.error(f"{len(problems)} problem(s) found. Nothing has been created.")
                            st.dataframe(
                                pd.DataFrame({"Problem": problems[:200]}),
                                hide_index=True, use_container_width=True,
                            )
                        elif st.button(
                            "Validate and create student accounts",
                            type="primary",
                            key="create_bulk_student_accounts",
                        ):
                            success, result = auth.create_student_accounts_bulk(
                                user["id"], account_frame.to_dict(orient="records")
                            )
                            if success:
                                credential_frame = pd.DataFrame(result)
                                st.success(
                                    f"Created {len(credential_frame):,} student account(s)."
                                )
                                st.warning(
                                    "The downloaded file contains temporary passwords. "
                                    "Each student must change this password at first sign-in. "
                                    "Download it now, distribute each student's credentials "
                                    "privately, and delete the file afterward. Do not email or "
                                    "publish the full credentials CSV."
                                )
                                st.download_button(
                                    "Download temporary credentials (keep private)",
                                    data=credential_frame.to_csv(index=False).encode("utf-8-sig"),
                                    file_name="student_account_credentials_private.csv",
                                    mime="text/csv",
                                    key="download_bulk_student_credentials",
                                )
                            else:
                                st.error(str(result))
                except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
                    st.error(f"Could not read this CSV: {exc}")

    accounts = auth.list_accounts(user["institute_id"])
    if accounts:
        st.subheader("Accounts")
        display = pd.DataFrame(
            [
                {
                    "Name": item["display_name"],
                    "Login ID": item["login_id"],
                    "Role": item["role"],
                    "Linked student": item["linked_student_id"] or "",
                    "Status": "Active" if item["is_active"] else "Deactivated",
                    "Created": item["created_at"],
                }
                for item in accounts
            ]
        )
        st.dataframe(display, hide_index=True, use_container_width=True)
        active_accounts = [item for item in accounts if item["is_active"]]
        if active_accounts:
            with st.expander("Reset an account password"):
                labels = {
                    f"{item['display_name']} · {item['login_id']} · {item['role']}": item["id"]
                    for item in active_accounts
                }
                with st.form("reset_password_form"):
                    label = st.selectbox("Account", list(labels))
                    new_password = st.text_input("New temporary password", type="password")
                    reset = st.form_submit_button("Reset password")
                if reset:
                    success, message = auth.reset_account_password(
                        user["id"], labels[label], new_password
                    )
                    (st.success if success else st.error)(message)
        manageable = [
            item for item in accounts
            if item["id"] != user["id"] and item["is_active"]
        ]
        if manageable:
            with st.expander("Deactivate an account"):
                labels = {
                    f"{item['display_name']} · {item['login_id']} · {item['role']}": item["id"]
                    for item in manageable
                }
                with st.form("deactivate_account_form"):
                    label = st.selectbox("Account to deactivate", list(labels))
                    deactivate = st.form_submit_button("Deactivate account")
                if deactivate:
                    success, message = auth.deactivate_account(user["id"], labels[label])
                    (st.success if success else st.error)(message)
                    if success:
                        st.rerun()
        inactive_accounts = [item for item in accounts if not item["is_active"]]
        if inactive_accounts:
            with st.expander("Reactivate an account"):
                labels = {
                    f"{item['display_name']} · {item['login_id']} · {item['role']}": item["id"]
                    for item in inactive_accounts
                }
                with st.form("reactivate_account_form"):
                    label = st.selectbox("Account to reactivate", list(labels))
                    reactivate = st.form_submit_button("Reactivate account")
                if reactivate:
                    success, message = auth.set_account_active(
                        user["id"], labels[label], True
                    )
                    (st.success if success else st.error)(message)
                    if success:
                        st.rerun()


def _render_validation(
    data: dict[str, pd.DataFrame], counts: dict[str, int], role: str, institute_id: str
) -> None:
    st.title("Data validation")
    st.caption("Database constraints reject invalid links and values before they can affect reports.")
    rows = []
    for name in TABLE_ORDER:
        if name == "fees" and role != "Owner":
            continue
        loaded = counts[name] > 0
        rows.append(
            {
                "Dataset": TABLE_SPECS[name]["title"],
                "Rows": counts[name],
                "State": "Available" if loaded else ("Required" if name in CORE_TABLES else "Optional"),
                "Used for": TABLE_SPECS[name]["unlocks"],
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    missing_core = [TABLE_SPECS[name]["title"] for name in CORE_TABLES if counts[name] == 0]
    if missing_core:
        st.warning("Core analysis is incomplete: " + ", ".join(missing_core) + ".")
    else:
        st.success("All four core datasets contain records.")

    st.subheader("Current integrity checks")
    checks = []
    for name in CORE_TABLES:
        checks.append(
            {
                "Check": f"{TABLE_SPECS[name]['title']} table",
                "Result": f"{counts[name]:,} row(s) stored",
            }
        )
    checks.extend(
        [
            {"Check": "Institute isolation", "Result": "Every record is stored with an institute ID."},
            {"Check": "Duplicate IDs", "Result": "Primary and unique keys prevent duplicate records."},
            {"Check": "Related records", "Result": "Foreign keys prevent orphan responses and student records."},
        ]
    )
    st.dataframe(pd.DataFrame(checks), hide_index=True, use_container_width=True)

    if role == "Owner":
        audit = recent_audit(institute_id)
        st.subheader("Recent data and access activity")
        if audit.empty:
            st.caption("No audit events have been recorded yet.")
        else:
            st.dataframe(audit, hide_index=True, use_container_width=True)


def _render_privacy(
    data: dict[str, pd.DataFrame], user: dict[str, Any], actual_role: str
) -> None:
    st.title("Privacy and access")
    st.write(
        "DataMoat limits records by institute and account role. It is a support tool, not a public ranking system."
    )
    st.subheader("Data controls")
    st.markdown(
        "- Student and parent views are restricted to the account's linked student.\n"
        "- Fee records are only loaded for an Owner account.\n"
        "- Parent reports require recorded parent-contact consent.\n"
        "- Uploaded files are validated and saved to PostgreSQL; uploaded passwords are not supported.\n"
        "- Audit events record account and data changes without storing password values."
    )
    st.caption(
        "These controls help organize access; they do not certify compliance with any law or regulation."
    )
    if actual_role == "Owner":
        st.subheader("Parent-contact consent")
        students = data["students"]
        if students.empty:
            st.info("Import student records to review consent status.")
        else:
            consent = students[["student_id", "name", "parent_contact_consent"]].copy()
            consent["parent_contact_consent"] = consent["parent_contact_consent"].map(
                lambda value: "Recorded" if bool(value) else "Not recorded"
            )
            st.dataframe(consent, hide_index=True, use_container_width=True)


def _render_methodology() -> None:
    st.title("How DataMoat calculates its metrics")
    st.caption(
        "A plain-language guide to the inputs, formulas, and limits behind the charts and summaries."
    )
    topics = [
        ("overview", "Overview and test averages"),
        ("mastery", "Topic mastery"),
        ("early_warning", "Early-warning support bands"),
        ("percentile", "Percentiles, forecasts, and medians"),
        ("economics", "Fee and economics totals"),
        ("departure", "Departure-risk model"),
        ("feature_chart", "Reading the feature comparison chart"),
    ]
    for key, title in topics:
        with st.expander(title):
            st.markdown(METHODOLOGY_TEXT[key])


NAV_GROUPS = [
    ("Insights", ["Overview", "Early warning", "Departure risk", "Topic mastery", "Percentile",
                  "My progress", "Study plan", "Child progress"]),
    ("Reports", ["Parent reports", "Reports", "Economics"]),
    ("Data", ["Data & imports", "Data records", "Data validation"]),
    ("Administration", ["User access", "Privacy"]),
    ("Help", ["Methodology"]),
]


def _render_sidebar(
    user: dict[str, Any], actual_role: str, roster: pd.DataFrame
) -> tuple[str, str, str | None]:
    sb = st.sidebar
    sb.markdown("### DataMoat")
    sb.caption(f"{user['institute_name']}  \n{user['display_name']} · {actual_role}")

    effective_role = actual_role
    owner_preview_student_id = None
    if actual_role == "Owner":
        with sb.expander("View as another role"):
            effective_role = st.selectbox(
                "Role", list(ROLE_PAGES), index=0,
                help="Read-only preview. It never changes your real permissions.",
            )
            if effective_role in {"Student", "Parent"} and not roster.empty:
                owner_preview_student_id = st.selectbox(
                    "Linked student",
                    roster["student_id"].astype(str).tolist(),
                    format_func=lambda sid: (
                        f"{roster.loc[roster['student_id'].astype(str) == sid, 'name'].iloc[0]} · {sid}"
                    ),
                )

    available = ROLE_PAGES[effective_role]
    if st.session_state.get("page") not in available:
        st.session_state["page"] = available[0]
    placed: set[str] = set()
    for group, pages in NAV_GROUPS:
        items = [p for p in pages if p in available]
        if not items:
            continue
        sb.caption(group.upper())
        for name in items:
            placed.add(name)
            if sb.button(name, key=f"nav_{name}", use_container_width=True,
                         type="primary" if st.session_state["page"] == name else "secondary"):
                st.session_state["page"] = name
    for name in available:
        if name not in placed:
            if sb.button(name, key=f"nav_{name}", use_container_width=True,
                         type="primary" if st.session_state["page"] == name else "secondary"):
                st.session_state["page"] = name
    return st.session_state["page"], effective_role, owner_preview_student_id


def _render_sidebar_footer(user: dict[str, Any]) -> None:
    sb = st.sidebar
    sb.divider()
    with sb.expander("Change password"):
        with st.form("voluntary_password_change"):
            cur = st.text_input("Current password", type="password")
            new_pw = st.text_input("New password", type="password")
            go = st.form_submit_button("Update")
        if go:
            ok, msg = auth.change_own_password(user["id"], cur, new_pw)
            (st.success if ok else st.error)(msg)
    if sb.button("Sign out", use_container_width=True):
        for key in ("auth_user_id", "view_role", "preview_student_id", "page", "attrition"):
            st.session_state.pop(key, None)
        st.rerun()


def _render_app(user: dict[str, Any]) -> None:
    actual_role = user["role"]
    initial_roster = (
        student_roster(user["institute_id"])
        if actual_role == "Owner"
        else pd.DataFrame()
    )
    page, effective_role, owner_preview_student_id = _render_sidebar(
        user, actual_role, initial_roster
    )
    student_id = _current_student(
        effective_role, user, initial_roster, owner_preview_student_id
    )
    if effective_role in {"Student", "Parent"}:
        linked_id = student_id
    else:
        linked_id = None
    data = load_dataset(user["institute_id"], effective_role, linked_id)

    if data["students"].empty:
        cohorts: list[str] = []
    else:
        cohorts = sorted(data["students"]["cohort"].dropna().astype(str).unique().tolist())
    if effective_role in {"Owner", "Teacher"}:
        st.sidebar.divider()
        st.sidebar.caption("FILTERS")
    if effective_role in {"Owner", "Teacher"} and cohorts:
        cohort = st.sidebar.selectbox("Cohort", cohorts)
    elif effective_role in {"Student", "Parent"} and not data["students"].empty:
        cohort = str(data["students"].iloc[0]["cohort"])
    else:
        cohort = None

    all_test_dates = pd.to_datetime(data["tests"].get("test_date", pd.Series(dtype=object)), errors="coerce").dropna()
    default_cutoff = all_test_dates.max().date() if not all_test_dates.empty else date.today()
    if effective_role in {"Owner", "Teacher"}:
        cutoff = st.sidebar.date_input(
            "Reporting cutoff",
            value=default_cutoff,
            max_value=max(date.today(), default_cutoff),
            help="Data after this date is excluded from time-based analysis.",
        )
    else:
        cutoff = default_cutoff

    if effective_role in {"Owner", "Teacher"} and cohort and not data["students"].empty:
        cohort_student_ids = set(
            data["students"].loc[
                data["students"]["cohort"].astype(str) == cohort, "student_id"
            ].astype(str)
        )
        data["students"] = data["students"][
            data["students"]["student_id"].astype(str).isin(cohort_student_ids)
        ].copy()
        data["test_responses"] = data["test_responses"][
            data["test_responses"]["student_id"].astype(str).isin(cohort_student_ids)
        ].copy()
        for name in ("attendance", "engagement", "fees"):
            if not data[name].empty and "student_id" in data[name]:
                data[name] = data[name][data[name]["student_id"].astype(str).isin(cohort_student_ids)].copy()
    analytics = compute_analytics(data, cohort, cutoff)
    roster = initial_roster if actual_role == "Owner" else data["students"]
    counts = dataset_counts(user["institute_id"], effective_role)


    if page == "Overview":
        _render_overview(data, analytics, cohort, user, cutoff, effective_role)
    elif page == "Topic mastery":
        _render_mastery(analytics, effective_role, student_id)
    elif page == "Early warning":
        _render_risk(analytics, effective_role)
    elif page == "Departure risk":
        if actual_role == "Owner" and effective_role == "Owner":
            _render_departure_risk(user)
        else:
            st.error("This page is restricted to institute Owners.")
    elif page == "Percentile":
        _render_predictions(analytics)
    elif page in {"Parent reports", "Reports"}:
        _render_parent_reports(
            analytics,
            roster,
            effective_role,
            user,
            student_id,
            cutoff,
        )
    elif page == "Economics":
        if actual_role == "Owner" and effective_role == "Owner":
            _render_economics(data, analytics, cutoff)
        else:
            st.error("This page is restricted to institute Owners.")
    elif page in {"Data & imports", "Data records"}:
        _render_data_page(data, user, actual_role, effective_role)
    elif page == "User access":
        if actual_role == "Owner" and effective_role == "Owner":
            _render_accounts(user)
        else:
            st.error("This page is restricted to institute Owners.")
    elif page == "Data validation":
        _render_validation(data, counts, effective_role, user["institute_id"])
    elif page == "My progress":
        _render_student_portal(analytics, student_id, user, show_plan=False)
    elif page == "Study plan":
        _render_student_portal(analytics, student_id, user, show_plan=True)
    elif page == "Child progress":
        _render_student_portal(analytics, student_id, user, show_plan=False)
    elif page == "Privacy":
        _render_privacy(data, user, effective_role)
    elif page == "Methodology":
        _render_methodology()

    _render_sidebar_footer(user)


def main() -> None:
    if not _password_secret_is_set():
        _render_database_error(
            "SESSION_SECRET is not set. Configure it before running DataMoat."
        )
    try:
        check_database()
        with session_scope() as session:
            session.scalar(select(Institute.id).limit(1))
    except Exception:
        _render_database_error(
            "Could not connect to PostgreSQL or find the DataMoat schema."
        )

    user_id = st.session_state.get("auth_user_id")
    if not user_id:
        _render_auth()
        return
    user = auth.get_active_user(user_id)
    if user is None:
        st.session_state.pop("auth_user_id", None)
        st.warning("Your account is no longer active. Sign in again or contact an Owner.")
        _render_auth()
        return
    if user.get("must_change_password"):
        _render_forced_password_change(user)
        return
    _render_app(user)


def _render_forced_password_change(user: dict[str, Any]) -> None:
    st.title("Set a new password")
    st.write("Your account uses a temporary password. Choose a new one to continue.")
    with st.form("forced_password_change"):
        current = st.text_input("Temporary password", type="password")
        new = st.text_input("New password (12 or more characters)", type="password")
        again = st.text_input("Repeat new password", type="password")
        submitted = st.form_submit_button("Update password", type="primary")
    if submitted:
        if new != again:
            st.error("The new passwords do not match.")
        else:
            ok, message = auth.change_own_password(user["id"], current, new)
            if ok:
                st.success(message)
                st.rerun()
            else:
                st.error(message)
    if st.button("Sign out"):
        st.session_state.pop("auth_user_id", None)
        st.rerun()


main()
