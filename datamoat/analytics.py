from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd


def _date_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if frame.empty or column not in frame:
        return pd.Series(dtype="datetime64[ns]")
    return pd.to_datetime(frame[column], errors="coerce")


def _mean(values: pd.Series) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.mean()) if not clean.empty else None


def _conformal_predictions(
    students: pd.DataFrame, summaries: dict[str, dict[str, Any]]
) -> tuple[dict[str, dict[str, float]], int, float | None]:
    labeled = []
    for _, student in students.iterrows():
        sid = str(student["student_id"])
        outcome = student.get("final_percentile")
        score = summaries.get(sid, {}).get("score_mean")
        try:
            outcome_value = float(outcome)
        except (TypeError, ValueError):
            continue
        if score is not None and math.isfinite(outcome_value) and 0 <= outcome_value <= 100:
            labeled.append((sid, float(score), outcome_value))
    if len(labeled) < 5:
        return {}, len(labeled), None

    x = np.array([item[1] for item in labeled], dtype=float)
    y = np.array([item[2] for item in labeled], dtype=float)
    residuals: list[float] = []
    for index in range(len(labeled)):
        keep = np.arange(len(labeled)) != index
        train_x = x[keep]
        train_y = y[keep]
        design = np.column_stack([np.ones(len(train_x)), train_x])
        coeffs, *_ = np.linalg.lstsq(design, train_y, rcond=None)
        estimate = float(coeffs[0] + coeffs[1] * x[index])
        residuals.append(abs(float(y[index]) - estimate))

    calibration = np.sort(np.asarray(residuals))
    quantile_index = min(
        len(calibration) - 1,
        max(0, math.ceil((len(calibration) + 1) * 0.8) - 1),
    )
    radius = float(calibration[quantile_index])
    full_design = np.column_stack([np.ones(len(x)), x])
    coeffs, *_ = np.linalg.lstsq(full_design, y, rcond=None)
    predictions: dict[str, dict[str, float]] = {}
    for sid, summary in summaries.items():
        score = summary.get("score_mean")
        if score is None:
            continue
        midpoint = float(np.clip(coeffs[0] + coeffs[1] * score, 0, 100))
        predictions[sid] = {
            "low": round(max(0, midpoint - radius), 1),
            "mid": round(midpoint, 1),
            "high": round(min(100, midpoint + radius), 1),
        }
    return predictions, len(labeled), radius


def compute_analytics(
    data: dict[str, pd.DataFrame], cohort: str | None, cutoff: date
) -> dict[str, Any]:
    students = data["students"].copy()
    if students.empty:
        return {
            "students": students,
            "mastery": pd.DataFrame(),
            "summaries": {},
            "risk": pd.DataFrame(),
            "predictions": {},
            "calibration_count": 0,
            "calibration_radius": None,
            "responses": pd.DataFrame(),
            "tests": pd.DataFrame(),
        }

    if cohort and "cohort" in students:
        students = students[students["cohort"].astype(str) == str(cohort)].copy()
    students = students.reset_index(drop=True)
    student_ids = set(students["student_id"].astype(str))

    tests = data["tests"].copy()
    if not tests.empty:
        tests["_date"] = _date_column(tests, "test_date")
        tests = tests[tests["_date"].dt.date <= cutoff]

    responses = data["test_responses"].copy()
    if not responses.empty:
        responses["student_id"] = responses["student_id"].astype(str)
        responses = responses[responses["student_id"].isin(student_ids)]
        responses = responses.merge(
            tests[["test_id", "_date"]] if not tests.empty else pd.DataFrame(columns=["test_id", "_date"]),
            on="test_id",
            how="inner",
        )
        responses["is_correct"] = responses["is_correct"].map(
            lambda value: 1 if value is True or str(value).strip().casefold() in {"1", "true", "yes"} else 0
        )
    scores_by_test = pd.DataFrame(columns=["student_id", "test_id", "score", "_date"])
    if not responses.empty:
        scores_by_test = (
            responses.groupby(["student_id", "test_id", "_date"], as_index=False)
            .agg(score=("is_correct", "mean"))
            .sort_values("_date")
        )

    attendance = data["attendance"].copy()
    if not attendance.empty:
        attendance["student_id"] = attendance["student_id"].astype(str)
        attendance["_date"] = _date_column(attendance, "date")
        attendance = attendance[
            attendance["student_id"].isin(student_ids)
            & attendance["_date"].notna()
            & (attendance["_date"].dt.date <= cutoff)
        ]
        attendance["present"] = attendance["present"].map(
            lambda value: bool(value) if isinstance(value, (bool, np.bool_)) else str(value).strip().casefold() in {"1", "true", "yes", "present"}
        )

    engagement = data["engagement"].copy()
    if not engagement.empty:
        engagement["student_id"] = engagement["student_id"].astype(str)
        engagement["_date"] = _date_column(engagement, "date")
        engagement = engagement[
            engagement["student_id"].isin(student_ids)
            & engagement["_date"].notna()
            & (engagement["_date"].dt.date <= cutoff)
        ]

    summaries: dict[str, dict[str, Any]] = {}
    cutoff_start = pd.Timestamp(cutoff - timedelta(days=27))
    for _, student in students.iterrows():
        sid = str(student["student_id"])
        student_scores = scores_by_test[scores_by_test["student_id"].astype(str) == sid]
        test_means = student_scores["score"].astype(float).tolist()
        recent_scores = student_scores.tail(3)["score"].astype(float).tolist()
        overall_score = float(np.mean(test_means)) if test_means else None
        recent_score = float(np.mean(recent_scores)) if recent_scores else overall_score

        student_att = attendance[attendance["student_id"] == sid] if not attendance.empty else attendance
        overall_att = _mean(student_att["present"].astype(float)) if not student_att.empty else None
        recent_att_rows = (
            student_att[student_att["_date"] >= cutoff_start] if not student_att.empty else student_att
        )
        recent_att = _mean(recent_att_rows["present"].astype(float)) if not recent_att_rows.empty else overall_att

        student_eng = engagement[engagement["student_id"] == sid] if not engagement.empty else engagement
        homework_rate = _mean(student_eng["homework_submitted"]) if not student_eng.empty else None
        summaries[sid] = {
            "student_id": sid,
            "name": str(student.get("name", sid)),
            "batch_id": str(student.get("batch_id", "")),
            "cohort": str(student.get("cohort", "")),
            "score_mean": overall_score,
            "score_recent": recent_score,
            "attendance_overall": overall_att,
            "attendance_recent": recent_att,
            "homework_rate": homework_rate,
            "test_count": len(test_means),
            "attendance_count": len(student_att),
            "engagement_count": len(student_eng),
        }

    questions = data["questions"].copy()
    mastery = pd.DataFrame()
    if not responses.empty and not questions.empty:
        mastery = responses.merge(
            questions[["question_id", "subject", "topic"]],
            on="question_id",
            how="inner",
        ).merge(
            students[["student_id", "batch_id"]],
            on="student_id",
            how="inner",
        )
        mastery = (
            mastery.groupby(["student_id", "batch_id", "subject", "topic"], as_index=False)
            .agg(attempts=("is_correct", "size"), correct=("is_correct", "sum"))
        )
        mastery["accuracy"] = mastery["correct"] / mastery["attempts"]
        # Beta(1,1) shrinkage avoids treating one lucky answer as mastery.
        mastery["mastery"] = (mastery["correct"] + 1) / (mastery["attempts"] + 2)

    risk_rows = []
    for sid, summary in summaries.items():
        score = summary["score_mean"]
        recent = summary["score_recent"]
        attendance_rate = summary["attendance_recent"]
        homework_rate = summary["homework_rate"]
        score_observed = score is not None
        attendance_observed = attendance_rate is not None
        homework_observed = homework_rate is not None
        indicators = sum([score_observed, attendance_observed, homework_observed])
        points = 0
        reasons: list[str] = []
        actions: list[str] = []
        if score_observed and score < 0.45:
            points += 28
            reasons.append(f"Recent test average is {score:.0%}.")
            actions.append("Offer focused revision and a teacher check-in.")
        if score_observed and recent is not None and score - recent >= 0.10:
            points += 22
            reasons.append(f"Scores are down {score - recent:.0%} from the overall average.")
            actions.append("Review recent mistakes together and set a short practice plan.")
        if attendance_observed and attendance_rate < 0.70:
            points += 35
            reasons.append(f"Recent attendance is {attendance_rate:.0%}.")
            actions.append("Check in privately to understand any barriers to attendance.")
        elif attendance_observed and attendance_rate < 0.80:
            points += 18
            reasons.append(f"Recent attendance is {attendance_rate:.0%}.")
            actions.append("Ask whether schedule or access support would help.")
        if homework_observed and homework_rate < 0.60:
            points += 15
            reasons.append(f"Homework completion is {homework_rate:.0%}.")
            actions.append("Offer a smaller, clearly timed set of practice work.")
        if indicators == 0:
            band = "Insufficient data"
            points = None
        elif points >= 55:
            band = "High support priority"
        elif points >= 30:
            band = "Needs a check-in"
        else:
            band = "Steady"
        if not actions and indicators:
            actions.append("Continue regular feedback and recognize steady effort.")
        risk_rows.append(
            {
                "student_id": sid,
                "name": summary["name"],
                "batch_id": summary["batch_id"],
                "support_score": points,
                "band": band,
                "signals": "; ".join(reasons) if reasons else "No concerning signals in available data.",
                "suggested_support": "; ".join(actions),
                "data_sources": indicators,
            }
        )
    risk = pd.DataFrame(risk_rows)
    if not risk.empty:
        risk["_sort"] = risk["support_score"].fillna(-1)
        risk = risk.sort_values("_sort", ascending=False).drop(columns="_sort")

    predictions, calibration_count, calibration_radius = _conformal_predictions(students, summaries)
    return {
        "students": students,
        "mastery": mastery,
        "summaries": summaries,
        "risk": risk,
        "predictions": predictions,
        "calibration_count": calibration_count,
        "calibration_radius": calibration_radius,
        "responses": responses,
        "tests": tests,
    }


def economic_summary(
    fees: pd.DataFrame, students: pd.DataFrame, cutoff: date
) -> tuple[dict[str, float], pd.DataFrame]:
    if fees.empty:
        return {
            "scheduled": 0.0,
            "collected": 0.0,
            "overdue": 0.0,
            "overdue_count": 0.0,
        }, pd.DataFrame()
    valid = fees.copy()
    valid["_due"] = _date_column(valid, "due_date")
    valid["_paid"] = _date_column(valid, "paid_date")
    valid["amount"] = pd.to_numeric(valid["amount"], errors="coerce").fillna(0)
    due_by_cutoff = valid["_due"].dt.date <= cutoff
    # A payment counts only if it was made on or before the reporting cutoff.
    paid = valid["_paid"].notna() & (valid["_paid"].dt.date <= cutoff)
    overdue = due_by_cutoff & ~paid
    valid = valid.merge(
        students[["student_id", "name", "batch_id"]],
        on="student_id",
        how="left",
    )
    overdue_rows = valid[overdue].copy()
    summary = {
        "scheduled": float(valid.loc[due_by_cutoff, "amount"].sum()),
        "collected": float(valid.loc[due_by_cutoff & paid, "amount"].sum()),
        "overdue": float(valid.loc[overdue, "amount"].sum()),
        "overdue_count": float(overdue.sum()),
    }
    return summary, overdue_rows[
        ["student_id", "name", "batch_id", "instalment_no", "due_date", "amount"]
    ].sort_values("due_date")


def weak_and_strong_topics(mastery: pd.DataFrame, student_id: str) -> tuple[list[str], list[str]]:
    if mastery.empty:
        return [], []
    rows = mastery[mastery["student_id"].astype(str) == str(student_id)].copy()
    rows = rows[rows["attempts"] > 0].sort_values("mastery")
    weak = rows["topic"].head(3).tolist()
    strong = rows.sort_values("mastery", ascending=False)["topic"].head(2).tolist()
    weak = [topic for topic in weak if topic not in strong]
    return weak, strong
