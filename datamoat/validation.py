from __future__ import annotations

import io
import re
from typing import Any

import pandas as pd


TABLE_ORDER = (
    "students",
    "questions",
    "tests",
    "test_responses",
    "attendance",
    "fees",
    "engagement",
)

TABLE_SPECS: dict[str, dict[str, Any]] = {
    "students": {
        "title": "Students",
        "required": ["student_id", "name", "batch_id", "join_date"],
        "columns": [
            "student_id", "name", "batch_id", "join_date", "cohort", "class_name",
            "parent_contact_consent", "parent_language", "dropout_date",
            "final_percentile",
        ],
        "unlocks": "Cohorts, student lists, mastery, and reports",
    },
    "questions": {
        "title": "Questions",
        "required": ["question_id", "subject", "topic"],
        "columns": ["question_id", "subject", "topic", "sub_topic"],
        "unlocks": "Topic mastery and targeted study plans",
    },
    "tests": {
        "title": "Tests",
        "required": ["test_id", "test_date"],
        "columns": ["test_id", "test_date", "max_marks"],
        "unlocks": "Time-based score trends",
    },
    "test_responses": {
        "title": "Test responses",
        "required": ["test_id", "student_id", "question_id", "is_correct"],
        "columns": [
            "test_id", "student_id", "question_id", "is_correct", "time_taken_sec"
        ],
        "unlocks": "Mastery and early-warning analysis",
    },
    "attendance": {
        "title": "Attendance",
        "required": ["student_id", "date", "present"],
        "columns": ["student_id", "date", "present", "session_id"],
        "unlocks": "Attendance trends in early-warning analysis",
    },
    "fees": {
        "title": "Fee instalments",
        "required": ["student_id", "instalment_no", "due_date", "amount"],
        "columns": [
            "student_id", "instalment_no", "due_date", "paid_date", "amount"
        ],
        "unlocks": "Owner-only fee collection and overdue summaries",
    },
    "engagement": {
        "title": "Engagement",
        "required": ["student_id", "date"],
        "columns": [
            "student_id", "date", "doubts_asked", "homework_submitted", "app_logins"
        ],
        "unlocks": "Homework and activity trends in early-warning analysis",
    },
}

ALIASES = {
    "class": "class_name",
    "studentid": "student_id",
    "questionid": "question_id",
    "testid": "test_id",
    "batch": "batch_id",
    "cohort_id": "cohort",
    "correct": "is_correct",
    "attendance": "present",
    "homework": "homework_submitted",
    "final_percentile_score": "final_percentile",
}

NATURAL_KEYS = {
    "students": ["student_id"],
    "questions": ["question_id"],
    "tests": ["test_id"],
    "test_responses": ["test_id", "student_id", "question_id"],
    "attendance": ["student_id", "date", "session_id"],
    "fees": ["student_id", "instalment_no"],
    "engagement": ["student_id", "date"],
}


def template_csv(table_name: str) -> str:
    spec = TABLE_SPECS[table_name]
    example = {
        "students": [
            "STU-001", "Riya Sharma", "JEE-11-A", "2026-04-06", "2026", "11",
            "no", "en", "", "",
        ],
        "questions": ["PHY-001", "Physics", "Kinematics", "1D motion"],
        "tests": ["TEST-001", "2026-04-20", "60"],
        "test_responses": ["TEST-001", "STU-001", "PHY-001", "1", "75"],
        "attendance": ["STU-001", "2026-04-12", "yes", "MORNING"],
        "fees": ["STU-001", "1", "2026-05-10", "", "10000"],
        "engagement": ["STU-001", "2026-04-15", "2", "1", "4"],
    }[table_name]
    return pd.DataFrame([example], columns=spec["columns"]).to_csv(index=False)


def parse_upload(file_name: str, contents: bytes) -> pd.DataFrame:
    if len(contents) > 20 * 1024 * 1024:
        raise ValueError("Files must be 20 MB or smaller.")
    lowered = file_name.lower()
    if lowered.endswith(".csv"):
        return pd.read_csv(io.BytesIO(contents), dtype=object)
    if lowered.endswith(".xlsx"):
        return pd.read_excel(io.BytesIO(contents), dtype=object, engine="openpyxl")
    raise ValueError("Upload a CSV or .xlsx file.")


def _text_cell(value: Any) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    cleaned = str(value).strip()
    return cleaned or None


def _bool_cell(value: Any) -> bool | None:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    cleaned = str(value).strip().casefold()
    if cleaned in {"1", "true", "yes", "y", "present"}:
        return True
    if cleaned in {"0", "false", "no", "n", "absent"}:
        return False
    return None


def _date_series(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    parsed = pd.to_datetime(series, errors="coerce")
    return parsed.dt.date, parsed.isna() & series.notna() & series.astype(str).str.strip().ne("")


def validate_upload(table_name: str, frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    if table_name not in TABLE_SPECS:
        return pd.DataFrame(), ["Choose a supported table."], []
    if frame.empty:
        return pd.DataFrame(), ["The uploaded file has no data rows."], []

    cleaned = frame.copy()
    cleaned.columns = [
        ALIASES.get(
            re.sub(r"[\s-]+", "_", str(column).strip().casefold()),
            re.sub(r"[\s-]+", "_", str(column).strip().casefold()),
        )
        for column in cleaned.columns
    ]
    if cleaned.columns.duplicated().any():
        return pd.DataFrame(), ["The file has duplicate columns after header normalization."], []

    spec = TABLE_SPECS[table_name]
    missing = [name for name in spec["required"] if name not in cleaned.columns]
    if missing:
        return pd.DataFrame(), [f"Missing required column(s): {', '.join(missing)}."], []

    for column in spec["columns"]:
        if column not in cleaned:
            cleaned[column] = None

    extras = [name for name in cleaned.columns if name not in spec["columns"]]
    warnings = []
    if extras:
        warnings.append(
            "Unrecognized columns will not be saved: " + ", ".join(extras[:12])
        )
    cleaned = cleaned[[name for name in cleaned.columns if name in spec["columns"]]]
    errors: list[str] = []

    text_columns = {
        "student_id", "name", "batch_id", "cohort", "class_name",
        "parent_language", "question_id", "subject", "topic", "sub_topic",
        "test_id", "session_id",
    }
    for column in text_columns.intersection(cleaned.columns):
        cleaned[column] = cleaned[column].map(_text_cell)

    for column in {"join_date", "dropout_date", "test_date", "date", "due_date", "paid_date"}.intersection(cleaned.columns):
        cleaned[column], invalid = _date_series(cleaned[column])
        if invalid.any():
            errors.append(f"{column} has {int(invalid.sum())} invalid date value(s).")

    boolean_columns = {"parent_contact_consent", "is_correct", "present"}.intersection(cleaned.columns)
    for column in boolean_columns:
        original = cleaned[column].copy()
        cleaned[column] = original.map(_bool_cell)
        invalid = original.notna() & original.astype(str).str.strip().ne("") & cleaned[column].isna()
        if invalid.any():
            errors.append(f"{column} accepts only yes/no, true/false, or 1/0.")

    numeric_columns = {
        "final_percentile", "max_marks", "time_taken_sec", "instalment_no",
        "amount", "doubts_asked", "homework_submitted", "app_logins",
    }.intersection(cleaned.columns)
    for column in numeric_columns:
        original = cleaned[column].copy()
        cleaned[column] = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & original.astype(str).str.strip().ne("") & cleaned[column].isna()
        if invalid.any():
            errors.append(f"{column} contains a non-numeric value.")

    # Fill only genuinely optional fields with safe values. Required values remain errors.
    if table_name == "students":
        if "cohort" not in cleaned:
            cleaned["cohort"] = cleaned["join_date"].map(lambda value: str(value.year) if value else None)
        else:
            empty_cohort = cleaned["cohort"].isna()
            cleaned.loc[empty_cohort, "cohort"] = cleaned.loc[empty_cohort, "join_date"].map(
                lambda value: str(value.year) if value else None
            )
        cleaned["class_name"] = cleaned.get("class_name", pd.Series(index=cleaned.index, dtype=object))
        cleaned["parent_contact_consent"] = cleaned.get(
            "parent_contact_consent", pd.Series(False, index=cleaned.index)
        ).fillna(False)
        cleaned["parent_language"] = cleaned.get(
            "parent_language", pd.Series("en", index=cleaned.index)
        ).fillna("en")
        cleaned["dropout_date"] = cleaned.get("dropout_date", pd.Series(index=cleaned.index, dtype=object))
        cleaned["final_percentile"] = cleaned.get("final_percentile", pd.Series(index=cleaned.index, dtype=float))
        invalid_outcome = cleaned["final_percentile"].notna() & ~cleaned["final_percentile"].between(0, 100)
        if invalid_outcome.any():
            errors.append("final_percentile must be between 0 and 100.")
        invalid_dropout = (
            cleaned["dropout_date"].notna()
            & cleaned["join_date"].notna()
            & (cleaned["dropout_date"] < cleaned["join_date"])
        )
        if invalid_dropout.any():
            errors.append("dropout_date cannot be earlier than join_date.")
    elif table_name == "questions":
        cleaned["sub_topic"] = cleaned.get("sub_topic", pd.Series(index=cleaned.index, dtype=object))
    elif table_name == "tests":
        cleaned["max_marks"] = cleaned.get("max_marks", pd.Series(index=cleaned.index, dtype=float))
        if (cleaned["max_marks"].dropna() <= 0).any():
            errors.append("max_marks must be greater than zero.")
    elif table_name == "test_responses":
        cleaned["time_taken_sec"] = cleaned.get("time_taken_sec", pd.Series(index=cleaned.index, dtype=float))
        if cleaned["is_correct"].isna().any():
            errors.append("is_correct must be provided for every response.")
        if (cleaned["time_taken_sec"].dropna() < 0).any():
            errors.append("time_taken_sec cannot be negative.")
    elif table_name == "attendance":
        cleaned["session_id"] = cleaned.get("session_id", pd.Series("", index=cleaned.index)).fillna("")
    elif table_name == "fees":
        if (cleaned["instalment_no"].dropna() < 1).any():
            errors.append("instalment_no must be a positive integer.")
        if (cleaned["amount"].dropna() < 0).any():
            errors.append("amount cannot be negative.")
    elif table_name == "engagement":
        cleaned["doubts_asked"] = cleaned.get("doubts_asked", pd.Series(0, index=cleaned.index)).fillna(0)
        cleaned["homework_submitted"] = cleaned.get("homework_submitted", pd.Series(index=cleaned.index, dtype=float))
        cleaned["app_logins"] = cleaned.get("app_logins", pd.Series(0, index=cleaned.index)).fillna(0)
        homework_values = cleaned["homework_submitted"].dropna()
        if (homework_values.lt(0) | homework_values.gt(1)).any():
            errors.append("homework_submitted must be between 0 and 1.")
        if (cleaned["doubts_asked"].dropna() < 0).any() or (cleaned["app_logins"].dropna() < 0).any():
            errors.append("doubts_asked and app_logins cannot be negative.")

    required_value_columns = spec["required"]
    for column in required_value_columns:
        empty = cleaned[column].isna()
        if cleaned[column].dtype == object:
            empty = empty | cleaned[column].astype(str).str.strip().eq("")
        if empty.any():
            errors.append(f"{column} is missing in {int(empty.sum())} row(s).")

    if table_name == "fees":
        cleaned["instalment_no"] = pd.to_numeric(cleaned["instalment_no"], errors="coerce")
        if (cleaned["instalment_no"].dropna() % 1 != 0).any():
            errors.append("instalment_no must contain whole numbers.")
    for column in {"time_taken_sec", "doubts_asked", "app_logins"}.intersection(cleaned.columns):
        if (cleaned[column].dropna() % 1 != 0).any():
            errors.append(f"{column} must contain whole numbers.")

    natural_key = NATURAL_KEYS[table_name]
    duplicates = cleaned.duplicated(subset=natural_key, keep=False)
    if duplicates.any():
        errors.append(
            f"{int(duplicates.sum())} rows repeat a table key ({', '.join(natural_key)}). "
            "Remove duplicates or correct the IDs before saving."
        )

    cleaned = cleaned[spec["columns"]]
    if errors:
        return cleaned, errors, warnings
    return cleaned, [], warnings
