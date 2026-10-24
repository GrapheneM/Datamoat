from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pandas as pd
from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .db import session_scope
from .models import (
    Attendance,
    AuditLog,
    Engagement,
    Fee,
    Prediction,
    Question,
    Student,
    Test,
    TestResponse,
    User,
)
from .validation import NATURAL_KEYS, TABLE_ORDER

TABLE_MODELS = {
    "students": Student,
    "questions": Question,
    "tests": Test,
    "test_responses": TestResponse,
    "attendance": Attendance,
    "fees": Fee,
    "engagement": Engagement,
}


def _rows_as_frame(rows: list[Any], model: Any) -> pd.DataFrame:
    names = [column.name for column in model.__table__.columns if column.name != "institute_id"]
    values = [{name: getattr(row, name) for name in names} for row in rows]
    return pd.DataFrame(values, columns=names)


def _query_table(
    session: Session,
    model: Any,
    institute_id: str,
    student_ids: list[str] | None,
) -> pd.DataFrame:
    query = select(model).where(model.institute_id == institute_id)
    if student_ids is not None and hasattr(model, "student_id"):
        if not student_ids:
            return _rows_as_frame([], model)
        query = query.where(model.student_id.in_(student_ids))
    return _rows_as_frame(list(session.scalars(query).all()), model)


def load_dataset(
    institute_id: str,
    role: str,
    linked_student_id: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Load only the rows and sensitive tables the current role is permitted to see."""
    restricted = role in {"Student", "Parent"}
    student_ids = [linked_student_id] if restricted and linked_student_id else (
        [] if restricted else None
    )
    tables: dict[str, pd.DataFrame] = {}
    with session_scope() as session:
        for name in TABLE_ORDER:
            model = TABLE_MODELS[name]
            if name == "fees" and role != "Owner":
                tables[name] = _rows_as_frame([], model)
                continue
            tables[name] = _query_table(session, model, institute_id, student_ids)
    return tables


def dataset_counts(institute_id: str, role: str = "Owner") -> dict[str, int]:
    counts = {}
    with session_scope() as session:
        for name, model in TABLE_MODELS.items():
            if name == "fees" and role != "Owner":
                continue
            counts[name] = int(
                session.scalar(
                    select(func.count())
                    .select_from(model)
                    .where(model.institute_id == institute_id)
                )
                or 0
            )
    return counts


def student_roster(institute_id: str) -> pd.DataFrame:
    with session_scope() as session:
        rows = session.scalars(
            select(Student)
            .where(Student.institute_id == institute_id)
            .order_by(Student.name)
        ).all()
        return pd.DataFrame(
            [
                {
                    "student_id": row.student_id,
                    "name": row.name,
                    "batch_id": row.batch_id,
                    "cohort": row.cohort,
                    "parent_contact_consent": row.parent_contact_consent,
                }
                for row in rows
            ],
            columns=[
                "student_id",
                "name",
                "batch_id",
                "cohort",
                "parent_contact_consent",
            ],
        )


def validate_references(
    institute_id: str, table_name: str, frame: pd.DataFrame
) -> list[str]:
    errors: list[str] = []
    with session_scope() as session:
        if table_name in {"attendance", "fees", "engagement", "test_responses"}:
            student_ids = set(
                session.scalars(
                    select(Student.student_id).where(Student.institute_id == institute_id)
                ).all()
            )
            missing = sorted(set(frame["student_id"].dropna()) - student_ids)
            if missing:
                errors.append(
                    "Unknown student ID(s): "
                    + ", ".join(str(value) for value in missing[:8])
                    + ". Import students first."
                )
        if table_name == "test_responses":
            for column, model, label in (
                ("test_id", Test, "test"),
                ("question_id", Question, "question"),
            ):
                known = set(
                    session.scalars(
                        select(getattr(model, column)).where(
                            model.institute_id == institute_id
                        )
                    ).all()
                )
                missing = sorted(set(frame[column].dropna()) - known)
                if missing:
                    errors.append(
                        f"Unknown {label} ID(s): "
                        + ", ".join(str(value) for value in missing[:8])
                        + f". Import {label}s first."
                    )
    return errors


def _plain_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if hasattr(value, "item"):
        return value.item()
    return value


def save_table(
    actor_id: str,
    institute_id: str,
    table_name: str,
    frame: pd.DataFrame,
    mode: str,
) -> tuple[bool, str]:
    if table_name not in TABLE_MODELS or mode not in {"append", "replace"}:
        return False, "Unsupported table or save mode."
    model = TABLE_MODELS[table_name]
    key_names = ["institute_id", *NATURAL_KEYS[table_name]]
    records = []
    for row in frame.to_dict(orient="records"):
        record = {key: _plain_value(value) for key, value in row.items()}
        record["institute_id"] = institute_id
        records.append(record)

    try:
        with session_scope() as session:
            actor = session.get(User, actor_id)
            if actor is None or not actor.is_active or actor.role != "Owner":
                return False, "Only an active institute owner can upload data."
            if actor.institute_id != institute_id:
                return False, "The selected institute does not match this account."
            if table_name == "students" and mode == "replace":
                new_ids = {str(row["student_id"]) for row in records}
                account_ids = set(
                    session.scalars(
                        select(User.linked_student_id).where(
                            User.institute_id == institute_id,
                            User.is_active.is_(True),
                            User.linked_student_id.is_not(None),
                        )
                    ).all()
                )
                removed_accounts = account_ids - new_ids
                if removed_accounts:
                    return (
                        False,
                        "Replacement would remove student IDs linked to active accounts: "
                        + ", ".join(sorted(removed_accounts)[:8])
                        + ". Update those accounts first.",
                    )
            if mode == "replace":
                session.execute(
                    delete(model).where(model.institute_id == institute_id)
                )
            if records:
                statement = pg_insert(model).values(records)
                update_values = {
                    column.name: getattr(statement.excluded, column.name)
                    for column in model.__table__.columns
                    if column.name not in key_names
                    and column.name not in {"id"}
                }
                if mode == "append" and update_values:
                    statement = statement.on_conflict_do_update(
                        index_elements=[getattr(model, key) for key in key_names],
                        set_=update_values,
                    )
                else:
                    statement = statement.on_conflict_do_nothing(
                        index_elements=[getattr(model, key) for key in key_names]
                    )
                session.execute(statement)
            session.add(
                AuditLog(
                    institute_id=institute_id,
                    actor_user_id=actor_id,
                    action="data.imported",
                    object_type=table_name,
                    object_id=None,
                    details={"rows": len(records), "mode": mode},
                )
            )
            session.commit()
            label = "replaced" if mode == "replace" else "added or updated"
            return True, f"{len(records)} row(s) {label} in {table_name}."
    except IntegrityError:
        return (
            False,
            "The database rejected this change because it conflicts with linked records "
            "or a table constraint. No rows were saved; check the data links and try again.",
        )


def clear_academic_data(actor_id: str, institute_id: str) -> tuple[bool, str]:
    # Child rows are removed first so foreign-key checks preserve referential integrity.
    deletion_order = (
        TestResponse,
        Attendance,
        Fee,
        Engagement,
        Test,
        Question,
        Student,
    )
    with session_scope() as session:
        actor = session.get(User, actor_id)
        if actor is None or not actor.is_active or actor.role != "Owner":
            return False, "Only an active institute owner can clear academic data."
        if actor.institute_id != institute_id:
            return False, "The selected institute does not match this account."
        for model in deletion_order:
            session.execute(delete(model).where(model.institute_id == institute_id))
        session.add(
            AuditLog(
                institute_id=institute_id,
                actor_user_id=actor_id,
                action="data.cleared",
                object_type="academic_data",
                object_id=None,
                details={},
            )
        )
        session.commit()
        return True, "Academic data cleared. Institute accounts and audit history remain."


def recent_audit(institute_id: str, limit: int = 50) -> pd.DataFrame:
    with session_scope() as session:
        logs = session.scalars(
            select(AuditLog)
            .where(AuditLog.institute_id == institute_id)
            .order_by(AuditLog.created_at.desc())
            .limit(limit)
        ).all()
        return pd.DataFrame(
            [
                {
                    "Time": item.created_at,
                    "Account": item.actor_user_id or "system",
                    "Action": item.action,
                    "Record type": item.object_type,
                    "Record ID": item.object_id or "",
                }
                for item in logs
            ],
            columns=["Time", "Account", "Action", "Record type", "Record ID"],
        )


def save_predictions(
    actor_id: str,
    institute_id: str,
    frame: pd.DataFrame,
    model_version: str,
    threshold: float | None,
    validated: bool,
) -> tuple[bool, str]:
    """Store one row per student for a model run. Re-saving the same run is a no-op update."""
    if frame.empty:
        return False, "There are no predictions to save."
    with session_scope() as session:
        actor = session.get(User, actor_id)
        if actor is None or not actor.is_active or actor.role != "Owner" or actor.institute_id != institute_id:
            return False, "Only an active institute owner can store predictions."
        valid_ids = set(
            session.scalars(
                select(Student.student_id).where(Student.institute_id == institute_id)
            ).all()
        )
        rows = [
            {
                "institute_id": institute_id,
                "student_id": str(r.student_id),
                "horizon_days": int(r.horizon_days),
                "probability": float(r.probability),
                "threshold": threshold,
                "model_version": model_version,
                "validated": bool(validated),
                "feature_cutoff": r.feature_cutoff,
            }
            for r in frame.itertuples()
            if str(r.student_id) in valid_ids
        ]
        stmt = pg_insert(Prediction).values(rows)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_dm_prediction_run",
            set_={"probability": stmt.excluded.probability, "threshold": stmt.excluded.threshold},
        )
        session.execute(stmt)
        session.add(
            AuditLog(
                institute_id=institute_id,
                actor_user_id=actor_id,
                action="predictions.saved",
                object_type="prediction_run",
                object_id=model_version,
                details={"rows": len(rows), "validated": bool(validated)},
            )
        )
        session.commit()
    return True, f"Stored {len(rows)} predictions for model {model_version}."


def load_predictions(institute_id: str, limit: int = 2000) -> pd.DataFrame:
    with session_scope() as session:
        rows = session.scalars(
            select(Prediction)
            .where(Prediction.institute_id == institute_id)
            .order_by(Prediction.predicted_at.desc())
            .limit(limit)
        ).all()
        return pd.DataFrame(
            [
                {
                    "student_id": r.student_id, "predicted_at": r.predicted_at,
                    "horizon_days": r.horizon_days, "probability": r.probability,
                    "model_version": r.model_version, "validated": r.validated,
                    "feature_cutoff": r.feature_cutoff, "outcome": r.outcome,
                }
                for r in rows
            ]
        )
