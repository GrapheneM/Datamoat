from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKeyConstraint,
    Integer,
    JSON,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Institute(Base):
    __tablename__ = "dm_institutes"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class User(Base):
    __tablename__ = "dm_users"
    __table_args__ = (
        UniqueConstraint("institute_id", "login_id", name="uq_dm_users_login"),
        ForeignKeyConstraint(
            ["institute_id"], ["dm_institutes.id"], ondelete="CASCADE"
        ),
        CheckConstraint(
            "role IN ('Owner', 'Teacher', 'Student', 'Parent')",
            name="ck_dm_users_role",
        ),
        CheckConstraint("failed_login_count >= 0", name="ck_dm_users_failures"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    institute_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    login_id: Mapped[str] = mapped_column(String(180), nullable=False)
    display_name: Mapped[str] = mapped_column(String(160), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(512), nullable=False)
    linked_student_id: Mapped[str | None] = mapped_column(String(80), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    failed_login_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    must_change_password: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    locked_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class Student(Base):
    __tablename__ = "dm_students"
    __table_args__ = (
        ForeignKeyConstraint(
            ["institute_id"], ["dm_institutes.id"], ondelete="CASCADE"
        ),
        CheckConstraint(
            "final_percentile IS NULL OR "
            "(final_percentile >= 0 AND final_percentile <= 100)",
            name="ck_dm_students_percentile",
        ),
        CheckConstraint(
            "dropout_date IS NULL OR dropout_date >= join_date",
            name="ck_dm_students_dropout_date",
        ),
    )

    institute_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    student_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    batch_id: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    join_date: Mapped[Any] = mapped_column(Date, nullable=False)
    cohort: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    class_name: Mapped[str | None] = mapped_column(String(40), nullable=True)
    parent_contact_consent: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    parent_language: Mapped[str] = mapped_column(
        String(8), nullable=False, default="en"
    )
    dropout_date: Mapped[Any | None] = mapped_column(Date, nullable=True)
    final_percentile: Mapped[float | None] = mapped_column(Float, nullable=True)


class Question(Base):
    __tablename__ = "dm_questions"
    __table_args__ = (
        ForeignKeyConstraint(
            ["institute_id"], ["dm_institutes.id"], ondelete="CASCADE"
        ),
    )

    institute_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    question_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    subject: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    topic: Mapped[str] = mapped_column(String(160), nullable=False, index=True)
    sub_topic: Mapped[str | None] = mapped_column(String(160), nullable=True)


class Test(Base):
    __tablename__ = "dm_tests"
    __table_args__ = (
        ForeignKeyConstraint(
            ["institute_id"], ["dm_institutes.id"], ondelete="CASCADE"
        ),
        CheckConstraint(
            "max_marks IS NULL OR max_marks > 0", name="ck_dm_tests_max_marks"
        ),
    )

    institute_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    test_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    test_date: Mapped[Any] = mapped_column(Date, nullable=False, index=True)
    max_marks: Mapped[float | None] = mapped_column(Float, nullable=True)


class TestResponse(Base):
    __tablename__ = "dm_test_responses"
    __table_args__ = (
        ForeignKeyConstraint(
            ["institute_id", "test_id"],
            ["dm_tests.institute_id", "dm_tests.test_id"],
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["institute_id", "student_id"],
            ["dm_students.institute_id", "dm_students.student_id"],
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["institute_id", "question_id"],
            ["dm_questions.institute_id", "dm_questions.question_id"],
            ondelete="CASCADE",
        ),
        CheckConstraint(
            "time_taken_sec IS NULL OR time_taken_sec >= 0",
            name="ck_dm_responses_time",
        ),
    )

    institute_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    test_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    student_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    question_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    is_correct: Mapped[bool] = mapped_column(Boolean, nullable=False)
    time_taken_sec: Mapped[int | None] = mapped_column(Integer, nullable=True)


class Attendance(Base):
    __tablename__ = "dm_attendance"
    __table_args__ = (
        UniqueConstraint(
            "institute_id",
            "student_id",
            "date",
            "session_id",
            name="uq_dm_attendance_session",
        ),
        ForeignKeyConstraint(
            ["institute_id", "student_id"],
            ["dm_students.institute_id", "dm_students.student_id"],
            ondelete="CASCADE",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    institute_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    student_id: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    date: Mapped[Any] = mapped_column(Date, nullable=False)
    present: Mapped[bool] = mapped_column(Boolean, nullable=False)
    session_id: Mapped[str] = mapped_column(String(80), nullable=False, default="")


class Fee(Base):
    __tablename__ = "dm_fees"
    __table_args__ = (
        UniqueConstraint(
            "institute_id",
            "student_id",
            "instalment_no",
            name="uq_dm_fee_instalment",
        ),
        ForeignKeyConstraint(
            ["institute_id", "student_id"],
            ["dm_students.institute_id", "dm_students.student_id"],
            ondelete="CASCADE",
        ),
        CheckConstraint("amount >= 0", name="ck_dm_fees_amount"),
        CheckConstraint("instalment_no > 0", name="ck_dm_fees_instalment"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    institute_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    student_id: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    instalment_no: Mapped[int] = mapped_column(Integer, nullable=False)
    due_date: Mapped[Any] = mapped_column(Date, nullable=False, index=True)
    paid_date: Mapped[Any | None] = mapped_column(Date, nullable=True)
    amount: Mapped[float] = mapped_column(Float, nullable=False)


class Engagement(Base):
    __tablename__ = "dm_engagement"
    __table_args__ = (
        UniqueConstraint(
            "institute_id", "student_id", "date", name="uq_dm_engagement_day"
        ),
        ForeignKeyConstraint(
            ["institute_id", "student_id"],
            ["dm_students.institute_id", "dm_students.student_id"],
            ondelete="CASCADE",
        ),
        CheckConstraint(
            "homework_submitted IS NULL OR "
            "(homework_submitted >= 0 AND homework_submitted <= 1)",
            name="ck_dm_engagement_homework",
        ),
        CheckConstraint(
            "doubts_asked >= 0 AND app_logins >= 0",
            name="ck_dm_engagement_counts",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    institute_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    student_id: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    date: Mapped[Any] = mapped_column(Date, nullable=False)
    doubts_asked: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    homework_submitted: Mapped[float | None] = mapped_column(
        Float, nullable=True, default=None
    )
    app_logins: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class AuditLog(Base):
    __tablename__ = "dm_audit_log"
    __table_args__ = (
        ForeignKeyConstraint(
            ["institute_id"], ["dm_institutes.id"], ondelete="CASCADE"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    institute_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    actor_user_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    action: Mapped[str] = mapped_column(String(80), nullable=False)
    object_type: Mapped[str] = mapped_column(String(80), nullable=False)
    object_id: Mapped[str | None] = mapped_column(String(180), nullable=True)
    details: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class Prediction(Base):
    """Stored departure-risk predictions with provenance (one per student per run)."""

    __tablename__ = "dm_predictions"
    __table_args__ = (
        UniqueConstraint(
            "institute_id", "student_id", "model_version", "feature_cutoff",
            name="uq_dm_prediction_run",
        ),
        ForeignKeyConstraint(
            ["institute_id", "student_id"],
            ["dm_students.institute_id", "dm_students.student_id"],
            ondelete="CASCADE",
        ),
        CheckConstraint("probability >= 0 AND probability <= 1", name="ck_dm_pred_prob"),
        CheckConstraint("horizon_days > 0", name="ck_dm_pred_horizon"),
        CheckConstraint(
            "outcome IS NULL OR outcome IN ('left', 'stayed')", name="ck_dm_pred_outcome"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    institute_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    student_id: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    predicted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    horizon_days: Mapped[int] = mapped_column(Integer, nullable=False)
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_version: Mapped[str] = mapped_column(String(80), nullable=False)
    validated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    feature_cutoff: Mapped[Any] = mapped_column(Date, nullable=False)
    outcome: Mapped[str | None] = mapped_column(String(10), nullable=True)
    outcome_date: Mapped[Any | None] = mapped_column(Date, nullable=True)
