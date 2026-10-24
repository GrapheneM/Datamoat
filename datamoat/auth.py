from __future__ import annotations

import re
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from .db import session_scope
from .models import AuditLog, Institute, Student, User

_hasher = PasswordHasher()
_dummy_hash = _hasher.hash("dummy-value-for-nonexistent-accounts")
_MAX_FAILURES = 8
_LOCK_MINUTES = 15


class AccountError(ValueError):
    pass


def normalize_institute_id(value: str) -> str:
    cleaned = re.sub(r"[\s_]+", "-", value.strip().upper())
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9-]{2,31}", cleaned):
        raise AccountError(
            "Institute ID must be 3–32 characters using letters, numbers, or hyphens."
        )
    return cleaned


def normalize_login_id(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise AccountError("Enter an email address or login ID.")
    if "@" in cleaned:
        cleaned = cleaned.casefold()
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", cleaned):
            raise AccountError("Enter a valid email address.")
    else:
        cleaned = cleaned.casefold()
    if len(cleaned) > 180:
        raise AccountError("Login ID is too long.")
    return cleaned


def normalize_password(value: str) -> str:
    if len(value) < 12:
        raise AccountError("Use a password with at least 12 characters.")
    if len(value) > 256:
        raise AccountError("Password is too long.")
    return value


def _hash_password(password: str) -> str:
    return _hasher.hash(normalize_password(password))


def _verify_password(password_hash: str, password: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def create_institute_owner(
    institute_id: str,
    institute_name: str,
    owner_name: str,
    login_id: str,
    password: str,
) -> tuple[bool, str]:
    try:
        cleaned_id = normalize_institute_id(institute_id)
        cleaned_name = institute_name.strip()
        cleaned_owner = owner_name.strip()
        cleaned_login = normalize_login_id(login_id)
        password_hash = _hash_password(password)
        if not cleaned_name or len(cleaned_name) > 160:
            raise AccountError("Institute name is required and must be under 160 characters.")
        if not cleaned_owner or len(cleaned_owner) > 160:
            raise AccountError("Owner name is required and must be under 160 characters.")

        with session_scope() as session:
            if session.get(Institute, cleaned_id):
                return False, "That Institute ID is already registered. Sign in instead."
            session.add(Institute(id=cleaned_id, name=cleaned_name))
            session.flush()
            owner = User(
                id=str(uuid.uuid4()),
                institute_id=cleaned_id,
                login_id=cleaned_login,
                display_name=cleaned_owner,
                role="Owner",
                password_hash=password_hash,
                is_active=True,
            )
            session.add(owner)
            session.flush()
            session.add(
                AuditLog(
                    institute_id=cleaned_id,
                    actor_user_id=owner.id,
                    action="institute.created",
                    object_type="institute",
                    object_id=cleaned_id,
                    details={},
                )
            )
            session.commit()
            return True, owner.id
    except AccountError as exc:
        return False, str(exc)
    except IntegrityError:
        return False, "The Institute ID or login ID is already in use."


def authenticate(
    institute_id: str, login_id: str, password: str
) -> tuple[bool, str]:
    try:
        cleaned_id = normalize_institute_id(institute_id)
        cleaned_login = normalize_login_id(login_id)
    except AccountError:
        _verify_password(_dummy_hash, password)
        return False, "Institute ID, login ID, or password is incorrect."

    with session_scope() as session:
        user = session.scalar(
            select(User)
            .where(
                User.institute_id == cleaned_id,
                User.login_id == cleaned_login,
                User.is_active.is_(True),
            )
            .with_for_update()
        )
        if user is None:
            _verify_password(_dummy_hash, password)
            return False, "Institute ID, login ID, or password is incorrect."

        now = datetime.now(timezone.utc)
        if user.locked_until and user.locked_until > now:
            _verify_password(_dummy_hash, password)
            return False, "Institute ID, login ID, or password is incorrect."
        if user.locked_until and user.locked_until <= now:
            user.failed_login_count = 0
            user.locked_until = None

        if not _verify_password(user.password_hash, password):
            user.failed_login_count += 1
            if user.failed_login_count >= _MAX_FAILURES:
                user.locked_until = now + timedelta(minutes=_LOCK_MINUTES)
            session.commit()
            return False, "Institute ID, login ID, or password is incorrect."

        user.failed_login_count = 0
        user.locked_until = None
        session.add(
            AuditLog(
                institute_id=cleaned_id,
                actor_user_id=user.id,
                action="auth.signed_in",
                object_type="user",
                object_id=user.id,
                details={},
            )
        )
        session.commit()
        return True, user.id


def get_active_user(user_id: str) -> dict[str, Any] | None:
    with session_scope() as session:
        row = session.execute(
            select(User, Institute.name.label("institute_name"))
            .join(Institute, Institute.id == User.institute_id)
            .where(User.id == user_id, User.is_active.is_(True))
        ).first()
        if row is None:
            return None
        user = row[0]
        return {
            "id": user.id,
            "institute_id": user.institute_id,
            "institute_name": row.institute_name,
            "login_id": user.login_id,
            "display_name": user.display_name,
            "role": user.role,
            "linked_student_id": user.linked_student_id,
            "must_change_password": bool(user.must_change_password),
        }


def list_accounts(institute_id: str) -> list[dict[str, Any]]:
    with session_scope() as session:
        users = session.scalars(
            select(User)
            .where(User.institute_id == institute_id)
            .order_by(User.role, User.display_name)
        ).all()
        return [
            {
                "id": user.id,
                "login_id": user.login_id,
                "display_name": user.display_name,
                "role": user.role,
                "linked_student_id": user.linked_student_id,
                "is_active": user.is_active,
                "created_at": user.created_at,
            }
            for user in users
        ]


def create_account(
    actor_id: str,
    role: str,
    display_name: str,
    login_id: str,
    password: str,
    linked_student_id: str | None = None,
) -> tuple[bool, str]:
    try:
        clean_role = role.strip().title()
        clean_name = display_name.strip()
        clean_login = normalize_login_id(login_id)
        password_hash = _hash_password(password)
        if clean_role not in {"Owner", "Teacher", "Student", "Parent"}:
            raise AccountError("Choose a supported account role.")
        if not clean_name or len(clean_name) > 160:
            raise AccountError("Name is required and must be under 160 characters.")
        if clean_role in {"Student", "Parent"} and not linked_student_id:
            raise AccountError("Select the student linked to this account.")

        with session_scope() as session:
            actor = session.get(User, actor_id)
            if actor is None or not actor.is_active or actor.role != "Owner":
                return False, "Only an active institute owner can manage accounts."
            if linked_student_id and not session.get(
                Student, (actor.institute_id, linked_student_id)
            ):
                return False, "That student does not exist in this institute."
            user = User(
                id=str(uuid.uuid4()),
                institute_id=actor.institute_id,
                login_id=clean_login,
                display_name=clean_name,
                role=clean_role,
                password_hash=password_hash,
                linked_student_id=linked_student_id or None,
                is_active=True,
                must_change_password=True,
            )
            session.add(user)
            session.flush()
            session.add(
                AuditLog(
                    institute_id=actor.institute_id,
                    actor_user_id=actor_id,
                    action="account.created",
                    object_type="user",
                    object_id=user.id,
                    details={"role": clean_role},
                )
            )
            session.commit()
            return True, user.id
    except AccountError as exc:
        return False, str(exc)
    except IntegrityError:
        return False, "That login ID is already in use at this institute."



def create_student_accounts_bulk(
    actor_id: str,
    rows: list[dict[str, str]],
) -> tuple[bool, str | list[dict[str, str]]]:
    """Create student accounts as one all-or-nothing operation.

    Input rows must reference existing student records. `login_id` may be blank,
    in which case the canonical student_id is used. Generated temporary passwords
    are returned once so the Owner can download and distribute them securely.
    """
    if not rows:
        return False, "The CSV contains no student rows."

    cleaned_rows: list[dict[str, str]] = []
    seen_students: set[str] = set()
    seen_logins: set[str] = set()
    for index, row in enumerate(rows, start=2):
        student_id = str(row.get("student_id", "")).strip()
        raw_login = str(row.get("login_id", "")).strip()
        if not student_id:
            return False, f"Row {index}: student_id is required."
        if student_id in seen_students:
            return False, f"Row {index}: duplicate student_id {student_id!r}."
        seen_students.add(student_id)
        try:
            login_id = normalize_login_id(raw_login or student_id)
        except AccountError as exc:
            return False, f"Row {index}: {exc}"
        if login_id in seen_logins:
            return False, f"Row {index}: duplicate login_id {login_id!r}."
        seen_logins.add(login_id)
        cleaned_rows.append({"student_id": student_id, "login_id": login_id})

    try:
        with session_scope() as session:
            actor = session.get(User, actor_id)
            if actor is None or not actor.is_active or actor.role != "Owner":
                return False, "Only an active institute owner can manage accounts."

            student_ids = [row["student_id"] for row in cleaned_rows]
            students = session.scalars(
                select(Student).where(
                    Student.institute_id == actor.institute_id,
                    Student.student_id.in_(student_ids),
                )
            ).all()
            students_by_id = {student.student_id: student for student in students}
            missing = [sid for sid in student_ids if sid not in students_by_id]
            if missing:
                preview = ", ".join(missing[:10])
                suffix = " …" if len(missing) > 10 else ""
                return False, (
                    "Every student_id must already exist in this institute's Students "
                    f"dataset. Unknown ID(s): {preview}{suffix}"
                )

            existing_by_login = set(
                session.scalars(
                    select(User.login_id).where(
                        User.institute_id == actor.institute_id,
                        User.login_id.in_(list(seen_logins)),
                    )
                ).all()
            )
            if existing_by_login:
                return False, "Login ID(s) already exist: " + ", ".join(sorted(existing_by_login)[:10])

            existing_linked = set(
                session.scalars(
                    select(User.linked_student_id).where(
                        User.institute_id == actor.institute_id,
                        User.role == "Student",
                        User.linked_student_id.in_(student_ids),
                    )
                ).all()
            )
            if existing_linked:
                return False, (
                    "Student account(s) already exist for: "
                    + ", ".join(sorted(existing_linked)[:10])
                    + ". Remove those rows from the CSV and retry."
                )

            credentials: list[dict[str, str]] = []
            for row in cleaned_rows:
                student_id = row["student_id"]
                student = students_by_id[student_id]
                temporary_password = secrets.token_urlsafe(18)
                user = User(
                    id=str(uuid.uuid4()),
                    institute_id=actor.institute_id,
                    login_id=row["login_id"],
                    display_name=student.name,
                    role="Student",
                    password_hash=_hash_password(temporary_password),
                    linked_student_id=student_id,
                    is_active=True,
                    must_change_password=True,
                )
                session.add(user)
                session.add(
                    AuditLog(
                        institute_id=actor.institute_id,
                        actor_user_id=actor_id,
                        action="account.created",
                        object_type="user",
                        object_id=user.id,
                        details={"role": "Student", "source": "bulk_csv", "student_id": student_id},
                    )
                )
                credentials.append(
                    {
                        "institute_id": actor.institute_id,
                        "student_id": student_id,
                        "display_name": student.name,
                        "login_id": row["login_id"],
                        "temporary_password": temporary_password,
                    }
                )
            session.commit()
            return True, credentials
    except IntegrityError:
        return False, "A login ID or account link conflicts with an existing account. No accounts were created."
    except AccountError as exc:
        return False, str(exc)


def reset_account_password(
    actor_id: str, target_user_id: str, new_password: str
) -> tuple[bool, str]:
    try:
        password_hash = _hash_password(new_password)
    except AccountError as exc:
        return False, str(exc)
    with session_scope() as session:
        actor = session.get(User, actor_id)
        target = session.get(User, target_user_id)
        if actor is None or not actor.is_active or actor.role != "Owner":
            return False, "Only an active institute owner can reset passwords."
        if target is None or target.institute_id != actor.institute_id:
            return False, "Account not found in this institute."
        target.password_hash = password_hash
        target.must_change_password = target.id != actor.id
        target.failed_login_count = 0
        target.locked_until = None
        session.add(
            AuditLog(
                institute_id=actor.institute_id,
                actor_user_id=actor_id,
                action="account.password_reset",
                object_type="user",
                object_id=target.id,
                details={},
            )
        )
        session.commit()
        return True, "Password reset. Share the new password with the account holder securely."


def set_account_active(
    actor_id: str, target_user_id: str, is_active: bool
) -> tuple[bool, str]:
    with session_scope() as session:
        actor = session.get(User, actor_id)
        target = session.get(User, target_user_id)
        if actor is None or not actor.is_active or actor.role != "Owner":
            return False, "Only an active institute owner can manage accounts."
        if target is None or target.institute_id != actor.institute_id:
            return False, "Account not found in this institute."
        if not is_active and target.id == actor.id:
            return False, "You cannot deactivate the account you are using."
        if not is_active and target.role == "Owner" and target.is_active:
            active_owners = session.scalar(
                select(func.count())
                .select_from(User)
                .where(
                    User.institute_id == actor.institute_id,
                    User.role == "Owner",
                    User.is_active.is_(True),
                )
            )
            if active_owners <= 1:
                return False, "An institute must keep at least one active owner."
        target.is_active = is_active
        if is_active:
            target.failed_login_count = 0
            target.locked_until = None
        session.add(
            AuditLog(
                institute_id=actor.institute_id,
                actor_user_id=actor_id,
                action="account.activated" if is_active else "account.deactivated",
                object_type="user",
                object_id=target.id,
                details={},
            )
        )
        session.commit()
        return True, "Account activated." if is_active else "Account deactivated."


def deactivate_account(actor_id: str, target_user_id: str) -> tuple[bool, str]:
    return set_account_active(actor_id, target_user_id, False)



def change_own_password(
    user_id: str, current_password: str, new_password: str
) -> tuple[bool, str]:
    """Let a signed-in user replace their password (also clears the forced-change flag)."""
    try:
        new_hash = _hash_password(new_password)
    except AccountError as exc:
        return False, str(exc)
    if current_password == new_password:
        return False, "Choose a new password that differs from the temporary one."
    with session_scope() as session:
        user = session.get(User, user_id)
        if user is None or not user.is_active:
            return False, "Account not found."
        if not _verify_password(user.password_hash, current_password):
            return False, "The current password is incorrect."
        user.password_hash = new_hash
        user.must_change_password = False
        session.add(
            AuditLog(
                institute_id=user.institute_id,
                actor_user_id=user.id,
                action="account.password_changed",
                object_type="user",
                object_id=user.id,
                details={},
            )
        )
        session.commit()
        return True, "Password updated."


def preview_student_accounts_bulk(
    actor_id: str, rows: list[dict[str, str]]
) -> list[str]:
    """Dry run: return every row-level problem at once. Writes nothing."""
    errors: list[str] = []
    if not rows:
        return ["The CSV contains no student rows."]
    seen_students: set[str] = set()
    seen_logins: set[str] = set()
    cleaned: list[tuple[int, str, str]] = []
    for index, row in enumerate(rows, start=2):
        student_id = str(row.get("student_id", "")).strip()
        if not student_id:
            errors.append(f"Row {index}: student_id is required.")
            continue
        if student_id in seen_students:
            errors.append(f"Row {index}: duplicate student_id {student_id!r}.")
        seen_students.add(student_id)
        try:
            login = normalize_login_id(str(row.get("login_id", "")).strip() or student_id)
        except AccountError as exc:
            errors.append(f"Row {index}: {exc}")
            continue
        if login in seen_logins:
            errors.append(f"Row {index}: duplicate login_id {login!r}.")
        seen_logins.add(login)
        cleaned.append((index, student_id, login))
    with session_scope() as session:
        actor = session.get(User, actor_id)
        if actor is None or not actor.is_active or actor.role != "Owner":
            return ["Only an active institute owner can manage accounts."]
        known = set(
            session.scalars(
                select(Student.student_id).where(
                    Student.institute_id == actor.institute_id,
                    Student.student_id.in_([c[1] for c in cleaned]),
                )
            ).all()
        )
        taken_logins = set(
            session.scalars(
                select(User.login_id).where(
                    User.institute_id == actor.institute_id,
                    User.login_id.in_([c[2] for c in cleaned]),
                )
            ).all()
        )
        has_account = set(
            session.scalars(
                select(User.linked_student_id).where(
                    User.institute_id == actor.institute_id,
                    User.role == "Student",
                    User.linked_student_id.in_([c[1] for c in cleaned]),
                )
            ).all()
        )
    for index, student_id, login in cleaned:
        if student_id not in known:
            errors.append(f"Row {index}: student_id {student_id!r} is not in this institute's roster.")
        if student_id in has_account:
            errors.append(f"Row {index}: {student_id!r} already has a Student account.")
        if login in taken_logins:
            errors.append(f"Row {index}: login_id {login!r} is already in use.")
    return errors
