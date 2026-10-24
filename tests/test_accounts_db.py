"""Database tests. Skipped unless TEST_DATABASE_URL points at a DISPOSABLE PostgreSQL database."""
import os, uuid
import pytest

URL = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL not set")

@pytest.fixture()
def env(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", URL)
    from datamoat import db, auth
    db.get_engine.cache_clear(); db.get_session_factory.cache_clear()
    from datamoat.models import Base, Student
    Base.metadata.create_all(db.get_engine())
    inst = auth.normalize_institute_id("t" + uuid.uuid4().hex[:8])
    ok, owner = auth.create_institute_owner(inst, "Test Institute", "owner", "Owner", "correct-horse-battery-1")
    assert ok, owner
    with db.session_scope() as s:
        for i in range(3):
            s.add(Student(institute_id=inst, student_id=f"S{i}", name=f"N{i}", batch_id="B", join_date="2025-01-01", cohort="c"))
        s.commit()
    return auth, db, inst, owner

def test_bulk_creates_hashed_accounts_and_forces_change(env):
    auth, db, inst, owner = env
    ok, creds = auth.create_student_accounts_bulk(owner, [{"student_id": "S0", "login_id": ""}, {"student_id": "S1", "login_id": "x1"}])
    assert ok and len(creds) == 2
    from datamoat.models import User
    from sqlalchemy import select
    with db.session_scope() as s:
        users = s.scalars(select(User).where(User.institute_id == inst, User.role == "Student")).all()
        assert all(u.must_change_password for u in users)
        assert all(c["temporary_password"] not in u.password_hash for c in creds for u in users)

def test_bulk_rolls_back_on_bad_row(env):
    auth, db, inst, owner = env
    ok, msg = auth.create_student_accounts_bulk(owner, [{"student_id": "S0", "login_id": ""}, {"student_id": "NOPE", "login_id": ""}])
    assert not ok
    assert all(a["role"] != "Student" for a in auth.list_accounts(inst))

def test_preview_reports_all_errors_and_duplicates_rejected(env):
    auth, db, inst, owner = env
    errs = auth.preview_student_accounts_bulk(owner, [{"student_id": "S0", "login_id": "a"}, {"student_id": "S0", "login_id": "b"}, {"student_id": "ZZ", "login_id": ""}])
    assert len(errs) >= 2

def test_change_password_clears_flag(env):
    auth, db, inst, owner = env
    ok, creds = auth.create_student_accounts_bulk(owner, [{"student_id": "S2", "login_id": ""}])
    uid = [a["id"] for a in auth.list_accounts(inst) if a["role"] == "Student"][0]
    assert not auth.change_own_password(uid, "wrong", "a-new-long-password")[0]
    assert auth.change_own_password(uid, creds[0]["temporary_password"], "a-new-long-password")[0]
    assert auth.get_active_user(uid)["must_change_password"] is False
