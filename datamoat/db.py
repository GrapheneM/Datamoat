from __future__ import annotations

import os
from contextlib import contextmanager
from functools import lru_cache
from typing import Iterator

from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.engine import make_url

load_dotenv()


def get_database_url() -> str:
    value = os.getenv("DATABASE_URL", "").strip()
    if not value:
        host = os.getenv("PGHOST")
        user = os.getenv("PGUSER")
        password = os.getenv("PGPASSWORD")
        database = os.getenv("PGDATABASE")
        if host and user and password and database:
            port = os.getenv("PGPORT", "5432")
            value = f"postgresql://{user}:{password}@{host}:{port}/{database}"
    if not value:
        raise RuntimeError(
            "PostgreSQL is not configured. Set DATABASE_URL in your environment."
        )
    if value.startswith("postgres://"):
        value = "postgresql://" + value[len("postgres://") :]
    parsed = make_url(value)
    if parsed.drivername in {"postgres", "postgresql"}:
        value = parsed.set(drivername="postgresql+psycopg").render_as_string(
            hide_password=False
        )
    return value


@lru_cache(maxsize=2)
def get_engine(database_url: str | None = None) -> Engine:
    return create_engine(
        database_url or get_database_url(),
        pool_pre_ping=True,
        pool_recycle=1800,
        future=True,
    )


@lru_cache(maxsize=2)
def get_session_factory(database_url: str | None = None) -> sessionmaker[Session]:
    return sessionmaker(
        bind=get_engine(database_url),
        autoflush=False,
        expire_on_commit=False,
    )


@contextmanager
def session_scope(database_url: str | None = None) -> Iterator[Session]:
    session = get_session_factory(database_url)()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_database() -> None:
    with get_engine().connect() as connection:
        connection.execute(text("SELECT 1"))
