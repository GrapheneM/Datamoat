# DataMoat Institute Intelligence Platform

DataMoat is a Python/Streamlit application for institute-scoped academic analytics, student support, parent reports, data imports, and owner-only fee summaries.

## Run & Operate

- **DataMoat Streamlit** workflow — run the app on port 5000
- `uv run python -m datamoat.migrate` — create the PostgreSQL schema manually (never run DDL at app startup)
- `STREAMLIT_SERVER_COOKIE_SECRET="$SESSION_SECRET" uv run streamlit run app.py --server.port 5000 --server.address 0.0.0.0` — run outside the managed workflow
- Required environment: `DATABASE_URL`, `SESSION_SECRET`
- `pnpm run typecheck` — typecheck the retained workspace scaffolding

## Stack

- Python 3.12+, Streamlit, SQLAlchemy, PostgreSQL/psycopg
- Pandas and NumPy for validated imports and analytics
- Argon2 password hashing for institute-managed accounts

## Where things live

- `app.py` — Streamlit sign-in, navigation, and page rendering
- `datamoat/models.py` — PostgreSQL tables and relational constraints
- `datamoat/auth.py` — institute registration, account authentication, and role checks
- `datamoat/data_access.py` — tenant-scoped reads, imports, and audit logging
- `datamoat/validation.py` — spreadsheet parsing, templates, and row validation
- `datamoat/analytics.py` — mastery, risk indicators, calibrated ranges, and fee summaries
- `datamoat/migrate.py` — explicit schema setup command
- `README.md` — local PostgreSQL setup and app usage

## Architecture decisions

- Keep the product UI, authentication, data access, and analytics in Python/Streamlit as requested.
- Store institute accounts and academic records in PostgreSQL; never use browser storage or demo credentials for authentication.
- Treat account role, institute membership, and linked student identity as server-side database facts.
- Keep fee records Owner-only and keep student/parent queries restricted to their linked student.
- Do not invent percentile predictions: require historical outcomes for calibration and show a clear unavailable state until enough data exists.

## Product

Owners can manage accounts and load institute data. Owners and teachers can review academic trends; students and parents get private linked-student views. Parent reports require recorded contact consent.

## User preferences

- The app should remain Python/Streamlit with PostgreSQL and be runnable locally.

## Gotchas

- Run `uv run python -m datamoat.migrate` once before the first app launch on a fresh database.
- Never put schema creation in the Streamlit startup path.
- Import students, questions, and tests before importing test responses.
- DataMoat is served by the dedicated Streamlit workflow on port 5000; it is not registered as an artifact route.
- NumPy wheels in this Replit image need the Nix GCC runtime library on `LD_LIBRARY_PATH`; the managed workflow sets it when starting Streamlit.

## Pointers

- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details.
