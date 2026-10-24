# DataMoat

DataMoat is a Python and Streamlit institute intelligence app backed by PostgreSQL. It includes institute-scoped accounts, academic data imports, topic mastery, supportive early-warning indicators, historically calibrated percentile ranges, parent reports, and owner-only fee summaries.

## Run in Replit

The app starts through the **DataMoat Streamlit** workflow. It uses the project's PostgreSQL `DATABASE_URL` and the `SESSION_SECRET` secret for Streamlit's signed session cookie.

On a fresh database, run the schema setup once:

```bash
uv run python -m datamoat.migrate
```

Then start the workflow. Open **Create an institute** to create the first Owner account; no demo credentials are enabled.

## Run locally

1. Install PostgreSQL and create an empty database named `datamoat`.
2. Copy `.env.example` to `.env` and set `DATABASE_URL` and a long random `SESSION_SECRET`.
3. Install Python 3.12 or newer and [uv](https://docs.astral.sh/uv/), then run:

```bash
uv sync
uv run python -m datamoat.migrate
STREAMLIT_SERVER_COOKIE_SECRET="$SESSION_SECRET" uv run streamlit run app.py --server.address 127.0.0.1 --server.port 5000
```

Load the `.env` values into your shell before these commands. The app does not fall back to SQLite; it reports a configuration error if PostgreSQL is unavailable.

## Import order

Owners should import the CSV or `.xlsx` files in this order:

1. Students
2. Questions
3. Tests
4. Test responses
5. Optional attendance, fee instalments, and engagement

Use the in-app templates. Passwords and credential columns are intentionally not accepted in data uploads. Student and parent accounts are managed separately from academic records.

## Access boundaries

- Owners manage data and accounts.
- Teachers can view institute academic summaries but cannot access fee records or change data.
- Student and parent accounts are scoped to their linked student.
- Parent reports are withheld unless parent-contact consent is recorded.
- Replacing a table is transactional and may be rejected when other records depend on it.

This app provides access controls and data validation, not a legal compliance certification.
