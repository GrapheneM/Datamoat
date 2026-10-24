"""Create or update the local development schema. Run manually, never at app startup."""

from sqlalchemy import text

from .db import get_engine
from .models import Base


def main() -> None:
    engine = get_engine()
    Base.metadata.create_all(engine)
    # create_all never alters existing tables, so add newer columns explicitly.
    with engine.begin() as connection:
        connection.execute(
            text(
                "ALTER TABLE dm_users ADD COLUMN IF NOT EXISTS "
                "must_change_password BOOLEAN NOT NULL DEFAULT FALSE"
            )
        )
        connection.execute(
            text(
                "ALTER TABLE dm_students ADD COLUMN IF NOT EXISTS "
                "dropout_date DATE"
            )
        )
        connection.execute(
            text(
                """
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_constraint
                        WHERE conname = 'ck_dm_students_dropout_date'
                    ) THEN
                        ALTER TABLE dm_students
                        ADD CONSTRAINT ck_dm_students_dropout_date
                        CHECK (dropout_date IS NULL OR dropout_date >= join_date);
                    END IF;
                END $$;
                """
            )
        )
    print("DataMoat schema is ready.")


if __name__ == "__main__":
    main()
