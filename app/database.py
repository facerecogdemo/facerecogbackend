import os

import psycopg
from psycopg.types.json import Jsonb


def get_database_url():
    url = os.environ.get("DATABASE_URL")

    if not url:
        raise RuntimeError(
            "DATABASE_URL environment variable is not set"
        )

    # Some hosting providers use the older postgres:// prefix.
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]

    return url


def get_connection():
    return psycopg.connect(get_database_url())


def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                id BIGSERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                role VARCHAR(20) NOT NULL DEFAULT 'user'
                    CHECK (role IN ('admin', 'user')),
                is_active BOOLEAN NOT NULL DEFAULT TRUE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS face_images (
                id BIGSERIAL PRIMARY KEY,
                person_id BIGINT NOT NULL
                    REFERENCES persons(id) ON DELETE CASCADE,
                image BYTEA NOT NULL,
                embedding JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_face_images_person_id
            ON face_images(person_id)
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id BIGSERIAL PRIMARY KEY,
                person_id BIGINT NOT NULL
                    REFERENCES persons(id) ON DELETE CASCADE,
                attendance_date DATE NOT NULL DEFAULT CURRENT_DATE,
                recognized_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE (person_id, attendance_date)
            )
        """)


def register_person(name, records, role):
    with get_connection() as conn:
        row = conn.execute(
            """
            INSERT INTO persons (name, role)
            VALUES (%s, %s)
            RETURNING id
            """,
            (name,role),
        ).fetchone()

        person_id = row[0]

        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO face_images (person_id, image, embedding)
                VALUES (%s, %s, %s)
                """,
                [
                    (person_id, image, Jsonb(embedding))
                    for image, embedding in records
                ],
            )

        return person_id

def get_registered_faces():
    with get_connection() as conn:
        return conn.execute("""
            SELECT p.id, p.name, p.role, f.embedding
            FROM persons p
            JOIN face_images f ON f.person_id = p.id
            WHERE p.is_active = TRUE
            ORDER BY p.id, f.id
        """).fetchall()

def get_person_by_id(person_id):
    with get_connection() as conn:
        row = conn.execute("""
            SELECT id, name, role, is_active, created_at
            FROM persons
            WHERE id = %s
        """, (person_id,)).fetchone()

    if not row:
        return None

    return {
        "id": row[0],
        "name": row[1],
        "role": row[2],
        "is_active": row[3],
        "created_at": row[4].isoformat(),
    }

def get_all_users():
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT id, name, role, is_active, created_at
            FROM persons
            ORDER BY id
        """).fetchall()

    return [
        {
            "id": row[0],
            "name": row[1],
            "role": row[2],
            "is_active": row[3],
            "created_at": row[4].isoformat(),
        }
        for row in rows
    ]


def count_active_admins():
    with get_connection() as conn:
        row = conn.execute("""
            SELECT COUNT(*)
            FROM persons
            WHERE role = 'admin'
              AND is_active = TRUE
        """).fetchone()

    return row[0]


def update_user_role(person_id, role):
    if role not in ("admin", "user"):
        raise ValueError("Invalid role")

    with get_connection() as conn:
        row = conn.execute("""
            UPDATE persons
            SET role = %s
            WHERE id = %s
            RETURNING id
        """, (role, person_id)).fetchone()

    return row is not None


def update_user_status(person_id, is_active):
    with get_connection() as conn:
        row = conn.execute("""
            UPDATE persons
            SET is_active = %s
            WHERE id = %s
            RETURNING id
        """, (is_active, person_id)).fetchone()

    return row is not None


def delete_person(person_id):
    # Existing foreign keys use ON DELETE CASCADE:
    # deleting a person also deletes their face images and attendance.
    with get_connection() as conn:
        row = conn.execute("""
            DELETE FROM persons
            WHERE id = %s
            RETURNING id
        """, (person_id,)).fetchone()

    return row is not None

def record_attendance(person_id):
    with get_connection() as conn:
        row = conn.execute(
            """
            INSERT INTO attendance (person_id)
            VALUES (%s)
            ON CONFLICT (person_id, attendance_date)
            DO NOTHING
            RETURNING id
            """,
            (person_id,),
        ).fetchone()

        return row is not None


def get_attendance(user=None, date=None):
    query = """
        SELECT
            p.id,
            p.name,
            a.attendance_date,
            a.recognized_at
        FROM attendance a
        JOIN persons p ON p.id = a.person_id
        WHERE TRUE
    """

    params = []

    if user:
        query += " AND p.name ILIKE %s"
        params.append(f"%{user}%")

    if date:
        query += " AND a.attendance_date = %s"
        params.append(date)

    query += " ORDER BY a.recognized_at DESC"

    with get_connection() as conn:
        return conn.execute(query, params).fetchall()
