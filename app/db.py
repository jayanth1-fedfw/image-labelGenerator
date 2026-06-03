import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

# ─── Connection ───────────────────────────────────────────────

def get_connection():
    """
    Returns a psycopg2 connection.
    Supports both local (host/port) and Cloud Run (Unix socket via Cloud SQL proxy).
    """
    db_host = os.getenv("DB_HOST", "127.0.0.1")
    db_name = os.getenv("DB_NAME", "imagelabels")
    db_user = os.getenv("DB_USER", "postgres")
    db_pass = os.getenv("DB_PASS", "")
    db_port = os.getenv("DB_PORT", "5432")

    # Cloud Run uses Unix socket path like /cloudsql/project:region:instance
    if db_host.startswith("/"):
        conn = psycopg2.connect(
            host=db_host,
            database=db_name,
            user=db_user,
            password=db_pass
        )
    else:
        conn = psycopg2.connect(
            host=db_host,
            port=int(db_port),
            database=db_name,
            user=db_user,
            password=db_pass
        )

    return conn


# ─── Schema Init ──────────────────────────────────────────────

def init_db():
    """
    Creates required tables if they don't exist.
    Safe to call on every app startup.
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS image_labels (
            id          SERIAL PRIMARY KEY,
            filename    TEXT        NOT NULL,
            gcs_url     TEXT        NOT NULL,
            labels      JSONB       NOT NULL,
            mode        TEXT        DEFAULT 'labels',
            created_at  TIMESTAMP   DEFAULT NOW()
        )
    """)

    # Index for fast history queries
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_image_labels_created_at
        ON image_labels (created_at DESC)
    """)

    # Index for filename search
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_image_labels_filename
        ON image_labels (filename)
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("✅ Database initialized successfully.")


# ─── Save Result ──────────────────────────────────────────────

def save_result(filename, gcs_url, labels, mode="labels"):
    """
    Insert a new image analysis result into the database.

    Args:
        filename (str): Original image filename
        gcs_url  (str): Public GCS URL of the uploaded image
        labels   (list|dict): JSON-serializable analysis result
        mode     (str): Detection mode — 'labels', 'objects', 'properties', 'full'

    Returns:
        int: ID of the newly inserted row
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO image_labels (filename, gcs_url, labels, mode)
        VALUES (%s, %s, %s, %s)
        RETURNING id
        """,
        (filename, gcs_url, json.dumps(labels), mode)
    )

    new_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()

    print(f"✅ Saved result ID={new_id} | file={filename} | mode={mode}")
    return new_id


# ─── Get All Results ──────────────────────────────────────────

def get_all_results(limit=50):
    """
    Fetch recent image analysis results ordered by newest first.

    Args:
        limit (int): Max number of records to return (default 50)

    Returns:
        list of tuples: (id, filename, gcs_url, labels, mode, created_at)
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT id, filename, gcs_url, labels, mode, created_at
        FROM image_labels
        ORDER BY created_at DESC
        LIMIT %s
        """,
        (limit,)
    )

    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


# ─── Get Single Result ────────────────────────────────────────

def get_result_by_id(record_id):
    """
    Fetch a single analysis result by its ID.

    Args:
        record_id (int): Row ID to fetch

    Returns:
        tuple or None: (id, filename, gcs_url, labels, mode, created_at)
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT id, filename, gcs_url, labels, mode, created_at
        FROM image_labels
        WHERE id = %s
        """,
        (record_id,)
    )

    row = cur.fetchone()
    cur.close()
    conn.close()
    return row


# ─── Search by Filename ───────────────────────────────────────

def search_by_filename(keyword):
    """
    Search results by partial filename match.

    Args:
        keyword (str): Partial filename to search

    Returns:
        list of tuples
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT id, filename, gcs_url, labels, mode, created_at
        FROM image_labels
        WHERE filename ILIKE %s
        ORDER BY created_at DESC
        """,
        (f"%{keyword}%",)
    )

    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


# ─── Search by Label ──────────────────────────────────────────

def search_by_label(label_keyword):
    """
    Search results where labels JSONB contains a matching label string.

    Args:
        label_keyword (str): Label text to search for

    Returns:
        list of tuples
    """
    conn = get_connection()
    cur = conn.cursor()

    # JSONB text search — works for both list of dicts and nested structures
    cur.execute(
        """
        SELECT id, filename, gcs_url, labels, mode, created_at
        FROM image_labels
        WHERE labels::text ILIKE %s
        ORDER BY created_at DESC
        """,
        (f"%{label_keyword}%",)
    )

    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


# ─── Delete Result ────────────────────────────────────────────

def delete_result(record_id):
    """
    Delete an analysis record by ID.

    Args:
        record_id (int): Row ID to delete

    Returns:
        bool: True if deleted, False if not found
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "DELETE FROM image_labels WHERE id = %s RETURNING id",
        (record_id,)
    )

    deleted = cur.fetchone()
    conn.commit()
    cur.close()
    conn.close()

    if deleted:
        print(f"🗑️ Deleted record ID={record_id}")
        return True
    return False


# ─── Stats ────────────────────────────────────────────────────

def get_stats():
    """
    Returns summary statistics from the database.

    Returns:
        dict: total_images, mode_breakdown, latest_upload
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM image_labels")
    total = cur.fetchone()[0]

    cur.execute(
        "SELECT mode, COUNT(*) FROM image_labels GROUP BY mode ORDER BY COUNT(*) DESC"
    )
    mode_rows = cur.fetchall()
    mode_breakdown = {row[0]: row[1] for row in mode_rows}

    cur.execute(
        "SELECT filename, created_at FROM image_labels ORDER BY created_at DESC LIMIT 1"
    )
    latest = cur.fetchone()

    cur.close()
    conn.close()

    return {
        "total_images": total,
        "mode_breakdown": mode_breakdown,
        "latest_upload": {
            "filename": latest[0] if latest else None,
            "created_at": str(latest[1]) if latest else None
        }
    }