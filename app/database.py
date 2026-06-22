import sqlite3
import json
from pathlib import Path
from loguru import logger
import threading

DB_PATH = Path("data/reports.db")
_db_lock = threading.Lock()

def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=15.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

def init_db():
    DB_PATH.parent.mkdir(exist_ok=True)
    with _db_lock:
        with _get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS product_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    asin TEXT NOT NULL,
                    product_name TEXT NOT NULL,
                    url TEXT,
                    quality_score REAL NOT NULL,
                    grade TEXT NOT NULL,
                    n_reviews INTEGER NOT NULL,
                    breakdown TEXT NOT NULL,
                    flags TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    full_report TEXT,
                    actual_rating REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            try:
                conn.execute("ALTER TABLE product_reports ADD COLUMN full_report TEXT;")
                conn.commit()
            except sqlite3.OperationalError:
                pass

            try:
                conn.execute("ALTER TABLE product_reports ADD COLUMN actual_rating REAL;")
                conn.commit()
            except sqlite3.OperationalError:
                pass
                
            conn.commit()
    logger.info(f"SQLite Database initialized at {DB_PATH}")

def save_report(asin: str, name: str, url: str, score: float, grade: str, n_reviews: int, breakdown: dict, flags: list, summary: str, full_report: dict, actual_rating: float):
    try:
        with _db_lock:
            with _get_connection() as conn:
                conn.execute(
                    "INSERT INTO product_reports (asin, product_name, url, quality_score, grade, n_reviews, breakdown, flags, summary, full_report, actual_rating) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (asin, name, url, score, grade, n_reviews, json.dumps(breakdown), json.dumps(flags), summary, json.dumps(full_report), actual_rating)
                )
                conn.commit()
        logger.info(f"Analysis report saved to SQLite for ASIN: {asin}")
    except Exception as e:
        logger.error(f"Failed to save report to SQLite: {e}")

def get_history(limit: int = 10):
    try:
        if not DB_PATH.exists():
            return []
        with _db_lock:
            with _get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                cur.execute("SELECT * FROM product_reports ORDER BY timestamp DESC LIMIT ?", (limit,))
                rows = cur.fetchall()
                return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Failed to fetch report history: {e}")
        return []