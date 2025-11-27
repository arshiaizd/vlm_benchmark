from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import sqlite3
import json
from datetime import datetime


class ResultCache:
    """
    SQLite-backed cache so we can resume experiments without re-calling models.

    Keyed by (sample_id, experiment_name, model_name).
    Stores an arbitrary JSON payload per combination.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS results (
                sample_id TEXT NOT NULL,
                experiment_name TEXT NOT NULL,
                model_name TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (sample_id, experiment_name, model_name)
            )
            """
        )
        self.conn.commit()

    def has(self, sample_id: str, experiment_name: str, model_name: str) -> bool:
        cur = self.conn.execute(
            """
            SELECT 1 FROM results
            WHERE sample_id = ? AND experiment_name = ? AND model_name = ?
            """,
            (sample_id, experiment_name, model_name),
        )
        return cur.fetchone() is not None

    def get(
        self, sample_id: str, experiment_name: str, model_name: str
    ) -> Optional[Dict[str, Any]]:
        cur = self.conn.execute(
            """
            SELECT payload_json FROM results
            WHERE sample_id = ? AND experiment_name = ? AND model_name = ?
            """,
            (sample_id, experiment_name, model_name),
        )
        row = cur.fetchone()
        if not row:
            return None
        return json.loads(row[0])

    def set(
        self,
        sample_id: str,
        experiment_name: str,
        model_name: str,
        payload: Dict[str, Any],
    ) -> None:
        payload_json = json.dumps(payload, ensure_ascii=False)
        created_at = datetime.utcnow().isoformat(timespec="seconds")
        self.conn.execute(
            """
            INSERT OR REPLACE INTO results
            (sample_id, experiment_name, model_name, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (sample_id, experiment_name, model_name, payload_json, created_at),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
