from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import csv
import json


class ResultCache:
    """
    Very simple CSV-backed cache so we can resume experiments without
    re-calling models.

    Internally keeps everything in memory as a dict keyed by
    (sample_id, experiment_name, model_name), and rewrites the whole
    CSV file on each write. This is intentionally simple and explicit.
    """

    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)
        self._data: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self._load()

    # --- internal helpers -------------------------------------------------

    def _load(self) -> None:
        if not self.csv_path.exists():
            return

        with self.csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sample_id = row.get("sample_id")
                experiment_name = row.get("experiment_name")
                model_name = row.get("model_name")
                payload_json = row.get("payload_json", "")

                if not (sample_id and experiment_name and model_name):
                    continue

                key = (sample_id, experiment_name, model_name)
                try:
                    payload = json.loads(payload_json)
                except Exception:
                    payload = {"raw_payload_json": payload_json}
                self._data[key] = payload

    def _write_all(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["sample_id", "experiment_name", "model_name", "payload_json"]
        with self.csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for (sample_id, experiment_name, model_name), payload in self._data.items():
                writer.writerow(
                    {
                        "sample_id": sample_id,
                        "experiment_name": experiment_name,
                        "model_name": model_name,
                        "payload_json": json.dumps(payload, ensure_ascii=False),
                    }
                )

    # --- public API (same as before) --------------------------------------

    def has(self, sample_id: str, experiment_name: str, model_name: str) -> bool:
        key = (sample_id, experiment_name, model_name)
        return key in self._data

    def get(
        self, sample_id: str, experiment_name: str, model_name: str
    ) -> Optional[Dict[str, Any]]:
        key = (sample_id, experiment_name, model_name)
        return self._data.get(key)

    def set(
        self,
        sample_id: str,
        experiment_name: str,
        model_name: str,
        payload: Dict[str, Any],
    ) -> None:
        key = (sample_id, experiment_name, model_name)
        self._data[key] = payload
        self._write_all()

    def close(self) -> None:
        # Nothing special to do for a CSV file.
        pass
