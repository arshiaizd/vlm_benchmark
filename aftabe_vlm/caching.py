import json
from pathlib import Path
from typing import Dict, Tuple, Any, Optional


class ResultCache:
    """
    Result cache backed by a JSONL file.

    Each line in the file is a JSON object like:
    {
      "sample_id": "...",
      "dataset_name": "...",
      "experiment_name": "...",
      "model_name": "...",
      "payload": { ... result_record ... }
    }

    In-memory, we index by (sample_id, dataset_name, experiment_name, model_name).
    """

    def __init__(self, jsonl_path: str | Path):
        self.jsonl_path = Path(jsonl_path)
        # key: (sample_id, dataset_name, experiment_name, model_name)
        self._data: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        """Load existing results from the JSONL file into memory (if it exists)."""
        if not self.jsonl_path.exists():
            return

        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    # Skip malformed lines instead of crashing.
                    continue

                sample_id = obj.get("sample_id")
                dataset_name = obj.get("dataset_name")  # may be None for old files
                experiment_name = obj.get("experiment_name")
                model_name = obj.get("model_name")
                payload = obj.get("payload")

                if not payload or payload.get("final_answer") is None:
                    continue

                if sample_id is None or experiment_name is None or model_name is None:
                    # Not a valid record for our cache; skip it.
                    continue

                # For backward compatibility: old cache files may not have dataset_name,
                # treat them as dataset_name = "" (they won't be hit if you pass a real name).
                sample_id = str(sample_id)
                dataset_name = "" if dataset_name is None else str(dataset_name)
                experiment_name = str(experiment_name)
                model_name = str(model_name)

                key = (sample_id, dataset_name, experiment_name, model_name)
                self._data[key] = payload

    def _write_all(self) -> None:
        """
        Rewrite the entire JSONL file from the in-memory dict.
        """
        if self.jsonl_path.parent:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)

        with self.jsonl_path.open("w", encoding="utf-8") as f:
            for (sample_id, dataset_name, experiment_name, model_name), payload in self._data.items():
                
                if payload.get("final_answer") is None:
                    continue
                obj = {
                    "sample_id": sample_id,
                    "dataset_name": dataset_name,
                    "experiment_name": experiment_name,
                    "model_name": model_name,
                    "payload": payload,
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def has(
        self,
        sample_id: str,
        dataset_name: str,
        experiment_name: str,
        model_name: str,
    ) -> bool:
        key = (str(sample_id), str(dataset_name), str(experiment_name), str(model_name))
        return key in self._data

    def get(
        self,
        sample_id: str,
        dataset_name: str,
        experiment_name: str,
        model_name: str,
    ) -> Optional[Dict[str, Any]]:
        key = (str(sample_id), str(dataset_name), str(experiment_name), str(model_name))
        return self._data.get(key)

    def set(
        self,
        sample_id: str,
        dataset_name: str,
        experiment_name: str,
        model_name: str,
        payload: Dict[str, Any],
    ) -> None:
        """
        Store or update a payload and rewrite the JSONL file.
        Called only from the main thread in our threaded runner.
        """
        if payload.get("final_answer") is None:
            return
        
        key = (str(sample_id), str(dataset_name), str(experiment_name), str(model_name))
        self._data[key] = payload
        self._write_all()

    def close(self) -> None:
        """No-op for now; kept for API compatibility."""
        pass
