from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List
import json
import csv


@dataclass
class PuzzleSample:
    """
    Representation of one puzzle.

    Expected fields in the source file:
        - id: unique identifier (string or int)
        - image_path: path to image (relative to dataset file or absolute)
        - answer: ground-truth word/phrase
        - answer_language: e.g. "en", "fa", "cross", "fa-en"
        - category: e.g. "english", "persian", "cross_lingual"
    """
    id: str
    image_path: str
    answer: str
    answer_language: str
    category: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def _load_jsonl(path: Path) -> List[PuzzleSample]:
    samples: List[PuzzleSample] = []
    base_dir = path.parent

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sid = str(obj.get("id"))
            img = obj.get("image_path") or obj.get("image") or obj.get("image_file")
            if img is None:
                raise ValueError(f"Missing image_path for sample {sid}")

            img_path = str((base_dir / img).resolve()) if not Path(img).is_absolute() else img

            answer = obj.get("answer")
            if answer is None:
                raise ValueError(f"Missing answer for sample {sid}")

            answer_language = obj.get("answer_language", "unknown")
            category = obj.get("category", obj.get("split", "unknown"))

            extra = {
                k: v
                for k, v in obj.items()
                if k
                not in {"id", "image_path", "image", "image_file", "answer", "answer_language", "category", "split"}
            }

            samples.append(
                PuzzleSample(
                    id=sid,
                    image_path=img_path,
                    answer=str(answer),
                    answer_language=str(answer_language),
                    category=str(category),
                    metadata=extra,
                )
            )
    return samples


def _load_csv(path: Path) -> List[PuzzleSample]:
    samples: List[PuzzleSample] = []
    base_dir = path.parent

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = str(row.get("id"))
            img = row.get("image_path") or row.get("image") or row.get("image_file")
            if img is None:
                raise ValueError(f"Missing image_path for sample {sid}")

            img_path = str((base_dir / img).resolve()) if not Path(img).is_absolute() else img

            answer = row.get("answer")
            if answer is None:
                raise ValueError(f"Missing answer for sample {sid}")

            answer_language = row.get("answer_language", "unknown")
            category = row.get("category", row.get("split", "unknown"))

            extra = {
                k: v
                for k, v in row.items()
                if k not in {"id", "image_path", "image", "image_file", "answer", "answer_language", "category", "split"}
            }

            samples.append(
                PuzzleSample(
                    id=sid,
                    image_path=img_path,
                    answer=str(answer),
                    answer_language=str(answer_language),
                    category=str(category),
                    metadata=extra,
                )
            )
    return samples


def load_dataset(path: str | Path) -> List[PuzzleSample]:
    """
    Load dataset from JSONL or CSV.

    For JSONL: each line is a JSON object with puzzle fields.
    For CSV: header row defines column names.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    suffix = p.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        return _load_jsonl(p)
    elif suffix == ".csv":
        return _load_csv(p)
    else:
        raise ValueError(f"Unsupported dataset format: {suffix}")
