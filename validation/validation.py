from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from aftabe_vlm.dataset import load_dataset, PuzzleSample
from aftabe_vlm.evaluation import parse_model_response, is_correct
from aftabe_vlm.models import VisionLanguageModel
from aftabe_vlm.models.metis_gemini_2_0_flash import MetisGemini20Flash
from aftabe_vlm.models.metis_gpt4o import MetisGPT4o
from prompts_config import SYSTEM_PROMPTS, return_user_prompts
from tqdm import tqdm


from aftabe_vlm.caching import ResultCache


CACHE_VERSION = "simple_validation_v1"


# =====================================================
# 1) INTERNAL HELPERS
# =====================================================

@dataclass
class PromptCombo:
    system_name: str
    user_name: str
    system_text: str
    user_template: str

    @property
    def combo_name(self) -> str:
        return f"{self.system_name}__{self.user_name}"


def _build_prompt_combos(answer_language: str) -> List[PromptCombo]:
    combos: List[PromptCombo] = []
    for sys in SYSTEM_PROMPTS:
        for usr in return_user_prompts(answer_language):
            combos.append(
                PromptCombo(
                    system_name=sys["name"],
                    user_name=usr["name"],
                    system_text=sys["text"],
                    user_template=usr["template"],
                )
            )
    return combos


def _build_user_prompt_from_template(template: str, sample: PuzzleSample) -> str:
    return template.format(
        id=sample.id,
        answer_language=sample.answer_language,
    )


def _sanitize_for_filename(part: str) -> str:
    # Keep alnum, dash, underscore; replace others with '_'
    return "".join(
        c if c.isalnum() or c in ("-", "_") else "_"
        for c in str(part)
    )


def _write_results_jsonl(
    results: List[Dict[str, Any]],
    dataset_name: str,
    model_name: str,
    combo: PromptCombo,
    output_dir: Path,
) -> Path:
    """
    Write per-sample results for a specific (dataset, model, combo) to JSONL.
    Returns the path of the written file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fname = (
        f"{_sanitize_for_filename(dataset_name)}__"
        f"{_sanitize_for_filename(model_name)}__"
        f"{_sanitize_for_filename(combo.system_name)}__"
        f"{_sanitize_for_filename(combo.user_name)}.jsonl"
    )
    path = output_dir / fname

    with path.open("w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return path


def _experiment_name_for_combo(combo: PromptCombo) -> str:
    """
    Build an experiment_name for the cache.

    ResultCache key is (sample_id, experiment_name, model_name).

    We include CACHE_VERSION and the prompt combo so that different
    system/user prompts don't collide and bumping CACHE_VERSION
    invalidates old results.
    """
    return f"{CACHE_VERSION}__{combo.combo_name}"


# =====================================================
# 2) PER-SAMPLE EVALUATION
# =====================================================

def _evaluate_one_sample(
    model: VisionLanguageModel,
    combo: PromptCombo,
    sample: PuzzleSample,
    dataset_name: str,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Run a single model call for (model, combo, sample).

    Returns:
      (correct: bool, debug_info: dict)
    """
    system_prompt = combo.system_text
    user_prompt = _build_user_prompt_from_template(combo.user_template, sample)

    extra_metadata = {
        "experiment": "simple_validation",
        "prompt_combo": combo.combo_name,
        "system_name": combo.system_name,
        "user_name": combo.user_name,
        "sample_id": sample.id,
        "dataset": dataset_name,
    }

    model_response = model.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image_path=sample.image_path,
        extra_metadata=extra_metadata,
    )

    parsed = parse_model_response(model_response.raw_text)
    correct = is_correct(
        predicted=parsed.final_answer,
        gold=sample.answer,
        language=sample.answer_language,
    )

    # Try to get reasoning trace if parse_model_response exposes it.
    reasoning = getattr(parsed, "reasoning", None)

    debug_info: Dict[str, Any] = {
        "dataset": dataset_name,
        "sample_id": str(sample.id),
        "model_name": model.name,
        "system_name": combo.system_name,
        "user_name": combo.user_name,
        "prompt_combo": combo.combo_name,
        # do NOT save raw prompts or image path
        "final_answer": parsed.final_answer,
        "gold": sample.answer,
        "answer_language": sample.answer_language,
        "correct": bool(correct),
        "reasoning": reasoning,  # model reasoning trace
    }

    return correct, debug_info


# =====================================================
# 3) MAIN VALIDATION LOGIC (MULTITHREADED + JSON OUTPUT + CACHE)
# =====================================================

def run_validation(
    dataset_path: str | Path,
    answer_language: str,
    models: List[VisionLanguageModel],
    max_samples: Optional[int] = None,
    workers: int = 4,
    output_dir: str | Path = "validation_outputs",
    cache: Optional[ResultCache] = None,
) -> Dict[str, Any]:
    """
    Run validation on a dataset JSONL using:
      - one attempt per sample (SimpleExperiment-like behaviour, no hints)
      - multiple models
      - all combinations of SYSTEM_PROMPTS x user prompt templates
      - parallelized model calls with a thread pool
      - per-(dataset, model, prompt combo) JSONL outputs
      - optional ResultCache to skip repeated API calls
    """
    dataset_path = Path(dataset_path)
    samples: List[PuzzleSample] = load_dataset(dataset_path)
    if max_samples is not None:
        samples = samples[:max_samples]

    if not samples:
        raise ValueError(f"No samples loaded from {dataset_path}")

    dataset_name = dataset_path.name
    combos = _build_prompt_combos(answer_language)
    if not combos:
        raise ValueError(
            "No prompt combinations defined. "
            "Check SYSTEM_PROMPTS and return_user_prompts()."
        )

    workers = max(1, int(workers))
    output_dir = Path(output_dir)

    stats: Dict[tuple, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})

    for combo in combos:
        experiment_name = _experiment_name_for_combo(combo)

        for model in models:
            print(
                f"\n=== Validation: model='{model.name}', "
                f"system='{combo.system_name}', user='{combo.user_name}' "
                f"on dataset='{dataset_name}' ==="
            )

            # Collect per-sample results to dump to JSONL after threads finish
            results_for_combo_model: List[Dict[str, Any]] = []

            # Split samples into cached vs to_run
            to_run: List[PuzzleSample] = []

            for sample in samples:
                sample_id_str = str(sample.id)

                if cache is not None and cache.has(sample_id_str, experiment_name, model.name):
                    payload = cache.get(sample_id_str, experiment_name, model.name)
                    if payload is None:
                        to_run.append(sample)
                        continue

                    # payload is exactly what we stored (debug_info)
                    correct_cached = bool(payload.get("correct", False))
                    key_stats = (model.name, combo.combo_name)
                    stats[key_stats]["total"] += 1
                    if correct_cached:
                        stats[key_stats]["correct"] += 1

                    results_for_combo_model.append(payload)
                else:
                    to_run.append(sample)

            if not to_run:
                print("  All samples for this (model, prompt) are cached; skipping API calls.")
                written_path = _write_results_jsonl(
                    results=results_for_combo_model,
                    dataset_name=dataset_name,
                    model_name=model.name,
                    combo=combo,
                    output_dir=output_dir,
                )
                print(f"  Saved results to: {written_path}")
                continue

            # Thread pool per (combo, model), parallel over uncached samples
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        _evaluate_one_sample,
                        model,
                        combo,
                        sample,
                        dataset_name,
                    ): sample
                    for sample in to_run
                }

                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"{model.name} / {combo.combo_name}",
                ):
                    sample = futures[future]
                    sample_id_str = str(sample.id)

                    try:
                        correct, debug = future.result()
                    except Exception as e:
                        # e.g. HTTP 400/500 etc.: no cache entry is written,
                        # and this sample will be retried on the next run.
                        print(
                            f"Error on sample {sample.id} for "
                            f"{model.name} / {combo.combo_name}: {e}"
                        )
                        continue

                    # Update stats
                    key_stats = (model.name, combo.combo_name)
                    stats[key_stats]["total"] += 1
                    if correct:
                        stats[key_stats]["correct"] += 1

                    # Collect for JSONL
                    results_for_combo_model.append(debug)

                    # Store in cache (payload is debug dict)
                    if cache is not None:
                        cache.set(
                            sample_id=sample_id_str,
                            experiment_name=experiment_name,
                            model_name=model.name,
                            payload=debug,
                        )

            # After all samples for this (model, combo) are done -> write JSONL
            written_path = _write_results_jsonl(
                results=results_for_combo_model,
                dataset_name=dataset_name,
                model_name=model.name,
                combo=combo,
                output_dir=output_dir,
            )
            print(f"  Saved results to: {written_path}")

    # ---------- Aggregate accuracies ----------

    per_model_prompt_accuracy: Dict[str, Dict[str, float]] = defaultdict(dict)
    for (model_name, combo_name), counts in stats.items():
        total = counts["total"]
        correct = counts["correct"]
        accuracy = correct / total if total > 0 else 0.0
        per_model_prompt_accuracy[model_name][combo_name] = accuracy

    best_prompt_per_model: Dict[str, Dict[str, Any]] = {}
    for model_name, combo_accs in per_model_prompt_accuracy.items():
        best_name = max(combo_accs, key=combo_accs.get)
        best_prompt_per_model[model_name] = {
            "combo_name": best_name,
            "accuracy": combo_accs[best_name],
        }

    from collections import defaultdict as dd

    combo_to_accs: Dict[str, List[float]] = dd(list)
    for model_name, combo_accs in per_model_prompt_accuracy.items():
        for combo_name, acc in combo_accs.items():
            combo_to_accs[combo_name].append(acc)

    overall_best_combo_name: Optional[str] = None
    overall_best_score: float = -1.0
    for combo_name, acc_list in combo_to_accs.items():
        if not acc_list:
            continue
        avg_acc = sum(acc_list) / len(acc_list)
        if avg_acc > overall_best_score:
            overall_best_score = avg_acc
            overall_best_combo_name = combo_name

    overall_best_prompt = {
        "combo_name": overall_best_combo_name,
        "avg_accuracy": overall_best_score,
    }

    return {
        "per_model_prompt_accuracy": dict(per_model_prompt_accuracy),
        "best_prompt_per_model": best_prompt_per_model,
        "overall_best_prompt": overall_best_prompt,
    }


# =====================================================
# 4) MAIN ENTRYPOINT
# =====================================================

def main() -> None:
    """
    Example entrypoint. Adjust models, dataset paths, and workers as needed.
    """
    models: List[VisionLanguageModel] = [
        MetisGemini20Flash(),
        MetisGPT4o(),
    ]

    datasets = [
        ("en", Path("../dataset/en_val/en-dataset-val.jsonl")),
        ("pe", Path("../dataset/pe_val/pe-dataset-val.jsonl")),
        ("cross", Path("../dataset/cross_val/cross-dataset-val.jsonl")),
    ]

    answer_languages = {
        "en": "English",
        "pe": "Persian",
        "cross": "Persian",
    }

    # One cache shared across all datasets/models/combos for this run.
    script_dir = Path(__file__).resolve().parent
    cache_path = script_dir / "validation_cache.jsonl"
    cache = ResultCache(cache_path)

    for dataset_name, val_jsonl_path in datasets:
        print(f"\n##### VALIDATION on {dataset_name} #####")
        results = run_validation(
            dataset_path=val_jsonl_path,
            answer_language=answer_languages[dataset_name],
            models=models,
            max_samples=50,    # or None for full val set
            workers=8,         # tweak as you like
            output_dir="validation_outputs",  # where JSONL logs go
            cache=cache,       # ✅ enable caching
        )

        print("\nPer-model / per-prompt-combo accuracy:")
        for model_name, combo_accs in results["per_model_prompt_accuracy"].items():
            for combo_name, acc in combo_accs.items():
                print(f"  {model_name} | {combo_name}: {acc:.3f}")

        print("\nBest prompt combo per model:")
        for model_name, info in results["best_prompt_per_model"].items():
            print(
                f"  {model_name}: best='{info['combo_name']}' "
                f"(acc={info['accuracy']:.3f})"
            )

        overall = results["overall_best_prompt"]
        print(
            f"\nOverall best combo (avg across models): "
            f"{overall['combo_name']} (avg_acc={overall['avg_accuracy']:.3f})"
        )

    cache.close()


if __name__ == "__main__":
    main()
