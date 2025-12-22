from __future__ import annotations

import json
from dataclasses import dataclass
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from aftabe_vlm.dataset import load_dataset, PuzzleSample
from aftabe_vlm.evaluation import parse_model_response, is_correct
from aftabe_vlm.models import VisionLanguageModel
from aftabe_vlm.models.metis_gemini_2_0_flash import MetisGemini20Flash
from aftabe_vlm.models.metis_gpt4o import MetisGPT4o
from aftabe_vlm.models.GoogleAPI import GoogleVertexGemini
# from aftabe_vlm.models.gemma3 import Gemma3
from validation.prompts_config import get_base_prompts, get_prompt_variants
from tqdm import tqdm
import re

from aftabe_vlm.caching import ResultCache


CACHE_VERSION = "simple_validation_v1"


# =====================================================
# 1) INTERNAL HELPERS
# =====================================================

@dataclass
class PromptCombo:
    """
    A single "experiment":

    - variant_name: which prompt variant (user_v1, user_v2, ...)
    - mode:        "system" or "user" (where the combined prompt is placed)
    - text:        the full combined prompt (base + variant)
    """
    variant_name: str
    mode: str   # "system" or "user"
    text: str

    @property
    def combo_name(self) -> str:
        # This is what we use to key stats / cache experiments
        return f"{self.variant_name}__as_{self.mode}"


def _build_prompt_combos(category: str) -> List[PromptCombo]:
    """
    Build all prompt combinations for a given dataset category (en, pe, cross).

    For each category:
      - get the base prompt via get_base_prompts(category)
      - get 3 variants via get_prompt_variants()
      - produce 3 combined prompts (base + variant)
      - for each combined prompt, create 2 combos:
          - as SYSTEM prompt  (mode="system")
          - as USER prompt    (mode="user")
    => 3 * 2 = 6 combos per category
    """
    combos: List[PromptCombo] = []

    base = get_base_prompts(category)
    variants = get_prompt_variants()

    for var in variants:
        variant_name = var["name"]
        variant_text = var["template"]
        combined = (base + "\n\n" + variant_text).strip()

        # Scenario 1: combined prompt as system prompt
        # combos.append(
        #     PromptCombo(
        #         variant_name=variant_name,
        #         mode="system",
        #         text=combined,
        #     )
        # )

        # Scenario 2: combined prompt as user prompt
        combos.append(
            PromptCombo(
                variant_name=variant_name,
                mode="user",
                text=combined,
            )
        )

    return combos


def _build_user_prompt_for_mode(
    combo: PromptCombo,
    sample: PuzzleSample,
    hint_text: str = "",
) -> Tuple[str, str]:
    """
    Given a combo and a sample, build (system_prompt, user_prompt)
    according to the mode.

    - If mode == "system":
        system_prompt = combo.text   (base + variant)
        user_prompt   = a minimal, sample-aware instruction
    - If mode == "user":
        system_prompt = a short generic system instruction
        user_prompt   = combo.text   (base + variant)

    You can tweak these if you want different behavior, but the key
    requirement is:
      - "system" mode => prompt before the image (system role)
      - "user"   mode => prompt after the image (user role)
    """
    if combo.mode == "system":
        system_prompt = combo.text
        user_prompt = (
            "Here is the puzzle image. Please provide the answer according to the system instructions."
        )
    else:  # combo.mode == "user"
        system_prompt = (
            "You are an expert assistant that solves picture word puzzles."
        )
        user_prompt = combo.text
        if hint_text:
            user_prompt += f"\n\n{hint_text}"

    return system_prompt, user_prompt


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
        f"{_sanitize_for_filename(combo.combo_name)}.jsonl"
    )
    path = output_dir / fname

    with path.open("w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return path


def _experiment_name_for_combo(combo: PromptCombo) -> str:
    """
    Build an experiment_name for the cache.

    ResultCache key is (sample_id, dataset_name, experiment_name, model_name).

    We include CACHE_VERSION and the combo_name so that:
    - different prompt variants/modes don't collide
    - bumping CACHE_VERSION invalidates old results.
    """
    return f"{CACHE_VERSION}__{combo.combo_name}"


# =====================================================
# 2) PER-SAMPLE EVALUATION
# =====================================================
answer_extractor = GoogleVertexGemini()

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
    answer_text = sample.answer
    non_space_count = len([c for c in answer_text if not c.isspace()])
    hint_text = f"HINT: The target answer has {non_space_count} non-space characters."

    # 2. Pass hint to the builder
    system_prompt, user_prompt = _build_user_prompt_for_mode(combo, sample, hint_text=hint_text)

    extra_metadata = {
        "experiment": "simple_validation",
        "prompt_combo": combo.combo_name,
        "prompt_variant": combo.variant_name,
        "prompt_mode": combo.mode,  # "system" or "user"
        "sample_id": sample.id,
        "dataset": dataset_name,
        "hint_used": hint_text, # Good to track that a hint was used
    }

    model_response = model.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image_path=sample.image_path,
        extra_metadata=extra_metadata,
    )

    # OLDER PARSE CODE (FOR JSON FORMAT)
    # parsed = parse_model_response(model_response.raw_text)
    # correct = is_correct(
    #     predicted=parsed.final_answer,
    #     gold=sample.answer,
    #     language=sample.answer_language,
    # )

    # reasoning = getattr(parsed, "reasoning", None)
    
    # debug_info: Dict[str, Any] = {
    #     "dataset": dataset_name,
    #     "sample_id": str(sample.id),
    #     "model_name": model.name,
    #     "prompt_combo": combo.combo_name,
    #     "prompt_variant": combo.variant_name,
    #     "prompt_mode": combo.mode,
    #     "final_answer": parsed.final_answer,
    #     "gold": sample.answer,
    #     "answer_language": sample.answer_language,
    #     "correct": bool(correct),
    #     "reasoning": reasoning,
    #     "hint_text": hint_text, # Add this to debug info
    #     "full_system_prompt": system_prompt,   # The exact system instruction used
    #     "full_user_prompt": user_prompt,       # The exact user message used
    #     "full_model_output": model_response.raw_text, # The raw unparsed string from the LLM
    # }

    # return correct, debug_info

    # 1. Parse the raw model output to separate reasoning from the final answer
    raw_output = model_response.raw_text
    # Pattern looks for everything up to "answer":[...]
    # Group 1 = Reasoning, Group 2 = The content inside the brackets
    def extract_final_answer(raw_model_output: str, vertex_client) -> Optional[str]:
        """
        Extracts the 'final answer' from raw text using the Vertex AI client.
        
        Args:
            raw_model_output: The long string containing reasoning and answer.
            vertex_client: An instance of GoogleVertexGemini.
        """

        # 2. Construct the Prompt
        # We explicitly tell the model to ignore the image.
        system_instructions = (
            "You are a text processing engine. "
            "IGNORE the provided image; it is a placeholder. "
            "Focus ONLY on the text provided by the user."
        )

        user_query = f"""
        I have a raw text output from a reasoning model. 
        It usually ends with a JSON-like format: "answer":"<CONTENT>"

        TASK:
        Extract strictly the <CONTENT> string.
        
        RULES:
        - Return ONLY the content string.
        - Do not include the key "answer".
        - Do not include quotes unless they are part of the content.
        - Do not include reasoning.

        --- RAW TEXT START ---
        {raw_model_output}
        --- RAW TEXT END ---
        """

        # 3. Call the API
        response = vertex_client.generate(
            system_prompt=system_instructions,
            user_prompt=user_query,
            image_path=None  # Passing the dummy image path
        )

        # 4. Cleanup and Format
        result = response.raw_text.strip()
        
        # Remove surrounding quotes if the model added them (e.g. "Cat" -> Cat)
        if len(result) > 1 and result.startswith('"') and result.endswith('"'):
            result = result[1:-1]

        return result

    extracted_answer = extract_final_answer(raw_output, answer_extractor)
    extracted_reasoning = raw_output

    correct = is_correct(
        predicted=extracted_answer,
        gold=sample.answer,
        language=sample.answer_language,
    )


    debug_info: Dict[str, Any] = {
        "dataset": dataset_name,
        "sample_id": str(sample.id),
        "model_name": model.name,
        "prompt_combo": combo.combo_name,
        "prompt_variant": combo.variant_name,
        "prompt_mode": combo.mode,
        "final_answer": extracted_answer,
        "gold": sample.answer,
        "answer_language": sample.answer_language,
        "correct": bool(correct),
        "reasoning": extracted_reasoning,
        "hint_text": hint_text, # Add this to debug info
        "full_system_prompt": system_prompt,   # The exact system instruction used
        "full_user_prompt": user_prompt,       # The exact user message used
        "full_model_output": model_response.raw_text, # The raw unparsed string from the LLM
    }

    return correct, debug_info


# =====================================================
# 3) MAIN VALIDATION LOGIC (MULTITHREADED + JSON OUTPUT + CACHE)
# =====================================================

def run_validation(
    dataset_path: str | Path,
    category: str,
    models: List[VisionLanguageModel],
    max_samples: Optional[int] = None,
    workers: int = 8,
    output_dir: str | Path = "validation_outputs",
    cache: Optional[ResultCache] = None,
) -> Dict[str, Any]:
    """
    Run validation on a dataset JSONL using:
      - one attempt per sample (SimpleExperiment-like behaviour, no hints)
      - multiple models
      - all combinations of (prompt_variant x mode) for the given category
      - parallelized model calls with a thread pool
      - per-(dataset, model, prompt combo) JSONL outputs
      - optional ResultCache to skip repeated API calls
    """
    dataset_path = Path(dataset_path)
    samples: List[PuzzleSample] = load_dataset(dataset_path)
    # if max_samples is not None:
    #     samples = samples[:max_samples]
    if not samples:
        raise ValueError(f"No samples loaded from {dataset_path}")

    dataset_name = dataset_path.name
    combos = _build_prompt_combos(category)
    if not combos:
        raise ValueError(
            "No prompt combinations defined. "
            "Check get_base_prompts() and get_prompt_variants()."
        )

    workers = max(1, int(workers))
    output_dir = Path(output_dir)

    stats: Dict[tuple, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})

    for combo in combos:
        experiment_name = _experiment_name_for_combo(combo)

        for model in models:
            print(
                f"\n=== Validation: model='{model.name}', "
                f"combo='{combo.combo_name}' "
                f"on dataset='{dataset_name}' (category='{category}') ==="
            )

            # Collect per-sample results to dump to JSONL after threads finish
            results_for_combo_model: List[Dict[str, Any]] = []

            # Split samples into cached vs to_run
            to_run: List[PuzzleSample] = []

            for sample in samples:
                sample_id_str = str(sample.id)

                if cache is not None and cache.has(
                    sample_id_str, dataset_name, experiment_name, model.name
                ):
                    payload = cache.get(
                        sample_id_str, dataset_name, experiment_name, model.name
                    )
                    if payload is None:
                        to_run.append(sample)
                        continue

                    correct_cached = bool(payload.get("correct", False))
                    key_stats = (model.name, combo.combo_name)
                    stats[key_stats]["total"] += 1
                    if correct_cached:
                        stats[key_stats]["correct"] += 1

                    results_for_combo_model.append(payload)
                else:
                    to_run.append(sample)

            if not to_run:
                print("  All samples for this (model, combo) are cached; skipping API calls.")
                written_path = _write_results_jsonl(
                    results=results_for_combo_model,
                    dataset_name=dataset_name,
                    model_name=model.name,
                    combo=combo,
                    output_dir=output_dir,
                )
                print(f" Saved results to: {written_path}")
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

                    key_stats = (model.name, combo.combo_name)
                    stats[key_stats]["total"] += 1
                    if correct:
                        stats[key_stats]["correct"] += 1

                    results_for_combo_model.append(debug)

                    if cache is not None:
                        cache.set(
                            sample_id=sample_id_str,
                            dataset_name=dataset_name,
                            experiment_name=experiment_name,
                            model_name=model.name,
                            payload=debug,
                        )

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
    Entry point. Runs all 3 categories (en, pe, cross) with
    the models and writes:
      - per-experiment JSONL files
      - a summary JSON file per category with accuracies
    """
    models: List[VisionLanguageModel] = [
        # MetisGemini20Flash(),
        GoogleVertexGemini(),
        # MetisGPT4o(model = "gpt-5.1",
        #            effort = "none"),
    ]

    # dataset_key == category: 'en', 'pe', 'cross'
    datasets = [
        ("en", Path("dataset/en/en-dataset.jsonl")),
        ("pe", Path("dataset/pe/pe-dataset.jsonl")),
        # ("cross", Path("dataset/cross/cross-dataset.jsonl")),
    ]

    script_dir = Path(__file__).resolve().parent
    cache_path = script_dir / "validation_cache.jsonl"
    cache = ResultCache(cache_path)

    summary_dir = script_dir / "validation_outputs"
    summary_dir.mkdir(parents=True, exist_ok=True)

    for dataset_key, val_jsonl_path in datasets:
        print(f"\n##### VALIDATION on category={dataset_key} #####")
        results = run_validation(
            dataset_path=val_jsonl_path,
            category=dataset_key,
            models=models,
            max_samples=None,    # or None for full val set
            workers=8,
            output_dir=summary_dir,
            cache=cache,
        )

        # Console summary
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

        # 🔥 Write a summary file with performance & results for this category
        summary_path = summary_dir / f"{dataset_key}__summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nSaved summary to: {summary_path}")

    cache.close()


if __name__ == "__main__":
    main()
