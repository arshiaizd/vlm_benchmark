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
from aftabe_vlm.models.OpenaiAPI import MetisGPT4o
from aftabe_vlm.models.GoogleAPI import GoogleVertexGemini
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
    Defines the single prompt configuration used for validation.
    """
    variant_name: str
    mode: str   # "system" or "user"
    text: str

    @property
    def combo_name(self) -> str:
        return f"{self.variant_name}__as_{self.mode}"


def _build_single_prompt_combo(category: str) -> PromptCombo:
    """
    Builds the SINGLE prompt combination to be used.
    """
    base = get_base_prompts(category)
    variants = get_prompt_variants()
    
    # --- CONFIGURATION: SELECT YOUR SINGLE VARIANT HERE ---
    # Currently selecting index 3 based on your previous code
    target_var = variants[3] 
    # ------------------------------------------------------

    variant_name = target_var["name"]
    variant_text = target_var["template"]
    
    combined_text = (base + "\n\n" + variant_text).strip()

    # Returns the single combo configuration (Mode: USER)
    return PromptCombo(
        variant_name=variant_name,
        mode="user",
        text=combined_text,
    )


def _build_user_prompt_for_mode(
    combo: PromptCombo,
    sample: PuzzleSample,
    hint_text: str = "",
) -> Tuple[str, str]:
    """
    Constructs the final system and user strings.
    """
    # Since we are defaulting to 'user' mode in the combo builder:
    system_prompt = "You are an expert assistant that solves picture word puzzles."
    user_prompt = combo.text
    
    if hint_text:
        user_prompt += f"\n\n{hint_text}"

    return system_prompt, user_prompt


def _sanitize_for_filename(part: str) -> str:
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
    return f"{CACHE_VERSION}__{combo.combo_name}"


# =====================================================
# 2) PER-SAMPLE EVALUATION
# =====================================================
answer_extractor = GoogleVertexGemini()

def _evaluate_one_sample(
    model: VisionLanguageModel,
    combo: PromptCombo,
    in_context_samples: List[PuzzleSample],
    sample: PuzzleSample,
    dataset_name: str,
) -> Tuple[bool, Dict[str, Any]]:
    
    answer_text = sample.answer
    non_space_count = len([c for c in answer_text if not c.isspace()])
    hint_text = f"HINT: The target answer has {non_space_count} non-space characters."
    
    system_prompt, user_prompt = _build_user_prompt_for_mode(combo, sample, hint_text=hint_text)

    extra_metadata = {
        "experiment": "simple_validation",
        "prompt_combo": combo.combo_name,
        "sample_id": sample.id,
        "dataset": dataset_name,
    }

    model_response = model.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image_path=sample.image_path,
        extra_metadata=extra_metadata,
    )

    # --- Result Extraction Logic ---
    raw_output = model_response.raw_text

    def extract_final_answer(raw_model_output: str, vertex_client) -> Optional[str]:
        system_instructions = (
            "You are a text processing engine. IGNORE the provided image. "
            "Focus ONLY on the text provided by the user."
        )
        user_query = f"""
        I have a raw text output from a reasoning model. 
        It usually ends with a JSON-like format: "answer":"<CONTENT>"
        TASK: Extract strictly the <CONTENT> string.
        --- RAW TEXT START ---
        {raw_model_output}
        --- RAW TEXT END ---
        """
        response = vertex_client.generate(
            system_prompt=system_instructions,
            user_prompt=user_query,
            image_path=None 
        )
        result = response.raw_text.strip()
        if len(result) > 1 and result.startswith('"') and result.endswith('"'):
            result = result[1:-1]
        return result

    extracted_answer = extract_final_answer(raw_output, answer_extractor)
    
    correct = is_correct(
        predicted=extracted_answer,
        gold=sample.answer,
        language=sample.answer_language,
    )

    debug_info: Dict[str, Any] = {
        "dataset": dataset_name,
        "sample_id": str(sample.id),
        "model_name": model.name,
        "final_answer": extracted_answer,
        "gold": sample.answer,
        "correct": bool(correct),
        "reasoning": raw_output,
        "full_model_output": raw_output,
    }

    return correct, debug_info


# =====================================================
# 3) MAIN VALIDATION LOGIC
# =====================================================

def run_validation(
    dataset_path: str | Path,
    category: str,
    models: List[VisionLanguageModel],
    workers: int = 8,
    output_dir: str | Path = "validation_outputs",
    cache: Optional[ResultCache] = None,
    in_context_image_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    
    dataset_path = Path(dataset_path)
    samples: List[PuzzleSample] = load_dataset(dataset_path)
    if not samples:
        raise ValueError(f"No samples loaded from {dataset_path}")

    dataset_name = dataset_path.name
    
    # --- SIMPLIFIED: Build only ONE combo ---
    combo = _build_single_prompt_combo(category)
    experiment_name = _experiment_name_for_combo(combo)
    
    in_context_samples = [sample for sample in samples if sample.id in (in_context_image_ids or [])]
    samples = [sample for sample in samples if sample.id not in (in_context_image_ids or [])]

    workers = max(1, int(workers))
    output_dir = Path(output_dir)

    # Stats: Key = model_name (since prompt is constant)
    stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})

    for model in models:
        print(
            f"\n=== Validation: model='{model.name}' | "
            f"combo='{combo.combo_name}' | dataset='{dataset_name}' ==="
        )

        results_for_model: List[Dict[str, Any]] = []
        to_run: List[PuzzleSample] = []

        # 1. Check Cache
        for sample in samples:
            sample_id_str = str(sample.id)
            if cache is not None and cache.has(sample_id_str, dataset_name, experiment_name, model.name):
                payload = cache.get(sample_id_str, dataset_name, experiment_name, model.name)
                if payload is None:
                    to_run.append(sample)
                    continue
                
                correct_cached = bool(payload.get("correct", False))
                stats[model.name]["total"] += 1
                if correct_cached:
                    stats[model.name]["correct"] += 1
                results_for_model.append(payload)
            else:
                to_run.append(sample)

        # 2. Run API calls for uncached
        if not to_run:
            print("  All samples cached.")
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        _evaluate_one_sample,
                        model,
                        combo,
                        in_context_samples,
                        sample,
                        dataset_name,
                    ): sample
                    for sample in to_run
                }

                for future in tqdm(as_completed(futures), total=len(futures), desc=model.name):
                    sample = futures[future]
                    try:
                        correct, debug = future.result()
                    except Exception as e:
                        print(f"Error on sample {sample.id}: {e}")
                        continue

                    stats[model.name]["total"] += 1
                    if correct:
                        stats[model.name]["correct"] += 1
                    
                    results_for_model.append(debug)

                    if cache is not None:
                        cache.set(str(sample.id), dataset_name, experiment_name, model.name, debug)

        # 3. Save JSONL
        _write_results_jsonl(results_for_model, dataset_name, model.name, combo, output_dir)

    # Calculate Accuracies
    accuracies = {}
    for model_name, counts in stats.items():
        total = counts["total"]
        acc = counts["correct"] / total if total > 0 else 0.0
        accuracies[model_name] = acc

    return {
        "prompt_name": combo.combo_name,
        "accuracies": accuracies
    }


# =====================================================
# 4) MAIN ENTRYPOINT
# =====================================================

def main() -> None:
    models: List[VisionLanguageModel] = [
        GoogleVertexGemini(),
        # MetisGPT4o(model="gpt-4o", effort="medium"),
    ]

    datasets = [
        ("en", Path("dataset/en/en-dataset.jsonl")),
        ("pe", Path("dataset/pe/pe-dataset.jsonl")),
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
            workers=8,
            output_dir=summary_dir,
            cache=cache,
        )

        print(f"\nResults for prompt: {results['prompt_name']}")
        for model_name, acc in results["accuracies"].items():
            print(f"  {model_name}: {acc:.3f}")

        # Save summary
        summary_path = summary_dir / f"{dataset_key}__summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    cache.close()


if __name__ == "__main__":
    main()