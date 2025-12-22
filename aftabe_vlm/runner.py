from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed

from pathlib import Path
from typing import List, Dict, Any, Optional
import os

try:
    from tqdm import tqdm
except ImportError:  # fallback if tqdm not installed
    def tqdm(x, **kwargs):
        return x  # type: ignore

from .dataset import load_dataset, PuzzleSample
from .prompts import BASE_SYSTEM_PROMPT, build_puzzle_user_prompt, get_prompt_variants
from .caching import ResultCache
from .evaluation import (
    parse_model_response,
    is_correct,
    SampleEvaluation,
    summarize_accuracy,
)
from .models import VisionLanguageModel, MetisGPT4o
from .models import GoogleAPI
from .experiments import (
    Experiment,
    SimpleExperiment,
    CharCountExperiment,
    PartialCharsExperiment,
    RetryWithFeedbackExperiment,
)


def create_default_models() -> List[VisionLanguageModel]:
    """Define which VLMs to test (here: Metis wrapper around GPT-4o)."""
    api_key = "tpsg-MNvTQUAqUL84o4THLV1395IqTBIZHJJ"
    # MetisGPT4o will re-check the key and raise a clearer error if missing.
    model = GoogleAPI.GoogleVertexGemini()
    # model = MetisGPT4o()
    return [model]


def create_default_experiments(
    retry_attempts: int = 3,
) -> List[Experiment]:
    return [
        # SimpleExperiment(),
        CharCountExperiment(),
        # PartialCharsExperiment(),
        # RetryWithFeedbackExperiment(max_attempts=retry_attempts),
    ]


def run_experiment_on_sample(
    experiment: Experiment,
    model: VisionLanguageModel,
    sample: PuzzleSample,
    system_prompt: str,
    variant: Dict[str, str],
) -> Dict[str, Any]:
    """Run a single experiment/model on a single sample (with retries if needed)."""
    attempts: List[Dict[str, Any]] = []

    for attempt_idx in range(experiment.max_attempts):
        hint_text = experiment.build_hint_text(sample, attempt_idx, attempts)
        user_prompt = build_puzzle_user_prompt(sample, hint_text, variant=variant)
        
        extra_metadata = {
            "experiment": experiment.name,
            "variant": variant["name"],
            "attempt_index": attempt_idx,
            "sample_id": sample.id,
        }

        model_response = model.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_path=sample.image_path,
            extra_metadata=extra_metadata,
        )

        parsed = parse_model_response(model_response.raw_text)
        correct = is_correct(parsed.final_answer, sample.answer, sample.answer_language)

        attempt_record: Dict[str, Any] = {
            "attempt_index": attempt_idx,
            "hint_text": hint_text,
            "user_prompt": user_prompt,
            "raw_output": model_response.raw_text,
            "parsed": {
                "reasoning": parsed.reasoning,
                "final_answer": parsed.final_answer,
            },
            "correct": correct,
            "provider_payload": model_response.provider_payload,
        }
        attempts.append(attempt_record)

        if correct:
            break  # stop retrying once we get a correct answer

    final_attempt = attempts[-1]
    result_record: Dict[str, Any] = {
        "sample_id": sample.id,
        "experiment_name": experiment.name,
        "variant_name": variant["name"],
        "model_name": model.name,
        "answer_language": sample.answer_language,
        "gold_answer": sample.answer,
        "correct": final_attempt["correct"],
        "attempts_used": len(attempts),
        "attempts": attempts,
    }

    return result_record


def run_all_experiments(
    dataset_path: str | Path,
    results_db_path: str | Path,
    max_samples: Optional[int] = None,
    retry_attempts: int = 3,
    workers: int = 4,
) -> None:
    """
    Run all defined experiments over the dataset with all configured models.

    This version parallelizes the slow LLM API calls using a thread pool.
    Only the main thread writes to the CSV cache to avoid corruption.
    """
    dataset_path = Path(dataset_path)
    results_db_path = Path(results_db_path)

    # Load dataset
    samples: List[PuzzleSample] = load_dataset(dataset_path)
    if max_samples is not None:
        samples = samples[:max_samples]

    # Set up cache, models, experiments, and system prompt
    cache = ResultCache(results_db_path)
    models: List[VisionLanguageModel] = create_default_models()
    experiments: List[Experiment] = create_default_experiments(
        retry_attempts=retry_attempts
    )
    # system_prompt = BASE_SYSTEM_PROMPT
    system_prompt = ""

    all_eval_records: List[SampleEvaluation] = []

    variants = get_prompt_variants()

    for experiment in experiments:
        for model in models:
            for variant in variants:
                print(
                    f"\n=== Exp: '{experiment.name}' | Model: '{model.name}' | Variant: '{variant['name']}' ==="
                )

                effective_exp_name = f"{experiment.name}_{variant['name']}"

                # Respect cache using the composite key
                samples_to_run: List[PuzzleSample] = [
                    s for s in samples
                    if not cache.has(s.id, dataset_path, effective_exp_name, model.name)
                ]

                if not samples_to_run:
                    print("All samples are already cached for this variant; skipping.")
                    continue

                max_workers = max(1, workers)

                # Threads do ONLY the LLM/API work (run_experiment_on_sample)
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_sample = {
                        executor.submit(
                            run_experiment_on_sample,
                            experiment,
                            model,
                            sample,
                            system_prompt,
                            variant,
                        ): sample
                        for sample in samples_to_run
                    }

                    # Main thread collects results and writes to cache
                    for future in tqdm(
                        as_completed(future_to_sample),
                        total=len(future_to_sample),
                        desc=f"{experiment.name} / {model.name}",
                    ):
                        sample = future_to_sample[future]

                        try:
                            result_record = future.result()
                        except Exception as e:
                            # Log error, skip this sample, continue with others
                            print(f"Error on sample {sample.id}: {e}")
                            continue

                        # 🔒 Only the main thread touches the cache / CSV
                        cache.set(
                            sample.id,
                            experiment.name,
                            model.name,
                            result_record,
                        )

                        eval_rec = SampleEvaluation(
                            sample_id=sample.id,
                            experiment_name=experiment.name,
                            model_name=model.name,
                            correct=bool(result_record["correct"]),
                            attempts_used=int(result_record["attempts_used"]),
                        )
                        all_eval_records.append(eval_rec)

    cache.close()

    if all_eval_records:
        summary = summarize_accuracy(all_eval_records)
        print("\n=== Overall summary for this run ===")
        print(f"Total evaluated samples: {summary['n']}")
        print(f"Accuracy: {summary['accuracy']:.3f}")
    else:
        print("\nNo new evaluations were run (all results were already in the cache).")