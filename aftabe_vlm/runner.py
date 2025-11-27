from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from tqdm import tqdm
except ImportError:  # fallback if tqdm not installed
    def tqdm(x, **kwargs):
        return x  # type: ignore

from .dataset import load_dataset, PuzzleSample
from .prompts import BASE_SYSTEM_PROMPT, build_puzzle_user_prompt
from .caching import ResultCache
from .evaluation import (
    parse_model_response,
    is_correct,
    SampleEvaluation,
    summarize_accuracy,
)
from .models import VisionLanguageModel, OpenAIGPT4o
from .experiments import (
    Experiment,
    SimpleExperiment,
    CharCountExperiment,
    PartialCharsExperiment,
    RetryWithFeedbackExperiment,
)


def create_default_models() -> List[VisionLanguageModel]:
    """
    Define which VLMs to test.
    """
    return [
        OpenAIGPT4o(model_name="gpt-4o"),
    ]


def create_default_experiments(
    retry_attempts: int = 3,
) -> List[Experiment]:
    return [
        SimpleExperiment(),
        CharCountExperiment(),
        PartialCharsExperiment(),
        RetryWithFeedbackExperiment(max_attempts=retry_attempts),
    ]


def run_experiment_on_sample(
    experiment: Experiment,
    model: VisionLanguageModel,
    sample: PuzzleSample,
    system_prompt: str,
) -> Dict[str, Any]:
    """
    Run a single experiment/model on a single sample,
    including handling retries (Experiment IV).
    """
    attempts: List[Dict[str, Any]] = []

    for attempt_idx in range(experiment.max_attempts):
        hint_text = experiment.build_hint_text(sample, attempt_idx, attempts)
        user_prompt = build_puzzle_user_prompt(sample, hint_text)

        extra_metadata = {
            "experiment": experiment.name,
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
        "model_name": model.name,
        "answer_language": sample.answer_language,
        "category": sample.category,
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
) -> None:
    """
    Top-level runner:
      - loads dataset,
      - iterates over experiments, models, and samples,
      - uses the SQLite cache to avoid re-running completed pairs,
      - prints aggregate accuracy stats at the end.
    """
    samples = load_dataset(dataset_path)
    if max_samples is not None:
        samples = samples[:max_samples]

    cache = ResultCache(results_db_path)
    models = create_default_models()
    experiments = create_default_experiments(retry_attempts=retry_attempts)

    system_prompt = BASE_SYSTEM_PROMPT

    all_eval_records: List[SampleEvaluation] = []

    for experiment in experiments:
        for model in models:
            print(f"\n=== Experiment: {experiment.name} | Model: {model.name} ===")
            it = tqdm(samples, desc=f"{experiment.name} / {model.name}")
            for sample in it:
                if cache.has(sample.id, experiment.name, model.name):
                    continue  # cached result, skip

                result_record = run_experiment_on_sample(
                    experiment=experiment,
                    model=model,
                    sample=sample,
                    system_prompt=system_prompt,
                )

                cache.set(
                    sample_id=sample.id,
                    experiment_name=experiment.name,
                    model_name=model.name,
                    payload=result_record,
                )

                all_eval_records.append(
                    SampleEvaluation(
                        sample_id=sample.id,
                        experiment_name=experiment.name,
                        model_name=model.name,
                        correct=result_record["correct"],
                        attempts_used=result_record["attempts_used"],
                    )
                )

    cache.close()

    summary = summarize_accuracy(all_eval_records)
    print("\n=== Overall summary across all experiments & models ===")
    print(f"Total samples evaluated: {summary['n']}")
    print(f"Overall accuracy: {summary['accuracy']:.3f}")
