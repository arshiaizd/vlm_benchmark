import os
import json
import random
import logging
import threading
import time
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import string
import re

# --- Import from your updated files ---
from clean.prompt import get_prompt
from clean.examples import load_derivations
from aftabe_vlm.models.GoogleAPI import GoogleVertexGemini, GoogleVertexConfig
from aftabe_vlm.models.AvalaiGemini import AvalAiGemini
from aftabe_vlm.models.DeepSeekAPI import DeepSeekAPI
from aftabe_vlm.models.GemmaAPI import GemmaAPI
from aftabe_vlm.models.OpenaiAPI import OpenaiGPT
from aftabe_vlm.models.QwenAPI import Qwen3
# --------------------------------------

# --- Configuration ---
DATASET_ROOT = "dataset"
CACHE_FILE = "results_cache.jsonl"
MAX_WORKERS = 8

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("experiment.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
cache_lock = threading.Lock()


def normalize_answer(text: str) -> str:
    """
    Normalizes text to handle robustness cases:
    - Case insensitive
    - Ignores spaces ("match point" == "matchpoint")
    - Ignores hyphens ("match-point" == "matchpoint")
    - Ignores trailing punctuation
    """
    if not text:
        return ""
    
    # 1. Lowercase

    text = re.sub(r'[\u064B-\u0652]', '', text)
    
    text = text.lower()
    # 2. Remove punctuation (keeps letters/digits safe, removes .,-!?)
    # This map removes all standard punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 3. Remove spaces
    text = text.replace(" ", "")
    
    return text

# =============================================================================
# Data Loading & Caching
# =============================================================================
def load_dataset(root_path: str) -> Dict[str, List[Dict]]:
    datasets = {}
    if not os.path.exists(root_path):
        logger.error(f"Dataset root not found: {root_path}")
        return {}

    lang_dirs = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
    lang_dirs = sorted(lang_dirs)
    
    for lang in lang_dirs:
        if lang not in ["en", "pe", "cross", "ar"]:
            continue
        jsonl_path = os.path.join(root_path, lang, f"{lang}-dataset.jsonl")
        if not os.path.exists(jsonl_path):
            continue
            
        entries = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        row = json.loads(line)
                        row['full_image_path'] = os.path.join(root_path, lang, row['image_path'])
                        row['language'] = lang
                        entries.append(row)
                    except json.JSONDecodeError:
                        continue
        
        # Sort by ID to ensure deterministic split
        try:
            entries.sort(key=lambda x: int(x['id']))
        except:
            pass
            
        datasets[lang] = entries
        logger.info(f"Loaded {len(entries)} samples for language '{lang}'")
        
    return datasets

def prepare_data_split(lang: str, dataset: List[Dict], num_examples: int):
    """
    Separates the first N items to be used EXCLUSIVELY as examples.
    Returns (examples_list, test_set_list)
    """
    if len(dataset) <= num_examples:
        logger.warning(f"Not enough data to split {num_examples} examples. Using 0.")
        return [], dataset
    
    derivations = load_derivations()

    examples = [[dataset[i], derivations[lang][i]] for i in derivations[lang].keys()]
    test_set = [item for i, item in enumerate(dataset) if i not in derivations[lang].keys()]
    return examples, test_set

def load_cache() -> Dict[str, Any]:
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    key = (
                        f"{data.get('model_name', 'unknown')}_"
                        f"{data['language']}_"
                        f"{data['id']}_"
                        f"{str(data.get('use_context'))}_"
                        f"{str(data.get('hint_type'))}_"
                        f"{str(data.get('pass_at_enabled'))}_"
                        f"{str(data.get('num_pass'))}"
                    )
                    if data.get('final_response') is not None:
                        cache[key] = data
                except json.JSONDecodeError:
                    continue
    return cache

def save_to_cache(result: Dict):
    with cache_lock:
        with open(CACHE_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

# --- Global Cache for Hints ---
# Structure: {'en': {'123': 'b_s_c', ...}, 'pe': {...}}
_CLEAN_PATTERNS_CACHE = {}
CLEAN_DATASET_ROOT = "clean"  # Ensure this folder exists and contains your generated files

def _ensure_clean_patterns_loaded():
    """Lazy-loads the clean patterns into memory only once."""
    global _CLEAN_PATTERNS_CACHE
    if _CLEAN_PATTERNS_CACHE:
        return

    # Check if folder exists
    if not os.path.exists(CLEAN_DATASET_ROOT):
        print(f"Warning: '{CLEAN_DATASET_ROOT}' folder not found. Hints will be random.")
        return

    # Load each language file
    for filename in os.listdir(CLEAN_DATASET_ROOT):
        if filename.endswith("-dataset.jsonl"):
            # Extract lang from filename (e.g., "en-dataset.jsonl" -> "en")
            lang = filename.split("-")[0] 
            _CLEAN_PATTERNS_CACHE[lang] = {}
            
            filepath = os.path.join(CLEAN_DATASET_ROOT, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            row = json.loads(line)
                            # Store by ID (ensure string format for consistency)
                            _CLEAN_PATTERNS_CACHE[lang][str(row['id'])] = row.get('hint_pattern', '')
            except Exception as e:
                print(f"Error loading clean file {filename}: {e}")

# =============================================================================
# Helpers
# =============================================================================
def generate_hint(sample_id: str, language: str, answer: str, hint_type: str) -> str:
    if not answer: return ""
    clean = answer.strip()
    
    if hint_type == "char_count":
        return f"\nHINT: The answer has {len(clean.replace(' ', ''))} characters (excluding spaces)."
    elif hint_type == "shuffle_chars":

        masked = _CLEAN_PATTERNS_CACHE[language].get(str(sample_id))

        return (
            f"\nHINT: The pattern of the answer is '{masked}'. "
            f"In this pattern, '_' represents a hidden character and spaces represent actual spaces in the answer."
        )
    return ""

def clean_json_response(text: str) -> Dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2: text = "\n".join(lines[1:-1])
    try:
        return json.loads(text)
    except:
        return {"error": "Invalid JSON", "raw": text}

# =============================================================================
# Processing Logic
# =============================================================================
def process_sample(
    sample: Dict, 
    examples: List[List[Any]], 
    model: Any, 
    use_context: bool, 
    hint_type: Optional[str],
    pass_at_enabled: bool,
    num_pass: int,
    model_name: str
):
    if 'gemini' in model_name.lower():
        role = 'model'
    else:
        role = 'assistant'
    try:
        lang = sample['language']
        id = sample['id']
        
        # --- Build Message History ---
        messages = []
        
        # 1. System Prompt
        sys_prompt = get_prompt(lang)
        
        # 2. In-Context Examples (Images + Answers)
        if use_context and examples:
            first_ex_data, derivation = examples[0]
            messages.append({
                "role": "user",
                "text": f"{sys_prompt}\n\nHere is an example puzzle:\n",
                "image_path": first_ex_data['full_image_path']
            })
            messages.append({
                "role": role,
                "text": json.dumps({
                    "primary_clues": derivation["primary_clues"],
                    "candidates": derivation["candidates"],
                    "final_answer": first_ex_data['answer']
                    }, ensure_ascii=False)
            })
            
            for ex_data, ex_derivation in examples[1:]:
                messages.append({
                    "role": "user",
                    "text": "Here is another example:",
                    "image_path": ex_data['full_image_path']
                })
                messages.append({
                    "role": role,
                    "text": json.dumps({
                        "primary_clues": ex_derivation["primary_clues"], # Using the derivation list here
                        "candidates": ex_derivation["candidates"],
                        "final_answer": ex_data['answer']
                    }, ensure_ascii=False)
                })
            
            # Final Target setup
            hint_str = generate_hint(sample['id'], sample['language'], sample['answer'], hint_type) if hint_type else ""
            messages.append({
                "role": "user",
                "text": f"Now solve this new puzzle. Provide the JSON output.{hint_str}",
                "image_path": sample['full_image_path']
            })
        else:
            # No context: System prompt + Target Image
            hint_str = generate_hint(sample['id'], sample['language'], sample['answer'], hint_type) if hint_type else ""
            messages.append({
                "role": "user",
                "text": f"{sys_prompt}\n\nAnalyze the image and provide the JSON solution.{hint_str}",
                "image_path": sample['full_image_path']
            })

        # --- Iterative Execution (Pass@) ---
        max_loops = num_pass if pass_at_enabled else 1
        attempts = []
        final_json = None
        is_correct = False
        
        # Copy for history tracking
        current_history = list(messages)

        norm_ground_truth = sample['answer'].strip()

        for i in range(max_loops):
            logger.info(f"Processing {lang}-{sample['id']} | Attempt {i+1}")
            
            # Call Model using the new generate_chat method
            response = model.generate_chat(current_history)
            parsed = clean_json_response(response.raw_text)
            
            model_ans = parsed.get("final_answer", "").strip()
            norm_model_ans = normalize_answer(model_ans)
            norm_ground_truth = normalize_answer(norm_ground_truth)
            is_right = norm_model_ans == norm_ground_truth
            
            attempts.append({
                "attempt_idx": i,
                "response": response.raw_text,
                "parsed": model_ans,
                "correct": is_right
            })
            
            if is_right:
                is_correct = True
                final_json = parsed
                break
            
            # If wrong, append feedback
            if i < max_loops - 1:
                current_history.append({
                    "role": role, 
                    "text": response.raw_text
                })
                current_history.append({
                    "role": "user", 
                    "text": f"The answer '{model_ans}' is incorrect. Please Step-by-Step:\n"
                            f"1. List the primary clues you see in the image again.\n"
                            f"2. Consider alternative interpretations or puns for these elements.\n"
                            f"3. Provide a NEW answer in the correct JSON format."
                })

        # --- Save Result ---
        result = {
            "id": sample['id'],
            "language": lang,
            "model_name": model_name,
            "ground_truth": sample['answer'],
            "model_ans": model_ans,
            "hint_type": hint_type,
            "use_context": use_context,
            "pass_at_enabled": pass_at_enabled,
            "num_pass": num_pass,
            "solved": is_correct,
            "attempts": attempts,
            "final_response": final_json or attempts[-1]["parsed"],
            "message_history": current_history
        }
        save_to_cache(result)

    except Exception as e:
        logger.error(f"Failed {sample['language']}-{sample['id']}: {e}")

# =============================================================================
# Main Execution
# =============================================================================
def main():
    # Init model (now containing generate_chat)
    models = {
        # "gemma": GemmaAPI(),
        # "deepseek": DeepSeekAPI(),
        "gemini-flash": AvalAiGemini(model="gemini-2.5-flash"),
        "gemini-pro": AvalAiGemini(model="gemini-2.5-pro"),
        # "gemini-flash": GoogleVertexGemini(GoogleVertexConfig(model_name="gemini-2.5-flash")),
        # "gemini-pro": GoogleVertexGemini(GoogleVertexConfig(model_name="gemini-2.5-pro")),
        "gpt": OpenaiGPT(),
        # "qwen": Qwen3(),
    }
    for model in models.keys():
        model = models[model]
        MODEL_NAME = model.model_name
        
        # Configuration
        USE_CONTEXT = False
        NUM_EXAMPLES = 3     
        HINT_TYPE = "shuffle_chars" 
        PASS_AT_ENABLED = False
        NUM_PASS = 3
        
        logger.info(f"Config: Model={MODEL_NAME}, Context={USE_CONTEXT}({NUM_EXAMPLES} imgs), Hint={HINT_TYPE}, Pass@={PASS_AT_ENABLED}")

        datasets = load_dataset(DATASET_ROOT)
        cache = load_cache()
        _ensure_clean_patterns_loaded()
        tasks = []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for lang, all_rows in datasets.items():
                
                # Split Data: Reserve first N as examples
                examples, test_set = prepare_data_split(lang, all_rows, NUM_EXAMPLES)
                
                logger.info(f"Language {lang}: Using {len(examples)} examples, {len(test_set)} test samples.")

                for row in test_set:
                    # Check Cache
                    key = (
                        f"{MODEL_NAME}_" 
                        f"{lang}_"
                        f"{row['id']}_"
                        f"{str(USE_CONTEXT)}_"
                        f"{str(HINT_TYPE)}_"
                        f"{str(PASS_AT_ENABLED)}_"
                        f"{str(NUM_PASS)}"
                    )
                    
                    if key in cache:
                        continue
                    
                    future = executor.submit(
                        process_sample,
                        sample=row,
                        examples=examples, 
                        model=model,
                        use_context=USE_CONTEXT,
                        hint_type=HINT_TYPE,
                        pass_at_enabled=PASS_AT_ENABLED,
                        num_pass=NUM_PASS,
                        model_name=MODEL_NAME
                    )
                    tasks.append(future)

                    # time.sleep(4) # To avoid rate limits

            for future in as_completed(tasks):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Thread error: {e}")

        logger.info("Done.")

if __name__ == "__main__":
    main()