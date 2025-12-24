import json
import os
from collections import defaultdict

# Path to your cache file
CACHE_FILE = "results_cache.jsonl"

def calculate_accuracy():
    """
    Reads the cache file and calculates accuracy per Model/Configuration per Language.
    """
    if not os.path.exists(CACHE_FILE):
        print(f"Error: Cache file '{CACHE_FILE}' not found.")
        return

    # Data Structure: 
    # stats[config_string][language] = {'correct': 0, 'total': 0}
    stats = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))

    print(f"Reading {CACHE_FILE}...")

    with open(CACHE_FILE, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON on line {line_num + 1}")
                continue

            # Extract fields
            # Default to 'unknown' if running on old cache data without model_name
            model_name = data.get('model_name', 'unknown') 
            language = data.get('language', 'unknown')
            is_solved = data.get('solved', False)
            
            # Extract Configuration Details
            use_context = data.get('use_context', False)
            hint_type = str(data.get('hint_type')) if data.get('hint_type') else "None"
            pass_at = data.get('pass_at_enabled', False)
            num_pass = data.get('num_pass', 1)

            # Create a unique readable key for this configuration
            # Format: "gpt-4o | Context: True | Hint: shuffle | Pass@: True(3)"
            pass_str = f"True({num_pass})" if pass_at else "False"
            config_key = (
                f"{model_name:<20} | "
                f"Ctx: {str(use_context):<5} | "
                f"Hint: {hint_type:<13} | "
                f"Pass@: {pass_str}"
            )

            # Update stats
            stats[config_key][language]['total'] += 1
            if is_solved:
                stats[config_key][language]['correct'] += 1

    # --- Print Table ---
    
    # Table Header
    header = f"{'CONFIGURATION':<75} | {'TASK':<6} | {'ACCURACY':<8} | {'COUNTS'}"
    print("\n" + "="*110)
    print(header)
    print("="*110)

    # Sort by Config Key for cleaner grouping
    for config in sorted(stats.keys()):
        lang_data = stats[config]
        
        # Sort languages (e.g., en, pe, cross)
        for lang in sorted(lang_data.keys()):
            counts = lang_data[lang]
            total = counts['total']
            correct = counts['correct']
            
            if total > 0:
                acc = (correct / total) * 100
            else:
                acc = 0.0
            
            # Print Row
            print(f"{config:<75} | {lang:<6} | {acc:6.2f}%  | {correct}/{total}")
        
        # Add a separator between different model configurations
        print("-" * 110)

if __name__ == "__main__":
    calculate_accuracy()