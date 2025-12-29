import json
import os

def split_jsonl_by_context(input_file, true_output_file, others_output_file):
    count_true = 0
    count_others = 0

    try:
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(true_output_file, 'w', encoding='utf-8') as f_true, \
             open(others_output_file, 'w', encoding='utf-8') as f_others:

            for line in f_in:
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    
                    # Check if 'model_name' is strictly "gemini-2.5-flash"
                    if data.get("model_name") == "gpt-5.2":
                        f_true.write(line + '\n')
                        count_true += 1
                    else:
                        f_others.write(line + '\n')
                        count_others += 1
                        
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line[:50]}...")

        print(f"Done!")
        print(f"Moved {count_true} lines to: {true_output_file}")
        print(f"Kept {count_others} lines in: {others_output_file}")

    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")

# --- usage ---
# Replace 'data.jsonl' with your actual file name
input_filename = 'results_cache.jsonl' 
output_filename = 'gpt-5.2_old_without_assistant.jsonl'
remaining_filename = 'results_cache11.jsonl'

split_jsonl_by_context(input_filename, output_filename, remaining_filename)