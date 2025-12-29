import os
import json
import random

# --- Configuration ---
SOURCE_ROOT = "dataset"       # Where your current data lives
TARGET_ROOT = "clean"         # Where the new "fixed" data will go
LANGUAGES = ["en", "pe", "cross", "ar"]

def generate_static_hint(answer: str) -> str:
    """
    Generates the 'shuffle_chars' pattern (10% revealed).
    We do this ONCE here so it is static for all models.
    """
    if not answer:
        return ""
    
    clean_answer = answer.strip()
    
    # Find indices of characters that are NOT spaces
    char_indices = [i for i, c in enumerate(clean_answer) if c != ' ']
    num_chars = len(char_indices)
    
    if num_chars == 0:
        return clean_answer

    # Calculate how many to reveal: 
    reveal_count = round(0.25 * num_chars)
    
    # Pick random indices to reveal
    # Note: Since this script runs once, this random choice becomes PERMANENT in the file
    reveal_indices = set(random.sample(char_indices, min(reveal_count, num_chars)))
    
    hint_chars = []
    for i, char in enumerate(clean_answer):
        if char == ' ':
            hint_chars.append(' ')
        elif i in reveal_indices:
            hint_chars.append(char)
        else:
            hint_chars.append('_')
    return "".join(hint_chars)

def main():
    # 1. Create target directory
    if not os.path.exists(TARGET_ROOT):
        os.makedirs(TARGET_ROOT)
        print(f"Created folder: {TARGET_ROOT}")

    # 2. Process each language
    for lang in LANGUAGES:
        source_path = os.path.join(SOURCE_ROOT, lang, f"{lang}-dataset.jsonl")
        target_path = os.path.join(TARGET_ROOT, f"{lang}-dataset.jsonl")
        
        if not os.path.exists(source_path):
            print(f"Skipping {lang} (Source not found: {source_path})")
            continue
            
        print(f"Processing {lang}...")
        
        processed_count = 0
        with open(source_path, 'r', encoding='utf-8') as f_in, \
             open(target_path, 'w', encoding='utf-8') as f_out:
            
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    row = json.loads(line)
                    
                    # --- GENERATE THE STATIC HINT ---
                    ground_truth = row.get("answer", "")
                    hint_pattern = generate_static_hint(ground_truth)
                    
                    # Add to the row
                    row["hint_pattern"] = hint_pattern
                    
                    # Save to new file
                    f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    processed_count += 1
                    
                except json.JSONDecodeError:
                    print(f"Warning: Skipped invalid JSON line in {lang}")

        print(f" -> Saved {processed_count} rows to {target_path}")


if __name__ == "__main__":
    main()