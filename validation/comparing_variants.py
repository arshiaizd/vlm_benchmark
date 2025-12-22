import json
import os
from pathlib import Path
from typing import Dict, List, Set, Any
from collections import defaultdict
from itertools import combinations

# ================= CONFIGURATION =================
INPUT_DIR = Path("validation/validation_outputs")  # Where your runner saved the .jsonl files
OUTPUT_DIR = Path("validation/comparison_results") # Where we will save the new comparison files
SUMMARY_FILE = OUTPUT_DIR / "comparison_summary.txt"
# =================================================

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

class VariantExporter:
    def __init__(self, input_dir: Path, output_dir: Path):
        self.input_dir = input_dir
        self.output_dir = output_dir
        # Structure: data[dataset][model][variant][sample_id] = { full_record }
        self.data: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(dict))
        )

    def load_data(self):
        """Load all JSONL files into memory."""
        if not self.input_dir.exists():
            print(f"Error: Input directory '{self.input_dir}' not found.")
            return

        print(f"Scanning {self.input_dir}...")
        files = list(self.input_dir.glob("*.jsonl"))

        for f in files:
            # Parse filename: {dataset}__{model}__{variant}.jsonl
            parts = f.stem.split("__")
            if len(parts) < 3:
                continue
            
            dataset = parts[0]
            model = parts[1]
            variant = "__".join(parts[2:])

            with f.open("r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    if not line: continue
                    try:
                        rec = json.loads(line)
                        sid = str(rec.get("sample_id"))
                        # Store the whole record so we can save answers later
                        self.data[dataset][model][variant][sid] = rec
                    except json.JSONDecodeError:
                        continue
        print("Data loaded.")

    def write_jsonl(self, path: Path, records: List[Dict]):
        """Helper to write a list of dicts to a JSONL file."""
        with path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def run_analysis(self):
        ensure_dir(self.output_dir)
        
        summary_lines = []

        for dataset, models in self.data.items():
            for model, variants_data in models.items():
                variant_names = sorted(list(variants_data.keys()))
                if not variant_names:
                    continue
                
                header = f"=== Dataset: {dataset} | Model: {model} ==="
                print(header)
                summary_lines.append(header)
                summary_lines.append(f"Variants found: {', '.join(variant_names)}\n")

                # Prepare sets of CORRECT sample IDs
                correct_ids = {}
                for v in variant_names:
                    correct_ids[v] = {
                        sid for sid, rec in variants_data[v].items() 
                        if rec.get("correct") is True
                    }

                # ---------------------------------------------------------
                # 1. PAIRWISE COMPARISONS
                # ---------------------------------------------------------
                for v1, v2 in combinations(variant_names, 2):
                    set1 = correct_ids[v1]
                    set2 = correct_ids[v2]

                    common = set1.intersection(set2)
                    only_v1 = set1.difference(set2)
                    only_v2 = set2.difference(set1)

                    n = len(common)
                    m = len(only_v1)
                    k = len(only_v2)

                    # --- Add to Summary Text ---
                    msg = (
                        f"Comparison [{v1}] vs [{v2}]:\n"
                        f"  - Common Correct (n): {n}\n"
                        f"  - Only {v1} Correct (m): {m}\n"
                        f"  - Only {v2} Correct (k): {k}\n"
                    )
                    summary_lines.append(msg)

                    # --- Save Detailed Files ---
                    # We create a subfolder for this pair
                    pair_folder = self.output_dir / dataset / model / f"{v1}_vs_{v2}"
                    ensure_dir(pair_folder)

                    # Save Common
                    common_recs = [variants_data[v1][sid] for sid in common] # take data from v1 (arbitrary)
                    self.write_jsonl(pair_folder / "common_correct.jsonl", common_recs)

                    # Save Disjoint V1
                    v1_recs = [variants_data[v1][sid] for sid in only_v1]
                    self.write_jsonl(pair_folder / f"only_{v1}_correct.jsonl", v1_recs)

                    # Save Disjoint V2
                    v2_recs = [variants_data[v2][sid] for sid in only_v2]
                    self.write_jsonl(pair_folder / f"only_{v2}_correct.jsonl", v2_recs)

                # ---------------------------------------------------------
                # 2. ALL TOGETHER (Intersection vs Union)
                # ---------------------------------------------------------
                if len(variant_names) > 2:
                    summary_lines.append("--- All Variants Combined ---")
                    
                    # Intersection: Correct in ALL variants
                    all_common = set.intersection(*correct_ids.values())
                    summary_lines.append(f"  - Correct in ALL variants: {len(all_common)}")
                    
                    # Union: Correct in ANY variant
                    any_correct = set.union(*correct_ids.values())
                    summary_lines.append(f"  - Correct in ANY variant: {len(any_correct)}")

                    # Uniquely Correct: Correct in ONLY one specific variant (and none of the others)
                    summary_lines.append("  - Uniquely Correct (Solved by ONLY this variant):")
                    
                    group_folder = self.output_dir / dataset / model / "all_together"
                    ensure_dir(group_folder)

                    # Save Intersection
                    if all_common:
                        # Grab record from first variant
                        first_v = variant_names[0]
                        recs = [variants_data[first_v][sid] for sid in all_common]
                        self.write_jsonl(group_folder / "common_to_all.jsonl", recs)

                    for v_target in variant_names:
                        # IDs correct in v_target
                        target_set = correct_ids[v_target]
                        # IDs correct in ANY OTHER variant
                        other_sets = [correct_ids[v] for v in variant_names if v != v_target]
                        union_others = set.union(*other_sets)
                        
                        # Difference
                        unique_to_target = target_set.difference(union_others)
                        
                        count = len(unique_to_target)
                        summary_lines.append(f"    * {v_target}: {count}")

                        # Save Unique Files
                        if unique_to_target:
                            recs = [variants_data[v_target][sid] for sid in unique_to_target]
                            self.write_jsonl(group_folder / f"unique_to_{v_target}.jsonl", recs)

                summary_lines.append("\n" + "="*40 + "\n")

        # Write the full text summary
        with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(summary_lines))
        
        print(f"\nAnalysis complete.")
        print(f"Summary saved to: {SUMMARY_FILE}")
        print(f"Detailed files saved in: {self.output_dir}")

if __name__ == "__main__":
    exporter = VariantExporter(INPUT_DIR, OUTPUT_DIR)
    exporter.load_data()
    exporter.run_analysis()