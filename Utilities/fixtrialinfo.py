"""
Ran into issue where the corpus.jsonl file format was not acceptable during the ranking and
i had to temporarily create a trial_info.json file JSON file. This can be moved to Utilities or Retreval section later
when you collect the Trial Corpus.
"""

import json

input_path = f"results/corpus.jsonl"
output_path = f"results/trial_info.json"

trial_info = {}

with open(input_path, "r", encoding="utf-8") as infile:
    for line in infile:
        entry = json.loads(line)
        trial_id = entry["_id"]
        meta = entry["metadata"]

        trial_info[trial_id] = {
            "brief_title": meta.get("brief_title", ""),
            "brief_summary": meta.get("brief_summary", ""),
            "inclusion_criteria": meta.get("inclusion_criteria", ""),
            "exclusion_criteria": meta.get("exclusion_criteria", ""),
            "diseases_list": meta.get("diseases_list", [])
        }

with open(output_path, "w", encoding="utf-8") as out:
    json.dump(trial_info, out, indent=2)

print(f"✅ trial_info.json created with {len(trial_info)} trials.")
