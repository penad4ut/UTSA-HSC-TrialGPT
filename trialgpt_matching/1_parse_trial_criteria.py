__author__ = "msyed"

"""
1_parse_trial_criteria.py

This code handles the cases where instead of saving each criteria in cache it is physically stored for matching.
Saw cases where LLM hallucinated often missing some criteria or merging few criteria together.

Extracts and structures inclusion/exclusion criteria from retrieved_trials.json.
Outputs a JSON file where each trial has numbered inclusion/exclusion lists.
"""

import json
import os
import re

# ----------------------------
# Config
# ----------------------------
retrieved_trials_path = "../results/retrieved_trials.json"
output_path = "../results/criteria_extracted.json"


# ----------------------------
# Helper: Parse criteria block
# ----------------------------
def parse_criteria_block(criteria_text, ctype="inclusion"):
    """
    Splits a block of criteria text into numbered items.
    Returns a list of dicts: [{"criterion_id": 0, "text": "..."}]
    """
    if not criteria_text:
        return []

    # Normalize whitespace
    text = criteria_text.strip()

    # Split on newlines or bullets
    raw_items = re.split(r"\n\s*[-*\d\)\.]*\s*", text)

    criteria = []
    idx = 0
    for item in raw_items:
        item = item.strip()
        if not item:
            continue

        # Skip headers like "Inclusion Criteria"
        if "inclusion criteria" in item.lower() or "exclusion criteria" in item.lower():
            continue

        # Filter out too-short fragments
        if len(item) < 5:
            continue

        criteria.append({"criterion_id": idx, "text": item})
        idx += 1

    return criteria


# ----------------------------
# Main Extraction
# ----------------------------
def extract_criteria():
    with open(retrieved_trials_path, "r") as f:
        retrieved_trials = json.load(f)

    trial_criteria = {}

    for instance in retrieved_trials:
        for label in ["0", "1", "2"]:  # relevance buckets
            if label not in instance:
                continue

            for trial in instance[label]:
                trial_id = trial["NCTID"]

                if trial_id not in trial_criteria:
                    trial_criteria[trial_id] = {"inclusion": [], "exclusion": []}

                inc_text = trial.get("inclusion_criteria", "")
                exc_text = trial.get("exclusion_criteria", "")

                inclusion = parse_criteria_block(inc_text, "inclusion")
                exclusion = parse_criteria_block(exc_text, "exclusion")

                # If already seen trial_id, skip duplicates
                if not trial_criteria[trial_id]["inclusion"]:
                    trial_criteria[trial_id]["inclusion"] = inclusion
                if not trial_criteria[trial_id]["exclusion"]:
                    trial_criteria[trial_id]["exclusion"] = exclusion

    # Save structured output
    with open(output_path, "w") as f:
        json.dump(trial_criteria, f, indent=4)

    print(f"[INFO] Criteria extracted for {len(trial_criteria)} trials")
    print(f"[INFO] Saved to {output_path}")


# ----------------------------
if __name__ == "__main__":
    extract_criteria()
