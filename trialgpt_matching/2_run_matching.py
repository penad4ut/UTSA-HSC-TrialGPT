
"""
Run TrialGPT Matching (criterion-by-criterion version).
- Loads retrieved_trials.json (patients + trials)
- Loads criteria_extracted.json (ground truth criteria)
- Matches each patient-note-trial pair against criteria
- Supports crash-resume (skip already processed triplets)
"""

import json
import os
from nltk.tokenize import sent_tokenize
from TrialGPT import trialgpt_matching

# ----------------------------
# Config
# ----------------------------
retrieved_trials_path = "../results/retrieved_trials.json"
criteria_path = "../results/criteria_extracted.json"
output_path = "../results/matching_results.json"


# ----------------------------
# Helper functions
# ----------------------------
def load_retrieved_trials(path):
    with open(path, "r") as f:
        return json.load(f)


def save_matching_results(output, path):
    with open(path, "w") as f:
        json.dump(output, f, indent=4)

# help detect parse errors when re-running
def has_parse_error(record):
    """
    Check if inclusion/exclusion results contain 'Error parsing model output'
    """
    for inc_exc in ["inclusion", "exclusion"]:
        for item in record.get("results", {}).get(inc_exc, []):
            if isinstance(item.get("reasoning", ""), str) and "Error parsing model output" in item["reasoning"]:
                return True
    return False



# ----------------------------
# Main
# ----------------------------
def main():
    print("[INFO] Starting TrialGPT Matching...")

    dataset = load_retrieved_trials(retrieved_trials_path)

    # Load criteria DB
    with open(criteria_path, "r") as f:
        criteria_db = json.load(f)

    # Resume if previous results exist
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            output = json.load(f)
    else:
        output = {}

    for instance in dataset:
        patient_id = instance["patient_id"]
        note_id = instance["note_id"]
        patient_note = instance["patient_raw"]

        # Number sentences
        sents = sent_tokenize(patient_note)
        sents.append(
            "The patient will provide informed consent, and will comply with the trial protocol without any practical issues."
        )
        numbered_sents = [f"{idx}. {sent}" for idx, sent in enumerate(sents)]
        final_patient_note = "\n".join(numbered_sents)

        # Init patient-note block
        if patient_id not in output:
            output[patient_id] = {}
        if note_id not in output[patient_id]:
            output[patient_id][note_id] = {"0": {}, "1": {}, "2": {}}

        for label in ["0", "1", "2"]:  # relevance buckets
            if label not in instance:
                continue
            for trial in instance[label]:
                trial_id = trial["NCTID"]

                # Skip already processed
                # Handle already processed
                if trial_id in output[patient_id][note_id][label]:
                    record = output[patient_id][note_id][label][trial_id]
                    if not has_parse_error(record):
                        print(f"[SKIP] {patient_id}-{note_id}-{trial_id} already complete")
                        continue
                    else:
                        print(f"[RE-RUN] Found parse error in {patient_id}-{note_id}-{trial_id}, removing old record")
                        del output[patient_id][note_id][label][trial_id]

                # Lookup criteria
                criteria_dict = criteria_db.get(
                    trial_id, {"inclusion": [], "exclusion": []}
                )

                try:
                    print(f"[MATCH] Patient {patient_id}-{note_id}, Trial {trial_id}, Bucket {label}")
                    match_result = trialgpt_matching(
                        trial_id, final_patient_note, criteria_dict
                    )
                    output[patient_id][note_id][label][trial_id] = match_result
                    save_matching_results(output, output_path)
                    print(f"[✅ DONE] {patient_id}-{note_id}-{trial_id}")
                except Exception as e:
                    print(f"[ERROR] Matching {patient_id}-{note_id}-{trial_id}: {e}")
                    continue

    print(f"[INFO] Trial matching completed. Results saved to {output_path}")


# ----------------------------
if __name__ == "__main__":
    main()
