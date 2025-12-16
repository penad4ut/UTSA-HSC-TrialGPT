import json
import os
import sys
from nltk.tokenize import sent_tokenize
from TrialGPT import trialgpt_aggregation

# Config
matching_results_path = "../results/matching_results.json"
trial_info_path = "../results/trial_info.json"
queries_path = "../data/queries.jsonl"
output_path = "../results/aggregation_results.json"

print("[INFO] Starting TrialGPT aggregation...")

###########################################
# Helper: Check completeness
###########################################
def is_incomplete(record: dict) -> bool:
    if not record:  # completely empty {}
        return True
    if "relevance_score_R" not in record or "eligibility_score_E" not in record:
        return True
    return False



# Load matching results
with open(matching_results_path, encoding="utf-8") as f:
    matching_results = json.load(f)
print(f"[INFO] Loaded {len(matching_results)} patients from matching results")

# Load trial metadata
with open(trial_info_path, encoding="utf-8") as f:
    trial2info = json.load(f)
print(f"[INFO] Loaded {len(trial2info)} trials from trial_info.json")

# Load queries (note-level)
queries = {}
with open(queries_path, encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        patient_id = obj["_id"]
        note_id = obj["note_id"]
        text = obj["text"]
        queries.setdefault(patient_id, {})[note_id] = text
print(f"[INFO] Loaded {len(queries)} patients from queries.jsonl")

# Load existing aggregation output if exists
if os.path.exists(output_path):
    with open(output_path, encoding="utf-8") as f:
        output = json.load(f)
    print(f"[INFO] Loaded previous aggregation results.")
else:
    output = {}

# Loop
for patient_id, note_dict in matching_results.items():
    for note_id, label_block in note_dict.items():
        if patient_id not in output:
            output[patient_id] = {}
        if note_id not in output[patient_id]:
            output[patient_id][note_id] = {}

        # Fetch and format note
        if patient_id not in queries or note_id not in queries[patient_id]:
            print(f"[WARN] Missing note for {patient_id} - {note_id}")
            continue

        sents = sent_tokenize(queries[patient_id][note_id])
        sents.append("The patient will provide informed consent, and will comply with the trial protocol without any practical issues.")
        formatted_note = "\n".join([f"{i}. {s}" for i, s in enumerate(sents)])

        # Track whether this note had any candidate trials at all
        had_trials = any(len(trials) > 0 for trials in label_block.values())
        added = False

        for label, trials in label_block.items():
            for trial_id, match in trials.items():
                if trial_id in output[patient_id][note_id]:
                    record = output[patient_id][note_id][trial_id]
                    if not is_incomplete(record):
                        print(f"[SKIP] Already processed {patient_id}-{note_id}-{trial_id}")
                        continue
                    else:
                        print(f"[RE-RUN] Incomplete aggregation for {patient_id}-{note_id}-{trial_id}")
                try:
                    trial_info = trial2info[trial_id]
                    print("@@@@@@Before Entering TrialGPT@@@@@@@@@")
                    print(f"{trial_id} *")
                    print(f"{patient_id} **")
                    print(f"{trial_info} ***")
                    print(f"{match} ****")
                    print(f"{formatted_note} *****")
                    result = trialgpt_aggregation(trial_id, patient_id, trial_info, match, formatted_note)
                    output[patient_id][note_id][trial_id] = result
                    added = True
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(output, f, indent=2)
                except Exception as e:
                    print(f"[ERROR] Aggregation failed for {patient_id}-{note_id}-{trial_id}: {e}")
                    continue

        # Only add NO_TRIAL if there were no trials at all for this note
        if not had_trials:
            output[patient_id][note_id]["NO_TRIAL"] = {
                "relevance_score_R": 0,
                "relevance_explanation": "No candidate trials retrieved.",
                "eligibility_score_E": 0,
                "eligibility_explanation": "No candidate trials retrieved."
            }
            with open(output_path, "w") as f:
                json.dump(output, f, indent=2)

print(f"Aggregation completed. Output saved to {output_path}")
