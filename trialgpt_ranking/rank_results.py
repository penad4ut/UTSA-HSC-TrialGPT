
"""
Rank the trials given the matching and aggregation results
"""

import json
import sys


def get_matching_score(results):
    eps = 1e-8  # to avoid divide-by-zero
    included = 0
    not_inc = 0
    no_info_inc = 0
    na_inc = 0

    excluded = 0
    not_exc = 0
    no_info_exc = 0
    na_exc = 0

    matching = results.get("results", results)  # unwrap if nested

    for item in matching.get("inclusion", []):
        label = item.get("label", "")
        if label == "included":
            included += 1
        elif label == "not included":
            not_inc += 1
        elif label == "not enough information":
            no_info_inc += 1
        elif label == "not applicable":
            na_inc += 1

    for item in matching.get("exclusion", []):
        label = item.get("label", "")
        if label == "excluded":
            excluded += 1
        elif label == "not excluded":
            not_exc += 1
        elif label == "not enough information":
            no_info_exc += 1
        elif label == "not applicable":
            na_exc += 1

    score = included / (included + not_inc + no_info_inc + eps)

    if not_inc > 0:
        score -= 1
    if excluded > 0:
        score -= 1

    return score


def get_agg_score(assessment):
    try:
        rel_score = float(assessment["relevance_score_R"])
        eli_score = float(assessment["eligibility_score_E"])
    except:
        rel_score = 0
        eli_score = 0

    score = (rel_score + eli_score) / 100

    return score

#load trials for later readibility or improved results
trial2info = {}
with open("../results/trial_info.json", "r", encoding="utf-8") as f:
    trial2info = json.load(f)


if __name__ == "__main__":
    # args are the results paths
    matching_results_path = f"../results/matching_results.json"
    agg_results_path = f"../results/aggregation_results.json"

    # loading the results
    matching_results = json.load(open(matching_results_path, encoding='utf-8'))
    agg_results = json.load(open(agg_results_path, encoding='utf-8'))

    # write to file at the end
    output_path = "../results/ranked_results.json"

    # Compute rankings
    all_ranked_results = {}

    for patient_id, note_dict in matching_results.items():
        all_ranked_results[patient_id] = {}

        for note_id, label_dict in note_dict.items():
            trial2score = {}

            for label, trial_dict in label_dict.items():
                for trial_id, results in trial_dict.items():
                    matching_score = get_matching_score(results)

                    # Get aggregation score
                    if (
                            patient_id not in agg_results or
                            note_id not in agg_results[patient_id] or
                            trial_id not in agg_results[patient_id][note_id]
                    ):
                        print(f"[WARN] Patient {patient_id}, Note {note_id}, Trial {trial_id} not in aggregation.")
                        agg_score = 0
                    else:
                        agg_score = get_agg_score(agg_results[patient_id][note_id][trial_id])

                    trial_score = matching_score + agg_score
                    trial2score[trial_id] = trial_score

            # Sort and format
            sorted_trials = sorted(trial2score.items(), key=lambda x: -x[1])
            all_ranked_results[patient_id][note_id] = [
                {
                    "trial_id": trial,
                    "total_score": round(score, 4),
                    "brief_title": trial2info.get(trial, {}).get("brief_title", "")
                }
                for trial, score in sorted_trials
            ]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_ranked_results, f, indent=2)

        print(f"\n✅ Ranked results saved to: {output_path}")





