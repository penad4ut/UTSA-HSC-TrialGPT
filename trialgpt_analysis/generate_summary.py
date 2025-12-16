import json
import pandas as pd

def generate_trial_summary(
    aggregation_path: str,
    ranked_path: str,
    trial_info_path: str,
    output_csv_path: str = None
):
    # Load JSON files
    with open(aggregation_path, "r") as f:
        aggregation_data = json.load(f)

    with open(ranked_path, "r") as f:
        ranked_data = json.load(f)

    with open(trial_info_path, "r") as f:
        trial_info_data = json.load(f)

    # Build combined summary table
    rows = []
    for patient_id, notes in ranked_data.items():
        for note_id, trials in notes.items():
            for trial in trials:
                trial_id = trial["trial_id"]
                total_score = trial["total_score"]
                brief_title = trial.get("brief_title", trial_info_data.get(trial_id, {}).get("brief_title", ""))

                agg = aggregation_data.get(patient_id, {}).get(note_id, {}).get(trial_id, {})
                relevance_score = agg.get("relevance_score_R", None)
                eligibility_score = agg.get("eligibility_score_E", None)

                rows.append({
                    "patient_id": patient_id,
                    "note_id": note_id,
                    "trial_id": trial_id,
                    "brief_title": brief_title,
                    "total_score": total_score,
                    "relevance_score_R": relevance_score,
                    "eligibility_score_E": eligibility_score
                })

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Export to CSV if path is given
    if output_csv_path:
        df.to_csv(output_csv_path, index=False)
        print(f"✅ Summary saved to {output_csv_path}")

    return df


# Example usage
if __name__ == "__main__":
    df_summary = generate_trial_summary(
        aggregation_path="../results/aggregation_results.json",
        ranked_path="../results/ranked_results.json",
        trial_info_path="../results/trial_info.json",
        output_csv_path="../results/trial_summary_output.csv"
    )
    print(df_summary.head())
