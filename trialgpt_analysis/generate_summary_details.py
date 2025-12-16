import json
import pandas as pd

def expand_matching_results(matching_path: str, output_csv_path: str = None) -> pd.DataFrame:
    with open(matching_path, "r") as f:
        matching_data = json.load(f)

    rows = []
    for patient_id, notes in matching_data.items():
        for note_id, blocks in notes.items():
            for label_block, trials in blocks.items():  # e.g., "0", "1", "2"
                for trial_id, match in trials.items():
                    result_block = match.get("results", {})

                    for criteria_type in ["inclusion", "exclusion"]:
                        criteria_list = result_block.get(criteria_type, [])

                        if isinstance(criteria_list, dict):
                            # Handle dict format (e.g., GPT4All)
                            for cid, values in criteria_list.items():
                                if isinstance(values, list) and len(values) == 3:
                                    reasoning, sentence_ids, label = values
                                    rows.append({
                                        "patient_id": patient_id,
                                        "note_id": note_id,
                                        "trial_id": trial_id,
                                        "criterion_id": int(cid),
                                        "type": criteria_type,
                                        "reasoning": reasoning,
                                        "label": label,
                                        "sentence_ids": sentence_ids
                                    })
                        elif isinstance(criteria_list, list):
                            # Handle LLaMA structured list format
                            for item in criteria_list:
                                rows.append({
                                    "patient_id": patient_id,
                                    "note_id": note_id,
                                    "trial_id": trial_id,
                                    "criterion_id": item.get("criterion_id"),
                                    "type": criteria_type,
                                    "reasoning": item.get("reasoning"),
                                    "label": item.get("label"),
                                    "sentence_ids": item.get("list_of_sentence_ids", [])
                                })

    df = pd.DataFrame(rows)

    if output_csv_path:
        df.to_csv(output_csv_path, index=False)
        print(f"✅ Detailed matching results saved to {output_csv_path}")

    return df


# Example usage
if __name__ == "__main__":
    df_detailed = expand_matching_results(
        matching_path="../results/matching_results.json",
        output_csv_path="../results/matching_details_output.csv"
    )
    print(df_detailed.head())
