import json
import os
# *************************Note*******************************
# this file takes output from keyword_generation.py to add span index; but not working as it supposed
# because sometimes model interprets or not exact disease/keyword so no exact text match of condition
# future work to visualize extracted keyword using this.
# *************************Note*******************************
def find_spans(note, keywords):
    spans = []
    lowered_note = note.lower()

    for kw in keywords:
        if isinstance(kw, dict) and "text" in kw:
            keyword_text = kw["text"]
        elif isinstance(kw, str):
            keyword_text = kw
        else:
            continue

        kw_lower = keyword_text.lower()
        start_idx = lowered_note.find(kw_lower)

        if start_idx != -1:
            spans.append({
                "text": keyword_text,
                "start": start_idx,
                "end": start_idx + len(keyword_text)
            })
        else:
            spans.append({
                "text": keyword_text,
                "start": None,
                "end": None
            })

    return spans

def enrich_json_with_spans(input_json_path, original_notes_path, output_json_path):
    # Load generated output with conditions/meds/labs
    with open(input_json_path, "r", encoding="utf-8") as f:
        outputs_list = []
        for line in f:
            if line.strip():
                outputs_list.append(json.loads(line))

        # Convert list into dict: {patient_id: {note_id: info}}
        outputs = {}
        for item in outputs_list:
            pid = item["patient_id"]
            nid = item["note_id"]
            if pid not in outputs:
                outputs[pid] = {}
            outputs[pid][nid] = item

    # Load notes from queries.jsonl
    notes = {}
    with open(original_notes_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            pid = entry["_id"]
            note_id = entry["note_id"]
            if pid not in notes:
                notes[pid] = {}
            notes[pid][note_id] = entry["text"]

    enriched_outputs = {}

    for pid, notes_dict in outputs.items():
        enriched_outputs[pid] = {}

        for note_id, info in notes_dict.items():
            note_text = notes.get(pid, {}).get(note_id, "")

            enriched_outputs[pid][note_id] = {
                # "note": info.get("note",""),
                "summary": info.get("summary", ""),
                "conditions": find_spans(note_text, info.get("conditions", []))
            }

    # Save enriched output
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(enriched_outputs, f, indent=4)

    print(f"Enriched output saved to: {output_json_path}")

if __name__ == "__main__":
    enrich_json_with_spans(
        input_json_path="results/retrieval_keywords.json",
        original_notes_path="data/queries.jsonl",
        output_json_path="results/retrieval_keywords_enriched.json"
    )
