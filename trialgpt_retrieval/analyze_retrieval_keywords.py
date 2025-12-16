'''

Code for analyzing if all files processed retrieval keywords generation step

'''

import json
from collections import defaultdict

# Input file
input_file = "../results/retrieval_keywords.json"

seen = defaultdict(list)   # (patient_id, note_id) -> line numbers
empty_conditions = []      # (patient_id, note_id, line_no)
total_records = 0

with open(input_file, "r", encoding="utf-8") as f:
    for line_no, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except Exception as e:
            print(f"[WARN] Line {line_no}: JSON parse error -> {e}")
            continue

        total_records += 1
        pid = record.get("patient_id")
        nid = record.get("note_id")
        conds = record.get("conditions", [])

        key = (pid, nid)
        seen[key].append(line_no)

        if not conds:  # empty list or missing
            empty_conditions.append((pid, nid, line_no))

# Report duplicates
duplicates = {k: v for k, v in seen.items() if len(v) > 1}

print("=== SUMMARY ===")
print(f"Total records: {total_records}")
print(f"Unique (patient,note) pairs: {len(seen)}")
print(f"Duplicates found: {len(duplicates)}")
print(f"Empty condition entries: {len(empty_conditions)}")
print("================")

if duplicates:
    print("\n=== DUPLICATE ENTRIES ===")
    for key, lines in duplicates.items():
        print(f"{key} appears {len(lines)} times (lines {lines})")

if empty_conditions:
    print("\n=== EMPTY CONDITIONS ===")
    for pid, nid, line_no in empty_conditions:
        print(f"Patient {pid}, Note {nid}, line {line_no}")
