'''
Run this code to cleanup the retrieval keywords file.
'''

import json

input_file = f"results/retrieval_keywords.jsonl"
output_file = f"results/retrieval_keywords.json"

data = []

# Read JSONL (line by line)
with open(input_file, "r", encoding="utf-8") as f_in:
    for line in f_in:
        line = line.strip()
        if not line:
            continue
        try:
            data.append(json.loads(line))
        except Exception as e:
            print(f"[❌ Error parsing line] {e}\n{line}")

# Write as JSON array
with open(output_file, "w", encoding="utf-8") as f_out:
    json.dump(data, f_out, indent=2, ensure_ascii=False)

print(f"✅ Converted {len(data)} records to {output_file}")
