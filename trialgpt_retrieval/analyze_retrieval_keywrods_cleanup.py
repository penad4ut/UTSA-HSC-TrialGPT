import json
import shutil
'''
Run only if you need to cleanup be careful.
'''
input_file = "../results/retrieval_keywords.json"
backup_file = "../results/retrieval_keywords_Orig.json"
output_file = "../results/retrieval_keywords.json"

# Step 1. Backup original file
shutil.copy(input_file, backup_file)
print(f"[INFO] Backup created: {backup_file}")

# Step 2. Load and filter
cleaned = []
total = 0
removed = 0

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except Exception as e:
            print(f"[WARN] Skipping invalid JSON line: {e}")
            continue

        total += 1
        if not record.get("conditions"):  # empty or missing
            removed += 1
            continue
        cleaned.append(record)

# Step 3. Write cleaned file
with open(output_file, "w", encoding="utf-8") as f:
    for rec in cleaned:
        f.write(json.dumps(rec) + "\n")

print("=== CLEANUP SUMMARY ===")
print(f"Total records in original: {total}")
print(f"Removed (empty conditions): {removed}")
print(f"Remaining records: {len(cleaned)}")
print(f"Cleaned file saved to: {output_file}")
