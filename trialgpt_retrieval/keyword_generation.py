
import json
import os
from llama_cpp import Llama

# Provide the saved local model path
DEFAULT_MODEL_PATH = "model/"

def clean_json_output(text: str) -> str:
    # Remove ```json ... ``` fences if present
    if text.startswith("```"):
        text = text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    return text


def generate_keywords(note, model_path=DEFAULT_MODEL_PATH):
    llm = Llama(
        model_path=model_path,
        n_ctx=32768,
        n_gpu_layers=999,
        gpu_split="25,25,25,25",
        f16_kv=True,
        verbose=False
    )

    prompt = (
        "You are a medical research assistant specializing in clinical trial matching. "
        "Your task is to analyze patient descriptions to identify upto 32 key medical conditions"
        "Prioritize accuracy and relevance in your analysis. "
        "Summarize the patient's primary medical problems clearly in 1-2 sentences and STRICTLY Extract ONLY what is explicitly mentioned in the same way in notes of the following categories: "
        "Output must be a single valid JSON object, with this exact structure:\n"
        "{"
        "\"summary\": \"summary text\", "
        "\"conditions\": [\"condition1\", \"condition2\"], "
        "}\n\n"
        f"Patient description:\n{note}\n\n"
        "JSON output:"
    )

    response = llm(prompt, max_tokens=4096, temperature=0.0)
    text = response["choices"][0]["text"].strip()
    text = clean_json_output (text)


    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            output_start = text.find('{')
            output_json = text[output_start: text.rfind('}')+1]
            return json.loads(output_json)
        except Exception as e:
            print(f"[JSON ERROR] Failed to parse model output:\n{text}\nReason: {e}")
            return {"summary": "", "conditions": [], "error": "JSON parse failed"}


if __name__ == "__main__":
    # queries.jsonl is your input data file with _id as patient_id, note_id, and text: as note text
    input_file = "../data/queries.jsonl"
    output_file = "../results/retrieval_keywords.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    #  Track already processed (if rerunning)
    seen = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f_out:
            for line in f_out:
                try:
                    item = json.loads(line)
                    # only mark as seen if conditions lis is non-empty
                    if item.get("conditions"):
                        seen.add((item["patient_id"], item["note_id"]))
                except Exception:
                    continue

    total = 0
    skipped = 0
    failed = 0

    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(output_file, "a", encoding="utf-8") as f_out:

        for line in f_in:
            entry = json.loads(line)
            pid = entry["_id"]
            nid = entry["note_id"]

            if (pid, nid) in seen:
                print(f"[⏭️ SKIPPED] {pid}-{nid}")
                skipped += 1
                continue

            try:
                result = generate_keywords(entry["text"])
                output = {
                    "patient_id": pid,
                    "note_id": nid,
                    # "note": entry["text"],
                    "summary": result.get("summary", ""),
                    "conditions": result.get("conditions", [])
                    # "medications": result.get("medications", []),
                    # "labs": result.get("labs", [])
                }
                f_out.write(json.dumps(output) + "\n")
                f_out.flush()
                total += 1
                print(f"[✅ DONE] {pid}-{nid}")
            except Exception as e:
                print(f"[❌ FAIL] {pid}-{nid}: {e}")
                failed += 1

    print(f"\n Finished: {total} processed, {skipped} skipped, {failed} failed")
