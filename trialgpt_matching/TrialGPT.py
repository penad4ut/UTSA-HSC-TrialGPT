
"""
TrialGPT Matching (Criterion-by-Criterion Version)

- Uses pre-parsed criteria from criteria_extracted.json
- Guarantees coverage of all inclusion/exclusion criteria
- One LLM call per criterion (safer, auditable)
"""

import json
from llama_cpp import Llama

# ----------------------------
# Load model ONCE globally
# ----------------------------
# path to local saved model
model_path = f"model/"
print(f"[INFO] Loading model from {model_path} ...")

model = Llama(
    model_path=model_path,
    n_ctx=24576,
    n_gpu_layers=999,
    f16_kv=True,
    use_mlock=True,
    verbose=False,
)


# ----------------------------
# Criterion-by-Criterion Matching
# ----------------------------
def trialgpt_matching(trial_id, patient_note, criteria_dict):
    """
    trial_id: str (NCT ID)
    patient_note: str (raw patient note, with sentence IDs added)
    criteria_dict: dict {"inclusion": [...], "exclusion": [...]}
    """

    results = {"inclusion": [], "exclusion": []}
    coverage = {}

    for inc_exc in ["inclusion", "exclusion"]:
        criteria_list = criteria_dict.get(inc_exc, [])

        if not criteria_list:
            coverage[inc_exc] = {"total": 0, "returned": 0, "missing": []}
            continue

        for c in criteria_list:
            cid = c["criterion_id"]
            ctext = c["text"]

            print(f"[INFO] Evaluating {trial_id} | {inc_exc} criterion #{cid}: {ctext[:80]}...")

            # Build prompt
            system_prompt = (
                f"You are a helpful assistant for clinical trial recruitment.\n"
                f"Compare the patient note against ONE {inc_exc} criterion.\n"
                f"Return JSON with the criterion_id, reasoning, and a label.\n\n"
            )
            if inc_exc == "inclusion":
                system_prompt += (
                    'Labels allowed: {"included", "not included", '
                    '"not applicable", "not enough information"}\n\n'
                )
            else:
                system_prompt += (
                    'Labels allowed: {"excluded", "not excluded", '
                    '"not applicable", "not enough information"}\n\n'
                )

            system_prompt += (
                "Format strictly as JSON:\n"
                '{"criterion_id": 0, "reasoning": "...", "label": "included"}\n\n'
            )

            user_prompt = (
                f"Patient note (sentence IDs added):\n{patient_note}\n\n"
                f"Trial {inc_exc} criterion #{cid}:\n{ctext}\n\n"
                f"JSON output:"
            )

            try:
                response = model.create_completion(
                    prompt=system_prompt + user_prompt,
                    max_tokens=24576,
                    temperature=0.0,
                    stop=["</s>"],
                )
                raw_text = response["choices"][0]["text"].strip()

                # Cleanup
                if raw_text.startswith("```json"):
                    raw_text = raw_text[len("```json"):].strip()
                if raw_text.endswith("```"):
                    raw_text = raw_text[:-3].strip()

                parsed = json.loads(raw_text)

                # Ensure criterion_id is correct
                parsed["criterion_id"] = cid

                results[inc_exc].append(parsed)

            except Exception as e:
                print(f"[ERROR] Failed for trial {trial_id} {inc_exc} #{cid}: {e}")
                results[inc_exc].append(
                    {
                        "criterion_id": cid,
                        "reasoning": "Error parsing model output",
                        "label": "not enough information",
                    }
                )

        # Coverage stats
        expected_ids = {c["criterion_id"] for c in criteria_list}
        returned_ids = {r["criterion_id"] for r in results[inc_exc]}
        missing = sorted(expected_ids - returned_ids)

        coverage[inc_exc] = {
            "total": len(expected_ids),
            "returned": len(returned_ids),
            "missing": missing,
        }

    return {"results": results, "coverage": coverage}
