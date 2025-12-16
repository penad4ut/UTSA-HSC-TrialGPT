import json
import re
from llama_cpp import Llama
import os


def convert_criteria_pred_to_string(prediction, trial_info):

    output = ""
    for inc_exc in ["inclusion", "exclusion"]:
        section_header = "Inclusion Criteria" if inc_exc == "inclusion" else "Exclusion Criteria"
        print(f"Section Header: {section_header}")
        output += f"\n{section_header}:\n"

        idx2criterion = {}
        criteria = trial_info.get(f"{inc_exc}_criteria", "").split("\n\n")
        print(f"Criteria: {criteria}")
        idx = 0
        for criterion in criteria:
            criterion = criterion.strip()
            if "inclusion criteria" in criterion.lower() or "exclusion criteria" in criterion.lower():
                continue
            if len(criterion) < 5:
                continue
            idx2criterion[str(idx)] = criterion
            idx += 1

        preds_dict = prediction.get(inc_exc, {})
        print(f"Preds_dict: {preds_dict}")
        if isinstance(preds_dict, list):
            # structured outputs
            for item in preds_dict:
                cid = str(item.get("criterion_id"))
                criterion = idx2criterion.get(cid, f"[criterion {cid}]")
                output += f"{cid}. {criterion}\n"
                output += f"\tReasoning: {item.get('reasoning')}\n"
                output += f"\tLabel: {item.get('label')}\n"
                print(f"Output from If clause: {output}")
        elif isinstance(preds_dict, dict):
            # dict-of-lists style (legacy)
            for cid, vals in preds_dict.items():
                if cid not in idx2criterion:
                    continue
                if not isinstance(vals, list) or len(vals) < 2:
                    continue
                output += f"{cid}. {idx2criterion[cid]}\n"
                output += f"\tReasoning: {vals[0]}\n"
                output += f"\tLabel: {vals[1]}\n"
                print(f"Output from Elif: {output}")

    return output


def safe_json_parse(response_text):

    try:
        raw = response_text.strip().replace("```json", "").replace("```", "").strip()
        # trim to last closing brace
        if "}" in raw:
            raw = raw[:raw.rfind("}")+1]
        # clean trailing commas
        raw = re.sub(r",\s*([\]}])", r"\1", raw)
        parsed = json.loads(raw)
    except Exception as e:
        return {
            "relevance_score_R": -1,
            "relevance_explanation": f"PARSE_ERROR: {e}",
            "eligibility_score_E": -1,
            "eligibility_explanation": f"PARSE_ERROR: {e}"
        }

    # enforce required keys
    if "relevance_score_R" not in parsed:
        parsed["relevance_score_R"] = -1
        parsed["relevance_explanation"] = "MISSING"
    if "eligibility_score_E" not in parsed:
        parsed["eligibility_score_E"] = -1
        parsed["eligibility_explanation"] = "MISSING"

    return parsed


def trialgpt_aggregation(trial_id, patient_id, trial_info, match_block, patient_note):
    pred_str = convert_criteria_pred_to_string(match_block.get("results", {}), trial_info)

    prompt = (
        f"Patient ID: {patient_id}\n\n"
        f"{patient_note}\n\n"
        f"Clinical Trial ID: {trial_id}\n"
        f"Clinical Trial Title: {trial_info.get('brief_title', '')}\n"
        f"Diseases: {', '.join(trial_info.get('diseases_list', []))}\n"
        f"Summary: {trial_info.get('brief_summary', '')}\n\n"
        f"Eligibility Reasoning:\n{pred_str}\n\n"
        f"Based on the above, provide:\n"
        f"- A relevance score R (0-100) indicating how relevant the patient is to the trial.\n"
        f"- An eligibility score E (0-100) based on inclusion/exclusion match.\n"
        f"- A short explanation for each.\n"
        f"Output must be ONLY a single valid JSON object with exactly these fields:\n"
        f'{{"relevance_score_R": <int>, "relevance_explanation": <str>, '
        f'"eligibility_score_E": <int>, "eligibility_explanation": <str>}}\n'
    )

    # ✅ Save prompt context to file
    os.makedirs("../results/prompt_logs", exist_ok=True)
    log_file = f"../results/prompt_logs/{patient_id}_{trial_id}.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(prompt)

    # path for the local model
    model_path = f"model"
    print(f"[INFO] Loading model from {model_path} ...")
    model = Llama(model_path=model_path,
                  n_ctx=32768,
                  n_gpu_layers=999,
                  f16_kv=True,
                  use_mlock=True,
                  verbose=False)

    print("===== AGGREGATION DEBUG =====")
    print(f"Patient ID: {patient_id}")
    print(f"Trial ID: {trial_id}")
    print("Prompt Preview:\n", prompt[:500], "..." if len(prompt) > 500 else "")
    print(f"{pred_str}")
    print(f"Prompt Full: {prompt}")
    print("=============================")

    try:
        completion = model.create_completion(
            prompt=prompt,
            max_tokens=2048,
            temperature=0.0,
            stop=["</s>"]
        )
        raw_text = completion["choices"][0]["text"].strip()
        parsed = safe_json_parse(raw_text)
    except Exception as e:
        parsed = {
            "relevance_score_R": -1,
            "relevance_explanation": f"ERROR: {e}",
            "eligibility_score_E": -1,
            "eligibility_explanation": f"ERROR: {e}"
        }

    return parsed
