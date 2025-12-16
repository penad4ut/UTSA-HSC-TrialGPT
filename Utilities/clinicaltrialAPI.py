import requests
import json
from time import sleep
import re


# ----------------------------
# Criteria Normalization
# ----------------------------



def normalize_criteria_block(criteria_text: str) -> str:
    """
    Normalize inclusion/exclusion criteria so sub-bullets following a parent
    ending with ':' are merged into one block, and stop when a new top-level
    criterion (numbered) begins. Handles cases like '\n9.' or '   9.' too.
    """
    if not criteria_text or criteria_text == "N/A":
        return ""

    lines = criteria_text.split("\n")
    normalized = []
    current = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Detect new top-level numbered criterion (even if leading spaces)
        if re.match(r"^\s*\d+\.", line):
            # flush old block
            if current:
                normalized.append(current)
            current = stripped

        # Sub-condition (bullets or indented continuation)
        elif line.startswith(" ") or stripped.startswith(("*", "-")):
            if current:
                current += " " + stripped
            else:
                current = stripped

        # Continuation of the same criterion
        else:
            if current:
                current += " " + stripped
            else:
                current = stripped

    if current:
        normalized.append(current)

    return "\n".join(normalized)




# ----------------------------
# Fetch Trial Details
# ----------------------------
def fetch_trial_details(nctid):
    url = f"https://clinicaltrials.gov/api/v2/studies/{nctid}"  # ✅ API v2
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.json()

        if "protocolSection" not in data:
            return None

        proto = data["protocolSection"]

        # === Extract fields ===
        brief_title = proto["identificationModule"].get("briefTitle", "N/A")
        brief_summary = proto["descriptionModule"].get("briefSummary", "N/A")
        diseases_list = proto.get("conditionsModule", {}).get("conditions", [])
        interventions = proto.get("armsInterventionsModule", {}).get("interventions", [])
        drug_list = [item["name"] for item in interventions if item["type"] == "DRUG"]

        # Handle enrollment
        enrollment_info = proto["designModule"].get("enrollmentInfo", {})
        enrollment = enrollment_info.get("count", "N/A")
        if isinstance(enrollment, int):
            enrollment = str(enrollment)

        # Handle phase
        phases = proto["designModule"].get("phases", [])
        phase = phases[0] if phases else "N/A"
        if phase.startswith("PHASE"):
            phase = phase.replace("PHASE", "Phase")

        # Extract inclusion/exclusion
        eligibility_text = proto.get("eligibilityModule", {}).get("eligibilityCriteria", "N/A")
        inclusion_criteria = exclusion_criteria = ""
        if eligibility_text != "N/A":
            if "Exclusion Criteria:" in eligibility_text:
                inclusion_criteria, exclusion_criteria = eligibility_text.split("Exclusion Criteria:", 1)
                inclusion_criteria = inclusion_criteria.strip()
                exclusion_criteria = exclusion_criteria.strip()
            else:
                inclusion_criteria = eligibility_text.strip()

        # ✅ Normalize criteria before saving
        inclusion_criteria = normalize_criteria_block(inclusion_criteria)
        exclusion_criteria = normalize_criteria_block(exclusion_criteria)

        # === Build Text ===
        full_text = f"Summary: {brief_summary}\n"

        # === Final trial entry ===
        trial_entry = {
            "_id": nctid,
            "title": brief_title,
            "text": full_text,
            "metadata": {
                "brief_title": brief_title,
                "phase": phase,
                "drugs": str(drug_list),
                "drugs_list": drug_list,
                "diseases": str(diseases_list),
                "diseases_list": diseases_list,
                "enrollment": enrollment,
                "inclusion_criteria": inclusion_criteria,
                "exclusion_criteria": exclusion_criteria,
                "brief_summary": brief_summary,
            },
        }

        return trial_entry

    except requests.exceptions.RequestException as e:
        print(f"Error fetching {nctid}: {e}")
        return None


# ----------------------------
# Fetch and Save Trials
# ----------------------------
nctids = ['NCT06516887']  # You can expand this list

outputpath = f"results/corpus.jsonl"

with open(outuputpath, "w", encoding="utf-8") as f:
    for nctid in nctids:
        trial = fetch_trial_details(nctid)
        if trial:
            f.write(json.dumps(trial) + "\n")
            print(f"Saved trial {nctid}")
        sleep(0.2)  # avoid API rate limit

print("All trials retrieved and saved correctly to corpus.jsonl")
