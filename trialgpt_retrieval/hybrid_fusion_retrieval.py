# GPU-Optimized hybrid_fusion_retrieval.py
# - FAISS now uses GPU
# - MedCPT model usage logs GPU device
# - Code logic unchanged otherwise

import json
import os
import torch
import tqdm
import nltk
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
from rank_bm25 import BM25Okapi
from nltk import word_tokenize

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
nltk.download('punkt')

###########################################
# Step i: Helper code for Device
###########################################

def _to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}

##########################################
# Step 0: Load patient raw notes
##########################################

def load_patient_raw_notes(queries_path):
    id2rawnotes = {}
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            pid = entry["_id"]
            note_id = entry["note_id"]
            if pid not in id2rawnotes:
                id2rawnotes[pid] = {}
            id2rawnotes[pid][note_id] = entry["text"]
    return id2rawnotes

##########################################
# Step 1: Load patient conditions
##########################################
def load_patient_conditions(enriched_queries_path):
    with open(enriched_queries_path, "r", encoding="utf-8") as f:
        all_queries = json.load(f)

    id2conditions = {}
    id2notes = {}
    for pid, notes_dict in all_queries.items():
        id2conditions[pid] = {}
        id2notes[pid] = {}
        for note_id, entry in notes_dict.items():
            conds = entry.get("conditions", [])
            cond_texts = [c["text"] for c in conds if "text" in c]
            id2conditions[pid][note_id] = cond_texts
            id2notes[pid][note_id] = entry.get("summary", "")
    return id2conditions, id2notes

##########################################
# Step 2: Load trial corpus
##########################################
def load_corpus(corpus_path):
    corpus = {}
    with open(corpus_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            corpus[entry["_id"]] = entry
    return corpus

##########################################
# Step 3: Build BM25 Index
##########################################
def build_bm25_index(corpus_path):
    tokenized_corpus = []
    corpus_nctids = []
    with open(corpus_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            corpus_nctids.append(entry["_id"])
            tokens = word_tokenize(entry["title"].lower()) * 3
            if "metadata" in entry and "diseases_list" in entry["metadata"]:
                for disease in entry["metadata"]["diseases_list"]:
                    tokens += word_tokenize(disease.lower()) * 2
            tokens += word_tokenize(entry["text"].lower())
            tokenized_corpus.append(tokens)
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, corpus_nctids

##########################################
# Step 4: Build MedCPT Index
##########################################
def build_medcpt_index(corpus_path, embeds_path, nctids_path):
    if os.path.exists(embeds_path) and os.path.exists(nctids_path):
        # Case 1: Prebuilt index files exist
        print(f"Loading MedCPT index from {embeds_path} and {nctids_path}...")
        embeds = np.load(embeds_path)
        nctids = json.load(open(nctids_path))
    else:
        # Case 2: Build from scratch
        print("MedCPT embeddings not found. Building index from corpus.jsonl...")
        embeds = []
        nctids = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder")
        model = torch.nn.DataParallel(model)  # spreads across all GPUs
        model = model.to(device)

        tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")

        with open(corpus_path, "r") as f:
            for line in tqdm.tqdm(f.readlines(), desc="Encoding Trial Corpus with MedCPT"):
                entry = json.loads(line)
                nctids.append(entry["_id"])
                title, text = entry["title"], entry["text"]
                inputs = tokenizer([[title, text]], padding=True, truncation=True, return_tensors="pt", max_length=512)
                # move to same device as model
                inputs = _to_device(inputs, device)
                with torch.no_grad():
                    outputs = model(**inputs)
                    embed = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    embeds.append(embed[0])

        embeds = np.array(embeds)
        np.save(embeds_path, embeds)
        with open(nctids_path, "w") as f:
            json.dump(nctids, f, indent=4)

    # FAISS GPU
    dim = embeds.shape[1]
    ngpus = faiss.get_num_gpus()
    res = [faiss.StandardGpuResources() for _ in range(ngpus)]
    index_flat = faiss.IndexFlatIP(dim)
    index = faiss.index_cpu_to_all_gpus(index_flat)
    index.add(embeds)
    print("FAISS index built on GPU")
    return index, nctids

##########################################
# Step 5: Match Trials per Patient
##########################################
def retrieve_trials(patient_conditions, bm25, bm25_nctids, medcpt, medcpt_nctids, tokenizer, model, k_fusion=20, bm25_wt=1, medcpt_wt=1, topN=50):
    qid2nctids = {}
    for pid, notes in tqdm.tqdm(patient_conditions.items()):
        qid2nctids[pid] = {}  # Initialize per-patient dict
        for note_id, conditions in notes.items():
            if not conditions:
                qid2nctids[pid][note_id] = []
                continue

            # --- BM25 ---
            bm25_scores = {}
            for condition in conditions:
                tokens = word_tokenize(condition.lower())
                top_nctids = bm25.get_top_n(tokens, bm25_nctids, n=topN)
                for rank, nctid in enumerate(top_nctids):
                    bm25_scores[nctid] = bm25_scores.get(nctid, 0) + (1 / (rank + k_fusion))

            # --- MedCPT ---
            with torch.no_grad():
                query_text = " ".join(conditions)
                encoded = tokenizer(query_text, truncation=True, padding=True, return_tensors="pt", max_length=256)
                #move to the same device as the model
                encoded = _to_device(encoded, next(model.parameters()).device)
                embeds = model(**encoded).last_hidden_state[:, 0, :].cpu().numpy()
                scores, inds = medcpt.search(embeds, k=topN)

            medcpt_scores = {}
            for ind_list in inds:
                for rank, idx in enumerate(ind_list):
                    nctid = medcpt_nctids[idx]
                    medcpt_scores[nctid] = medcpt_scores.get(nctid, 0) + (1 / (rank + k_fusion))

            # --- Fusion ---
            nctid2score = {}
            for nctid in set(bm25_scores) | set(medcpt_scores):
                nctid2score[nctid] = (
                    bm25_wt * bm25_scores.get(nctid, 0) +
                    medcpt_wt * medcpt_scores.get(nctid, 0)
                )
            sorted_nctids = sorted(nctid2score.items(), key=lambda x: -x[1])
            qid2nctids[pid][note_id] = [nctid for nctid, _ in sorted_nctids[:topN]]
    return qid2nctids


##########################################
# Step 5a: Build Final Retrieved Trials Output
##########################################

def extract_inclusion_exclusion(text):
    """
    Parses the text field to extract inclusion and exclusion criteria if present.
    Returns (inclusion_criteria, exclusion_criteria)
    """
    if not text:
        return "", ""

    inclusion = ""
    exclusion = ""

    lowered = text.lower()
    inc_idx = lowered.find("inclusion criteria")
    exc_idx = lowered.find("exclusion criteria")

    if inc_idx != -1 and exc_idx != -1 and exc_idx > inc_idx:
        inclusion = text[inc_idx:exc_idx].strip()
        exclusion = text[exc_idx:].strip()
    elif inc_idx != -1:
        inclusion = text[inc_idx:].strip()
    elif exc_idx != -1:
        exclusion = text[exc_idx:].strip()

    return inclusion, exclusion



##########################################
# Step 6: Build Final Retrieved Trials Output
##########################################
def build_final_output(qid2nctids, id2notes, id2rawnotes, corpus, output_path):
    final_results = []

    for pid, note_dict in qid2nctids.items():
        for note_id, nctids in note_dict.items():
            patient_entry = {
                "patient_id": pid,
                "note_id": note_id,
                "patient": id2notes.get(pid, {}).get(note_id, ""),
                "patient_raw": id2rawnotes.get(pid, {}).get(note_id, ""),
                "0": [],
                "1": [],
                "2": []
            }

            relevance_buckets = {"0": 5, "1": 10, "2": 15}
            current_idx = 0

            for rel_label, num_trials in relevance_buckets.items():
                trials = []
                for _ in range(num_trials):
                    if current_idx >= len(nctids):
                        break
                    nctid = nctids[current_idx]
                    trial = corpus.get(nctid, None)
                    if trial:
                        meta = trial.get("metadata", {})
                        inclusion = meta.get("inclusion_criteria", "")
                        exclusion = meta.get("exclusion_criteria", "")
                        if not inclusion or not exclusion:
                            inclusion_from_text, exclusion_from_text = extract_inclusion_exclusion(trial.get("text", ""))
                            if not inclusion:
                                inclusion = inclusion_from_text
                            if not exclusion:
                                exclusion = exclusion_from_text

                        trials.append({
                            "brief_title": trial.get("title", ""),
                            "phase": meta.get("phase", ""),
                            "drugs": str(meta.get("drugs_list", [])),
                            "drugs_list": meta.get("drugs_list", []),
                            "diseases": str(meta.get("diseases_list", [])),
                            "diseases_list": meta.get("diseases_list", []),
                            "enrollment": meta.get("enrollment", ""),
                            "inclusion_criteria": inclusion,
                            "exclusion_criteria": exclusion,
                            "brief_summary": trial.get("text", ""),
                            "NCTID": trial.get("_id", "")
                        })
                    current_idx += 1
                patient_entry[rel_label] = trials

            final_results.append(patient_entry)

    with open(output_path, "w") as f:
        json.dump(final_results, f, indent=4)
    print(f"Retrieved trials saved to: {output_path}")

##########################################
# MAIN
##########################################
if __name__ == "__main__":
    queries_path = f"../data/queries.jsonl"
    # corpus_name = "sigir"  # or "trec2021", "trec2022", whatever

    # corpus_path = f"../data/{corpus_name}_corpus.jsonl"
    corpus_path = f"../results/corpus.jsonl"

    enriched_queries_path=f"../results/retrieval_keywords_enriched.json"

    # embeds_path = f"../trialgpt_retrieval/{corpus_name}_embeds.npy"
    embeds_path = f"../trialgpt_retrieval/sourcename_embeds.npy"
    # nctids_path = f"../trialgpt_retrieval/{corpus_name}_nctids.json"
    nctids_path = f"../trialgpt_retrieval/sourcename_nctids.json"

    outputfile= f"../results/retrieved_trials.json"

    # Load patient raw queries (notes)

    id2rawnotes = load_patient_raw_notes(queries_path)

    # Load patient queries (conditions)
    id2conditions, id2notes = load_patient_conditions("../results/retrieval_keywords_enriched.json")
    # print(id2conditions)
    # print(id2notes)

    # Load trial corpus
    corpus = load_corpus(corpus_path)

    # print(corpus)

    # Load BM25 index
    bm25, bm25_nctids = build_bm25_index(corpus_path)
    # print(bm25)
    # print(bm25_nctids)

    # Load MedCPT index
    medcpt, medcpt_nctids = build_medcpt_index(corpus_path, embeds_path, nctids_path)
    # print(medcpt)
    # print(medcpt_nctids)

    # Load MedCPT model and tokenizer (use CPU)

    model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder", trust_remote_code=True).cuda()
    tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder", trust_remote_code=True)
    device = next(model.parameters()).device

    # Retrieve Trials
    qid2nctids = retrieve_trials(id2conditions, bm25, bm25_nctids, medcpt, medcpt_nctids, tokenizer, model)
    # print(qid2nctids)

    # Build final output
    build_final_output(qid2nctids, id2notes, id2rawnotes, corpus, outputfile)

    print("✅ End-to-End Trial Matching Done!")