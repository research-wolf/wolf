import os
import torch
from openai import OpenAI
import json
import sys
from tqdm import tqdm
from const import SYSTEM_MSG, FEW_SHOT
from instruct_generate import PromptGenerator
from mixtral_vllm_api import call_mixtral
import random
import asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

def main(args, client, resume=0):
    part = args.part
    basis = json.load(open(args.seed_dataset, "r"))
    results = []
    if resume:
        resume_dict = json.load(open(f"results/generated-data-mixtral-backup.json", "r"))
        exist_st_ids = {d["study_id"]: 1 for d in resume_dict}
        results.extend(resume_dict)

        for i, resume_dict in enumerate(results):
            resume_dict["id"] = i
            resume_dict["result"] = resume_dict["result"].split("<|im_end|>")[0]

    whole_study = []

    num_st=0
    for json_idx, single_json in enumerate(tqdm(basis, desc="Patient", ncols=75)):
        patient_id = single_json["patient_id"]
        global_ehr = single_json["global"]
        for study in single_json["study"]:
            num_st+=1
            study["patient_id"] = patient_id
            study_id = study["study_id"]
            study.setdefault("q-ehr", []).extend(global_ehr["q-ehr"])
            study.setdefault("a-ehr", []).extend(global_ehr["a-ehr"])
            if resume and exist_st_ids.get(study_id, 0):
                continue
            if len(global_ehr["q-ehr"]) > 5:
                random_indices = random.sample(range(len(global_ehr["q-ehr"])), 5)
                new_q_ehr = []
                new_a_ehr = []
                for index in random_indices:
                    new_q_ehr.append(study["q-ehr"][index])
                    new_a_ehr.append(study["a-ehr"][index])
                study["q-ehr"] = new_q_ehr
                study["a-ehr"] = new_a_ehr
            whole_study.append(study)

    print(f"Total Study: {len(whole_study)}")
    batch = []
    bs = 1
    tq_st = results[-1]["id"]+1 if resume else 0
    tq_ed = num_st

    for n, study in enumerate(tqdm(whole_study, total=tq_ed, initial=tq_st, desc=f"Using Mixtral-8*7B", ncols=75)):
        # if not study["q-ehr"]:
        #     continue
        batch.append(study)
        if len(batch) >= bs:
            call_results = call_mixtral(client, batch, lambda x: PromptGenerator.wrap_gen_message(x))
            tail_id = results[-1]["id"] if results else -1
            call_results[0]["id"] = tail_id + 1
            results.extend(call_results)

            batch = []

        if n % 10 == 0:
            with open(f"results/generated-data-backup.json", "w") as f:
                json.dump(results, f, indent=4)
            print(f">>> Saved backup", end="\t")
        print(f">>> Result Size: {results[-1]['id'] + 1}\n")

    with open(f"results/generated-data-final.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--resume", type=int, default=0)
    args.add_argument("--seed-dataset", type=str, default=1)
    args = args.parse_args()

    resume = args.resume
    os.makedirs("results", exist_ok=True)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url="http://localhost:8000/v1",
    )

    main(args, client, resume)
