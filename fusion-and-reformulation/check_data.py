import functools
import json
import logging
import os
import os.path as osp
import pickle
import re
from collections import defaultdict

from tqdm import tqdm

ROOT = "your-mimic-cxr-dataset-path"
VQAROOT = "your-cxrvqa-dataset-path"
EHRROOT = "your-ehrxqa-dataset-path"
SPLITROOT = "physionet.org/files"
METAROOT = "physionet.org/files"


def save_to_json(merged, file_name):
    with open(file_name, "w") as json_file:
        json.dump(merged, json_file, indent=4)


@functools.lru_cache(maxsize=None)  # memoization
def load_json(file_path):
    """
    Loads a JSON file from the specified path.

    Args:
    file_path (str): The path to the JSON file to be loaded.

    Returns:
    dict: The JSON object loaded from the file.
    """
    with open(file_path, "r") as file:
        return json.load(file)


@functools.lru_cache(maxsize=None)
def find_study_to_patient_mapping(root_path=ROOT):
    mapping_file = "utils/study_to_patient.pkl"

    # If the mapping file exists, load it and return
    if os.path.exists(mapping_file):
        with open(mapping_file, "rb") as f:
            return pickle.load(f)

    study_to_patient = {}
    for root, dirs, files in os.walk(root_path):
        for dir_name in dirs:
            if dir_name.startswith("s"):
                patient_id = os.path.basename(root)
                study_to_patient[int(dir_name[1:])] = int(patient_id[1:])
    with open(mapping_file, "wb") as f:
        pickle.dump(study_to_patient, f)

    return study_to_patient


def check_sentences_with_keywords(text, keywords):
    # Use a regular expression to split the text into sentences
    sentences = re.split(r"(?<=[./!?])\s+", text)

    # Check if any sentence contains a keyword or matches a regex pattern
    for sentence in sentences:
        for keyword in keywords:
            # Directly use the keyword as a regular expression pattern
            if re.search(keyword, sentence):
                return True

    return False


def multi_subject_answer(pack, merged):
    flag = False
    if pack["answer"]:
        for answer in pack["answer"]:
            if re.compile(r"^1\d{7}$").match(str(answer)):
                patient_id = "p" + str(answer)
                merged.setdefault(patient_id, {"global": {"q-ehr": [], "a-ehr": []}})
                merged[patient_id]["global"]["q-ehr"].append(pack["template"])
                merged[patient_id]["global"]["a-ehr"].append("Assistant: True.")
                flag = True
    return merged, flag


def convert_binary_answer(pack):
    if check_sentences_with_keywords(pack["template"], ["count", "number", "how many", r"21\d\d"]):
        return pack
    if len(pack["answer"]) == 1:
        if pack["answer"][0] == 1:
            pack["answer"][0] = "Yes."
        elif pack["answer"][0] == 0:
            pack["answer"][0] = "No."
    elif len(pack["answer"]) == 0:
        pack["answer"] = ["Nothing."]
    return pack

def replace_gender_abbreviations_conditional(text):
    if re.fullmatch(r'[MF]+', text):
        text = re.sub(r'M', 'Male', text)
        text = re.sub(r'F', 'Female', text)
        return text
    else:
        return text
    
def modify_sentences(*args, vqa=True):
    def modify_sentence(sentence, is_answer=False, vqa=False):
        role = "Assistant" if is_answer else "User"
        if is_answer:
            sentence = ", ".join(list(map(str, sentence)))
        sentence = sentence.lower()
        sentence = re.sub(r"\s\d{8}[ ,]", " ", f"{role}: " + sentence)
        sentence = re.sub(r"\s\d{8}[?]", "?", sentence)
        sentence = sentence.replace("had patient", "had the patient")
        sentence = re.sub(r"given the study.*?(?=[A-Za-z0-9])", "", sentence)
        sentence = re.sub(r"given the.*?, ", "", sentence)
        sentence = re.sub("list the ids of", "Is the patient", sentence)
        sentence = re.sub("f patients", "female patient", sentence)
        sentence = re.sub("m patients", "male patient", sentence)
        sentence = re.sub("compared to the study", "compared to the other study of this patient", sentence)
        if re.compile(r"list all.*?").search(sentence):
            sentence = re.sub("list all", "Could you list all", sentence)
        sentence = re.sub("list the ids of patients", "Is the patient", sentence)

        temp = sentence.split(": ")
        sentence = temp[0] + ": " + temp[1].capitalize()
        if not is_answer:
            if sentence[-1] != "?":
                sentence = sentence[:-1] + "?"
        return sentence

    res = []
    for arg in args:
        if isinstance(arg, list):
            res.append(modify_sentence(arg, is_answer=True, vqa=vqa))
        else:
            res.append(modify_sentence(arg, vqa=vqa))
    return res


def exclude(data):
    new_data = []
    for d in data:
        if len(d.get("study")) > 0:
            new_data.append(d)
    return new_data


def mix_studyehr_with_studyvqa(data):
    for d in data:
        if len(d.get("study")) > 0:
            for study in d.get("study"):
                if (study.get("q-ehr", None) is not None) and (study.get("q-vqa", None) is not None):
                    study["q-vqa"].extend(study.pop("q-ehr"))
                    study["a-vqa"].extend(study.pop("a-ehr"))
    return data


def transform_dict(input_dict):
    new_json = []
    patient_ids = list(input_dict.keys())
    for patient_id in patient_ids:
        input_dict[patient_id].setdefault("global", {"q-ehr": [], "a-ehr": []})
        global_data = input_dict[patient_id]["global"]

        studies = []

        for study_id, study_data in input_dict[patient_id].items():
            if study_id != "global":
                new_study = {"study_id": study_id, **study_data}
                studies.append(new_study)

        new_json.append({"patient_id": patient_id, "global": global_data, "study": studies})
    return new_json


def merge_vqa_ehr(vqa, ehr, merged):
    global ROOT
    logging.basicConfig(level=logging.INFO)

    study_to_subject = find_study_to_patient_mapping(ROOT)

    # VQA
    for pack in tqdm(vqa, ncols=75):
        if pack["answer"] == []:
            pack["answer"] = ["Nothing."]
        study_id = "s" + pack["study_id"]
        patient_id = "p" + pack["subject_id"]
        pack["question"], pack["answer"] = modify_sentences(pack["question"], pack["answer"], vqa=True)
        merged[patient_id].setdefault("global", {"q-ehr": [], "a-ehr": []})
        merged[patient_id].setdefault(study_id, {"q-ehr": [], "a-ehr": []})
        merged[patient_id][study_id].setdefault("q-vqa", list()).append(pack["question"])
        merged[patient_id][study_id].setdefault("a-vqa", list()).append(pack["answer"])

    for patient_id in merged.keys():
        for study_id in merged[patient_id].keys():
            if study_id == "global":
                continue
            assert len(merged[patient_id][study_id]["q-vqa"]) == len(merged[patient_id][study_id]["a-vqa"]), "q-vqa and a-vqa are not same"

    # EHR
    patient_count = defaultdict(set)
    for pack in tqdm(ehr, ncols=75):
        if check_sentences_with_keywords(
            pack["template"],
            ["ago", "since", "until", "when", r"21\d\d"],
        ):
            continue
        
        pack = convert_binary_answer(pack)
        patient_id = pack["value"].get("patient_id", None)  # subject id
        patient_id = "p" + str(patient_id) if patient_id is not None else None

        pack["template"], pack["answer"] = modify_sentences(pack["template"], pack["answer"])

        if patient_id is None:
            merged, flag = multi_subject_answer(pack, merged)
            if flag:
                continue

        study_ids = [pack["value"].get(f"study_id{i}", None) for i in ["", 1, 2]]
        if patient_id is None:
            if all(study_id is None for study_id in study_ids):
                continue
            else:
                for study_id in study_ids:
                    if study_id is None:
                        continue
                    # Global EHR for One Subject
                    patient_id = "p" + str(study_to_subject[study_id])  # Find Subject Id from Study Id
                    study_id = "s" + str(study_id)
                    merged.setdefault(patient_id, {})
                    merged[patient_id].setdefault("global", {"q-ehr": [], "a-ehr": []})
                    merged[patient_id].setdefault(study_id, {"q-ehr": [], "a-ehr": []})
                    merged[patient_id][study_id].setdefault("q-vqa", list())
                    merged[patient_id][study_id].setdefault("a-vqa", list())
                    merged[patient_id][study_id]["q-ehr"].append(pack["template"])
                    merged[patient_id][study_id]["a-ehr"].append(pack["answer"])
                    patient_count[patient_id].add(study_id)

        # patient_id is not None
        else:
            merged.setdefault(patient_id, {})
            merged[patient_id].setdefault("global", {"q-ehr": [], "a-ehr": []})

            if all(study_id is None for study_id in study_ids):
                # Only Subject Id
                # merged[patient_id].setdefault("global", dict()).setdefault("q-ehr", list()).append(pack["question"])
                merged[patient_id]["global"]["q-ehr"].append(pack["template"])
                merged[patient_id]["global"]["a-ehr"].append(pack["answer"])
                patient_count[patient_id].add("global")
                continue
            else:
                for study_id in study_ids:
                    if study_id is None:
                        continue
                    study_id = "s" + str(study_id)
                    merged[patient_id].setdefault(study_id, {"q-ehr": [], "a-ehr": []})
                    merged[patient_id][study_id]["q-ehr"].append(pack["template"])
                    merged[patient_id][study_id]["a-ehr"].append(pack["answer"])
                    patient_count[patient_id].add(study_id)

    for patient_id in merged.keys():
        for study_id in merged[patient_id].keys():
            assert (
                # len(merged[patient_id][study_id]["q-ehr"]) == len(merged[patient_id][study_id]["q-ehr"]) == len(merged[patient_id][study_id]["a-ehr"])
                len(merged[patient_id][study_id]["q-ehr"])
                == len(merged[patient_id][study_id]["a-ehr"])
            ), "q-ehr and q-ehr and a-ehr are not same"

    for patient_id in merged.keys():
        for study_id in merged[patient_id].keys():
            if study_id == "global":
                continue
            merged[patient_id][study_id]["file_path"] = osp.join(osp.join(*ROOT.split(os.sep)[4:]), patient_id[:3], patient_id, study_id)

    return merged

def main():
    global VQAROOT, EHRROOT, SPLITROOT, ROOT
    splits = ["train", "valid", "test"]
    split_json_data = load_json(osp.join(SPLITROOT, "all-reports-splits.json"))
    meta_json_data = load_json(osp.join(METAROOT, "mimic-metadata.json"))
    for split in splits:
        dst_path = "datasets/ehr-vqa-merged"
        merged = defaultdict(dict)
        vqa_json_data = load_json(osp.join(VQAROOT, split + ".json"))
        ehr_json_data = load_json(osp.join(EHRROOT, split + ".json"))

        # merging start
        merged = merge_vqa_ehr(vqa_json_data, ehr_json_data, merged)

        for pack in tqdm(split_json_data, ncols=50):
            patient_id = pack["subject_id"]
            study_id = pack["id"]
            gt_report = pack["raw"]["findings"] + " " + pack["raw"]["impression"]
            gt_report = re.sub(r"\s\s", " ", gt_report)
            gt_report = re.sub(r"\n", " ", gt_report)
            gt_report.rstrip()
            try:
                merged[patient_id][study_id]["gt_report"] = gt_report
            except KeyError:
                continue

        for pack in tqdm(meta_json_data, ncols=50):
            patient_id = "p" + str(pack["subject_id"])
            study_id = "s" + str(pack["study_id"])
            study_date = re.sub(r"(\d{4})(\d{2})(\d{2})", r"\1-\2-\3", str(pack["StudyDate"]))
            dicom_id = pack["dicom_id"]
            view_position = pack["ViewPosition"]
            try:
                merged[patient_id][study_id]["study_date"] = study_date
                merged[patient_id][study_id].setdefault("dicom_id", list()).append(dicom_id)
                merged[patient_id][study_id].setdefault("view_position", list()).append(view_position)
            except KeyError:
                continue

        merged = transform_dict(merged)
        merged = exclude(merged)
        merged = mix_studyehr_with_studyvqa(merged)

        logging.info(f"Number of subjects: {len(merged)}")
        logging.info(f"Number of studies: {sum([len(d['study']) for d in merged])}")
        no_ehr_p = 0
        for mm in merged:
            if mm["global"]["q-ehr"]:
                no_ehr_p += 1
        logging.info(f"Number of subjects without ehr: {no_ehr_p}")
        save_to_json(merged, osp.join(dst_path, f"ehr-vqa-merged-all-{split}" + ".json"))

if __name__ == "__main__":
    main()
