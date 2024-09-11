import glob
import json
import os
import random
import re
from collections import defaultdict, deque

from tqdm import tqdm

# Knowledge graph G
KEYWORD_SETS = dict(
    heart=["heart", "cardiomegaly"],
    spine=["spine", "scoliosis"],
    pleural=["pleural", "effusion", "thickening", "pneumothorax"],
    bone=["bone"],
    lung=["lung", "emphysema", "pneumonia", "edema", "atelectasis", "cicatrix", "opacity", "lesion"],
    mediastinum=["mediastinum", "hernia", "calcinosis"],
    airspace=["airspace", "hypoinflation"],
)

NORMAL_SENS = [
    "There is no evidence of {disease}.",
    "No disease of {disease}",
    "There is no disease of {disease}",
]

ROOT_PATH = "your-mimic-cxr-report-path"


def get_file_paths(directory=ROOT_PATH):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, start=directory)
            relative_path = relative_path.replace(os.sep, "/")
            file_paths.append(relative_path)
    file_paths.sort()
    return file_paths


def read_text_file(file_path):
    file_path = os.path.join(ROOT_PATH, file_path)
    _, ext = os.path.splitext(file_path)
    if ext != ".txt":
        return None

    try:
        with open(file_path, "r") as file:
            lines = file.readlines()[1:]
            return "".join(lines)
    except IOError as e:
        return None


def extract_key_value_pairs(report):
    lines = report.split("\n")
    report_json = {}
    current_key = None
    current_value = []

    for line in lines:
        if ":" in line and line.split(":")[0].isupper():
            if current_key:
                report_json[current_key] = " ".join(current_value).strip()
            current_key = line.split(":")[0].strip()
            current_value = [line.split(":", 1)[1].strip()]
        else:
            current_value.append(line.strip())

    # Adding the last key-value pair if exists
    if current_key:
        report_json[current_key] = " ".join(current_value).strip()

    # Modify string
    for key, value in report_json.items():
        if value == "None." or value == "None":
            report_json[key] = None
            continue
        report_json[key] = re.sub(r"\s+", " ", value)

    return json.dumps(report_json, indent=2)


def save_json_to_file(json_data, file_path):
    try:
        with open(file_path, "w") as file:
            file.write(json_data)
    except IOError as e:


def contains_word(sentence, word):
    return word in sentence


def convert_to_lower(sentence):
    return sentence.lower()


def main():
    output_path = "datasets"
    all_reports = []

    paths = get_file_paths()
    empty_cnt = 0
    for text_file_path in tqdm(paths):
        report_text = read_text_file(text_file_path)
        if report_text is None or len(report_text) == 0:
            continue

        report_json = json.loads(extract_key_value_pairs(report_text))

        report_data = defaultdict(list)
        report_data.update(
            {
                "id": os.path.splitext(text_file_path)[0].split("/")[-1],
                "filename": text_file_path,
                "subject_id": os.path.splitext(text_file_path)[0].split("/")[1],
                "raw": {"findings": report_json.get("FINDINGS", ""), "impression": report_json.get("IMPRESSION", "")},
            }
        )

        if report_data["raw"]["findings"] == "" and report_data["raw"]["impression"] == "":
            empty_cnt += 1
            print(f"Empty report: {text_file_path}, {empty_cnt}")
            continue

        fi = convert_to_lower(report_json.get("FINDINGS", "") + " " + report_json.get("IMPRESSION", ""))
        fi_sens = fi.split(".")
        fi_words = [fi_sen.split(" ") for fi_sen in fi_sens]
        for sen in fi_sens:
            for key, vals in KEYWORD_SETS.items():
                for v in vals:
                    if contains_word(sen, v):
                        report_data[key].append(sen.replace("\n", "").strip())
                        break

        for key in KEYWORD_SETS.keys():
            if report_data.get(key, None) is None:
                report_data[key] = [NORMAL_SENS[random.randint(0, len(NORMAL_SENS) - 1)].format(disease=key)]

        all_reports.append(report_data)

    with open(os.path.join(output_path, "organized-report.json"), "w") as json_file:
        json.dump(all_reports, json_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
