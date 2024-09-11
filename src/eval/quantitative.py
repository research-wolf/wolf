"""
DESCRIPTION
    Evaluate BLEU-1,2,3,4 / ROUGE-L / METOER, given two jsonl files.
    One is the candidate file, and the other one is reference file.
    Considering the reference file as the ground truth, this code evaluates the score of the candidate.
    Note that two files must contain "dicom_id" and "text" as keys.
    Thus the files will be aggregation of {"dicom_id": <id>, "text": <sentence>, ...} elements.
    
EXAMPLE
    >>> python evaluate_report.py -c your_candidate.jsonl -r your_reference.jsonl
"""

import jsonlines
import argparse
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
import nltk
nltk.download("punkt")
from nltk.tokenize import word_tokenize
from collections import defaultdict
import json

def preprocess(s: str) -> str:
    s = s.replace("\n", "")
    s = s.replace("<s>", "")
    s = s.replace("</s>", "")
    return s


def main(args) -> dict[str, float]:
    model_prediction_path = args.candidate_path
    reference_path = args.reference_path

    # Loading the evaluators
    scorers = [(Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]), (Meteor(), "METEOR"), (Rouge(), "ROUGE_L")]
    # Get predictions & references
    predictions = defaultdict()
    with jsonlines.open(model_prediction_path) as f:
        for line in f.iter():
            predictions[line["question_id"]] = line["text"]
    references = defaultdict()
    reference_data = json.load(open(reference_path, "r"))
    for qid, line in enumerate(reference_data):
        if line['report'] is None:
            continue
        if qid > list(predictions.keys())[-1]:
            break
        references[qid] = line["report"]
    assert predictions.keys() == references.keys()
    # Pre-process sentences
    print("Tokenization...")
    for dicom in references.keys():
        pred_text = " ".join(word_tokenize(preprocess(predictions[dicom]))).lower()
        ref_text = " ".join(word_tokenize(preprocess(references[dicom]))).lower()

        predictions[dicom] = [pred_text]
        references[dicom] = [ref_text]
    # Compute scores
    final_scores = {}
    print("Evaluation Result")
    for scorer, method in scorers:
        print(f"Computing {scorer.method()} score...")
        if type(method) == list:
            score, scores = scorer.compute_score(references, predictions, verbose=0)
            for sc, scs, m in zip(score, scores, method):
                final_scores[m] = sc
                print("%s: %0.3f" % (m, sc))
        else:
            score, scores = scorer.compute_score(references, predictions)
            print("%s: %0.3f" % (method, score))
            final_scores[method] = score

    return final_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--candidate-path")  # must contain dicom_id, text keys
    parser.add_argument("-r", "--reference-path")

    args = parser.parse_args()
    main(args)
