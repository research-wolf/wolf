import json 
import os
import jsonlines
import argparse
from collections import defaultdict
import re

from sympy import pretty

def yes_or_no(our_answer, answer):
    yes_pat = r"(^|\W)yes(\W|$)"
    no_pat = r"(n't)|((^|\W)no(\W|$))"
    s = 0
    if all([not bool(re.search(pat, answer.lower())) for pat in [yes_pat, no_pat]]):
        if not bool(re.search(no_pat, our_answer.lower())):
            s += 1
    if all([bool(re.search(yes_pat, st)) for st in [answer.lower(), our_answer.lower()]]):
        s += 1
    if all([bool(re.search(no_pat, st)) for st in [answer.lower(), our_answer.lower()]]):
        s += 1
    if bool(re.search(yes_pat, answer.lower())):
        if not bool(re.search(no_pat, our_answer.lower())):
            s += 1
    return True if s > 0 else False

def short_answer_question(our_answer, answer):
    beam = answer.split()
    our_beam = our_answer.split()
    no_pat = r"(n't)|((^|\W)no(\W|$))"
    if re.search(no_pat, answer):
        return False
    count = len(beam)
    for word in beam:
        for our_word in our_beam:
            if word in our_word:
                count-=1
            if count<2:
                return True

def scoring(score_template):
    score = {k: round(100 * sum(v) / len(v), 3) for k, v in score_template.items()}
    pretty_score = ""
    for k, v in score.items():
        pretty_score += f"{k}: {v}\n"
        
    pretty_score += f"\n=======Total: {round(sum(score.values()) / len(score), 3)}"
    return pretty_score

def main(our_answer_path, answer_path):
    score_template = defaultdict(list)
    yes_pat = r"(^|\W)yes(\W|$)|(sure)"
    no_pat = r"(n't)|((^|\W)no(\W|$))|(not)"

    for our_line, line in zip(our_answer_path, answer_path):
        our_answer = our_line["text"].lower()
        answer = line["text"].lower()
        answer_type = line["answer_type"]
        if re.search(yes_pat, answer) or re.search(no_pat, answer):
            correct = yes_or_no(our_answer, answer)
        else:
            correct = short_answer_question(our_answer, answer)

        if correct :
            score_template[answer_type].append(1)
        else :
            score_template[answer_type].append(0)

    print(scoring(score_template))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--our-answer-path", type=str, help="Path.")
    parser.add_argument("--answer-path", type=str, help="Path.")

    args = parser.parse_args()
    
    answer_path = []
    with open(args.answer_path, "r") as f:
        for line in f:
            answer_path.append( json.loads(line) )
    
    our_answer_path = []
    with open(args.our_answer_path, "r") as f:
        for line in f:
            our_answer_path.append( json.loads(line) )
            
    main(our_answer_path, answer_path)
