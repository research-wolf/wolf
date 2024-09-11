import argparse
import jsonlines
import numpy as np
from itertools import combinations
from collections import defaultdict
import re
def parse_ranks(s: str) -> np.ndarray:
    """Considering the evalution comment contains field wrapped with "+++",
    parse the given comment and outputs the ranks in the form of 1D numpy array."""
    
    score_pattern = r"\d{1,2}\s\d{1,2}"
    split_string = s.split('+++')[1]
    return np.array(split_string.split())

def calculate_score_by_items(keyword, checker, checker_idx, text):
    keyword_to_key = {  'accuracy:':'accuracy',
                        'harmlessness:':'harmlessness',
                        'logicality:':'logicality',
                        'fluency:':'fluency',
                        'informativeness:':'informativeness'}
    
    score_pattern = r"\d"
    if keyword in text and checker[checker_idx] == 0:
        text = re.sub(r"\s+", " ", text)
        # text = re.sub("-", "", text)
        scores = re.findall(score_pattern, text)[0]
        scores = scores.split()
        if len(scores) == 1:
            scores.append( float(scores[0]) )
        if len(scores) > 2:
            scores = scores[:2]
        for i, s in enumerate(scores):
            try:
                scores[i] = float(s)
            except:
                scores[i] = re.sub(r'[^0-9]', "", s)
                if scores[i] == "":
                    scores[i] = 0.0
                else:
                    scores[i] = float(scores[i])
        checker[checker_idx] += 1
        return checker, np.array(scores)
    

def main(args) -> None:
    eval_path = args.evaluation_path
    eval_result = []
    with jsonlines.open(eval_path) as f:
        for line in f.iter():
            eval_result.append(line["decision"])
    
    n_model = 2  
    n_results = len(eval_result)
    ranks = np.empty((n_results, 5, n_model), dtype=np.int8)

    for id, txt in enumerate(eval_result):
        score_board = txt.split('\n')
        checker = [0,0,0,0,0]
        for line in score_board:
            if sum(checker) == 5:
                break
            line = line.strip().lower()
            for ik, keyword in enumerate(['accuracy', 'harmlessness', 'logicality', 'fluency', 'informativeness']):
                if re.search(fr"{keyword}:\s*\d+\s*\d+", line) is not None :
                    checker, scores[id, ik, :] = calculate_score_by_items(keyword, checker, ik, line)
                if sum(checker) == 5:
                    break

    # Calculate
    models = list(combinations(range(n_model), 1))
    for model in models:
        scores = defaultdict()
        # Calculate
        for ik, keyword in enumerate(["accuracy", "helpfulness", "relevance", "hallucination", "universal"]):
            score = scores[:, ik]
        
            # Print
            print(f"{keyword.upper()}: Score is as below.", end=" >>> ")
            print("Score of model{}: {:.3f}".format(model, score))
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--evaluation_path')
    
    args = parser.parse_args()
    main(args)