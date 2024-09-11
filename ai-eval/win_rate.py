"""
DESCRIPTIONS
    Given evaluation result (rank, not score!), output win rates.
    Note that the result file should be jsonl file, and must contain
    "text" key as the output of your AI evaluator.
    Also, the value of the "text" key should contain the rank field
    which can be splited from other texts using "+++".
    For example, a line of your evaluation result file should be as follows:
        {..., "text": "...+++1 3 2 4+++...", ...}
EXAMPLE    
    >>> python evaluate_win_rate -e {{your_eval_result_path}}
    Win Rates are as below.
    Win Rate of model0 vs. model1: 0.625
    Win Rate of model0 vs. model2: 0.750
    Win Rate of model0 vs. model3: 1.000
    Win Rate of model1 vs. model2: 0.750
    Win Rate of model1 vs. model3: 1.000
    Win Rate of model2 vs. model3: 0.750
"""
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
    keyword_to_key = {  'naturalness:':'naturalness',
                        'coherence:':'coherence',
                        'engagingness:':'engagingness',
                        'groundeness:':'groundeness'}
    
    
    score_pattern = r"\d{1,2}\s\d{1,2}"
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
            line = re.sub("total score", "universal", line)
            for ik, keyword in enumerate(['naturalness', 'coherence', 'engagingness', 'groundeness']):
                if re.search(fr"{keyword}:\s*\d+\s*\d+", line) is not None :
                    checker, ranks[id, ik, :] = calculate_score_by_items(keyword, checker, ik, line)
                if sum(checker) == 5:
                    break

    # Calculate Win Rate
    pairs = list(combinations(range(n_model), 2))
    for pair in pairs:
        win_rates = defaultdict()
        # Calculate Win Rate of model1 vs. model2
        for ik, keyword in enumerate(['naturalness', 'coherence', 'engagingness', 'groundeness']):
            rank_1 = ranks[:, ik, pair[0]]
            rank_2 = ranks[:, ik, pair[1]]
            win_rates[pair] = np.mean(rank_1 > rank_2)
        
            # Print
            print(f"{keyword.upper()}: Win Rates are as below.", end=" >>> ")
            print("Win Rate of model{} vs. model{}: {:.3f}".format(pair[0], pair[1], win_rates[(pair[0], pair[1])]))
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--evaluation_path')
    
    args = parser.parse_args()
    main(args)