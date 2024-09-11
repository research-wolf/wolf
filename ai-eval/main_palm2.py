from openai import OpenAI
import argparse
import json
import jsonlines
from consts import CRITERIA, CRITERIA_rev, CRITERIA_PROMPT, CRITERIA_wr, REFERENCE_PROMPT, ANSWER_PROMPT
import random
import os
import numpy as np
import logging
import pathlib
import textwrap

import google.generativeai as genai
import vertexai
from vertexai.language_models import ChatModel, InputOutputTextPair, TextGenerationModel
import re

def build_sending_message(query, ref_ans, our_ans, compare_ans, reverse=False, wr=False):
    if wr:
        criteria = CRITERIA_wr
        message = CRITERIA_PROMPT.format(question=query, 
                                       ref_ans=ref_ans,
                                       ans_1=compare_ans,
                                       ans_2=our_ans)
        return criteria, message
    
    if reverse:
        criteria = CRITERIA_rev
        message = CRITERIA_PROMPT.format(question=query, 
                                       ref_ans=ref_ans,
                                       ans_1=compare_ans,
                                       ans_2=our_ans)
    else:
        criteria = CRITERIA
        message = CRITERIA_PROMPT.format(question=query, 
                                       ref_ans=ref_ans,
                                       ans_1=our_ans,
                                       ans_2=compare_ans)
    return criteria, message

def can_convert_float(scores):
    flag = True
    for s in scores:
        try:
            float(s)
        except:
            flag = False
            break
    return flag

def main(args):
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S', format='%(asctime)s\n%(message)s')
    logger.setLevel(logging.INFO)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s\n%(message)s\n', datefmt='%Y-%m-%d %H:%M:%S')
    os.makedirs('logs/palm2', exist_ok=True)
    
    fh = logging.FileHandler(args.log_path, mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
    vertexai.init(project="my-preject-palm-2")
    model = TextGenerationModel.from_pretrained("text-bison")
    parameters = {
        "temperature": 0.01,  # Temperature controls the degree of randomness in token selection.
        "max_output_tokens": 128,  # Token limit determines the maximum amount of text output.
        "top_p": 1.0,  # Tokens are selected from most probable to least until the sum of their probabilities equals the top_p value.
        "top_k": 50,  # A top_k of 1 means the selected token is the most probable among all tokens.
    }

    questions = []
    with jsonlines.open(args.our_response) as f:
        our_response = []
        for line in f:
            our_response.append(dict(question_id=line['question_id'], text=line['text']))
            questions.append(dict(question_id=line['question_id'], text=line['prompt']))

    with jsonlines.open(args.other_response) as f:
        other_response = []
        for line in f:
            other_response.append(dict(question_id=line['question_id'], text=line['text']))

    with jsonlines.open(args.ref_response) as f:
        ref_response = []
        for line in f:
            ref_response.append(dict(question_id=line['question_id'], text=line['text']))

    random_qi = list(range(len(questions)))
    assistant1_scores = []
    assistant2_scores = []
    os.makedirs(os.path.dirname(args.save_result_path), exist_ok=True)

    cnt = 0
    mode='w'
    # resume
    if os.path.exists(args.save_result_path):
        mode='a'
        with open(args.save_result_path, 'r', encoding="utf-8") as f:
            for line in f:
                cnt += 1
    max_qid = max(random_qi)
    while True:
        if cnt > max_qid:
            break
        i = random_qi[cnt]
        try:
            query = questions[i]
            ref_ans = ref_response[i]
            our_ans = our_response[i]
            other_ans = other_response[i]

            assert query['question_id'] == ref_ans['question_id'] == our_ans['question_id'] == other_ans['question_id']
            decision_dict = dict(qid=query['question_id'])
            criteria, message = build_sending_message(query["text"], ref_ans["text"], our_ans["text"], other_ans["text"], reverse=args.reverse, wr=args.wr)
            messages = [
                    {
                        "author": "USER",
                        "content": criteria + "\n" + message
                    },
                ]

            scoring_response = model.predict(
                                messages[0]['content'],
                                **parameters
                            )
            decision = scoring_response.text
            decision_dict['decision'] = decision
            
            with open(args.save_result_path, 'a', encoding="utf-8") as f:
                json.dump(decision_dict, f, ensure_ascii=False)
                f.write("\n")
            
            logger.info(decision)
            if cnt % 10 == 0 or cnt == len(questions) - 1:
                logger.info(f">>>>>>>>>>\nTotal Questions: {cnt:,} \nMean Score: \nAssistant1 - {np.mean(assistant1_scores):.4f}\nAssistant2 - {np.mean(assistant2_scores):.4f}\n<<<<<<<<<<\n")
        except Exception as e:
            logger.error(e)
            continue
    logger.info(f">>>>>>>>>>\nFINAL RESULTS\nTotal Questions: {cnt:,} \nMean Score: \nAssistant1 - {np.mean(assistant1_scores):.4f}\nAssistant2 - {np.mean(assistant2_scores):.4f}\n<<<<<<<<<<\n")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenAI GPT Scoring")
    parser.add_argument("--our-response", type=str, help="Our response to score")
    parser.add_argument("--other-response", type=str, help="Other response to score")
    parser.add_argument("--ref-response", type=str, help="Other response to score")
    parser.add_argument("--save-result-path", type=str, help="Other response to score")
    parser.add_argument("--log-path", type=str, help="Other response to score")
    parser.add_argument("--reverse", action='store_true', help="Other response to score")
    parser.add_argument("--wr", action='store_true', help="for win rate result")

    args = parser.parse_args()

    main(args)


