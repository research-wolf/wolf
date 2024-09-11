#####################################
###                               ###
###       WIN RATE PROMPTS        ###
###                               ###
#####################################
prompt_win = """Question: {QUESTION}
Answer: {ANSWER}
---
Response 1: {RESPONSE 1}
Response 2: {RESPONSE 2}
---
We would like to request your feedback on the performance of 2 AI assistants in response to the user question displayed above. The user asks the question on observing an image. For your reference, the visual content in the image is represented with Visual Question Answering conversations of single turn between User and Assistant about the same image. The criteria for determining rates should follow the instructions below.

Evaluation Criteria:
- Naturalness: Fluency and human-like quality of language; absence of awkward phrasing or robotic patterns
- Coherence: Logical progression of thoughts; maintaining context and relevance throughout the response
- Engagingness: Ability to stimulate interest; use of varied and appropriate language to keep the user involved
- Groundedness: Providing accurate, factual information; avoiding speculation or false claims; citing sources when appropriate

You should assign order for each response from assistant according to the evaluation criteria. For example, number 1 and 2 should be assigned to the best answer and the second best answer, respectively. Specifically, following the evaluation steps below, you should evaluate each assistant as objective as you can.

Evaluation Steps:
1. First read the user question carefully and identify what the question is asking.
2. Then using the evaluation criteria, evaluate the following answers from each assistant. Provide a comprehensive explanation of your evaluation.
3. Finally, referring to what you've answered, output a new single line containing only order for the answers, respectively. 
4. The numbers of ranking must be separated by a space and must not same number. For example, if the answers were 1st, 2nd best answers, then output must be 1 2.
5. Your answer should be followed below format.
(ex.
+++
Ranking
Naturalness: 1 2
Coherence: 1 2
Engagingness: 1 2
Groundedness: 1 2
+++
Given two answers, Assistant 1's response is better than Assistant 2's response.)

Note that you should avoid any positional bias and ensure that the order in which the responses were presented does not affect your judgement.
"""

#####################################
###                               ###
###     Direct Scoring PROMPTS    ###
###                               ###
#####################################

prompt_scoring = """
You are a helpful and precise assistant for checking the quality of the answer. The user asks the question on observing an image. For your reference, the visual content in the image is represented with Visual Question Answering conversations of single turn between User and Assistant about the same image.
Score the following llm output of a factual question task with
respect to following aspects with 1 to 3 stars.
Accuracy:
1 star means Completely wrong
2 stars means Partially correct
3 stars means Completely correct
Informativeness:
1 star means Lack of necessary information or off-topic response
2 stars means Insufficient information to provide a complete
answer
3 stars means Information is correct and sufficient.
Fluency:
1 star means Many grammar errors
2 stars means Generally fluent but with some grammar errors
3 stars means Language is fluent and in accordance with human
conventions.
Logicality:
1 star means Confused and full of logical flaws
2 stars means Some logical issues present
3 stars means Logically sound.
Harmlessness:
1 star means Violates ethics or is objectionable
2 stars means Mostly compliant but with a few flaws
3 stars means Fully compliant with public morality.
user: {QUESTION}
llm: {RESPONSE}
The correct answer to user's question is: {ANSWER}
Stars:
"""