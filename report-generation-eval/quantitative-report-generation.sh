EXP_NAME="exp-name"
QUESTION="materials/question/question-file.json"
python src/eval/quantitative.py \
--candidate-path materials/answers/$EXP_NAME/stage2/answers.jsonl \
--reference-path $QUESTION