IMAGE_DIR='your-mimic-cxr-root-path'
EXP_NAME="exp-name"
MODEL_BASE="lmsys/vicuna-7b-v1.5"
CONV_MODE="wolf_vicuna_test"
QUESTION="materials/for-vqa/question-file.json"
export CUDA_VISIBLE_DEVICES="0"
python src/eval/model_vqa.py \
    --conv-mode $CONV_MODE \
    --model-base $MODEL_BASE \
    --lora-enable 1 \
    --temperature 1.0 \
    --max-new-tokens 256 \
    --image-root $IMAGE_DIR \
    --question-file $QUESTION \
    --answers-file materials/answers/$EXP_NAME/test-answers-ours.jsonl \
    --model-path checkpoints/$EXP_NAME \
    --mm-projector checkpoints/$EXP_NAME/mm_projector.bin
