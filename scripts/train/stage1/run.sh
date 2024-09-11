#!/bin/bash
EXP_NAME=your-exp-name-here

MODEL_BASE=lmsys/vicuna-7b-v1.5
ROOT_DIR=WoLF
IMAGE_DIR=your-image-root-dir-here
JSON_PATH=materials/train/generated-data.json
OUTPUT_DIR=checkpoints/stage1/${EXP_NAME}
DEEPSPEED=./scripts/deepspeed/zero2.json

deepspeed --include localhost:0,1,2,3 --master_port 25600 src/train/train_mem_lora.py --deepspeed $DEEPSPEED \
--lora_enable True \
--model_name_or_path $MODEL_BASE \
--data_path $JSON_PATH \
--image_folder $IMAGE_DIR \
--output_dir $OUTPUT_DIR \
--vision_tower openai/clip-vit-large-patch14 \
--tune_mm_mlp_adapter True \
--mm_vision_select_layer -2 \
--mm_use_im_start_end \
--bf16 True \
--num_train_epochs 2 \
--per_device_train_batch_size 8 \
--gradient_accumulation_steps 2 \
--per_device_eval_batch_size 4 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 500 \
--save_total_limit 3 \
--learning_rate 2e-5 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 True \
--model_max_length 2048 \
--gradient_checkpointing True \
--lazy_preprocess True \
--report_to none \
