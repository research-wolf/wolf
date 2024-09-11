#!/bin/bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export RAY_DEDUP_LOGS=0
TEMPLATE="template_chatml.jinja"
python -m vllm.entrypoints.openai.api_server \
--model mistralai/Mixtral-8x7B-Instruct-v0.1 \
--chat-template $TEMPLATE \
--tensor-parallel-size 4 \
--max-model-len 6400
