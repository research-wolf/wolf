import argparse
from calendar import c
import keyword
from tempfile import tempdir
from tracemalloc import start
from unittest.mock import DEFAULT
from jinja2 import TemplateError
import torch
import os
import json
from tqdm import tqdm
import sys
import shortuuid

# from src.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from src.conversation import conv_templates, SeparatorStyle
from src.model.builder import load_pretrained_model
from src.utils import disable_torch_init
from src.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from transformers import AutoTokenizer, CLIPVisionModel, CLIPImageProcessor, StoppingCriteria, AutoConfig, AutoModelForCausalLM
from PIL import Image
import math
from src import WolfLlamaForCausalLM
import re
import jsonlines

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def patch_config(config):
    patch_dict = {"use_mm_proj": True, "mm_vision_tower": "openai/clip-vit-large-patch14", "mm_hidden_size": 1024}

    cfg = AutoConfig.from_pretrained(config)
    if not hasattr(cfg, "mm_vision_tower"):
        print(f"`mm_vision_tower` not found in `{config}`, applying patch and save to disk.")
        for k, v in patch_dict.items():
            setattr(cfg, k, v)
        cfg.save_pretrained(config)


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def eval_model(args):
    set_seed(args.seed)
    # Model
    disable_torch_init()
    model_base = os.path.expanduser(args.model_base) # expand home directory
    model_path = os.path.expanduser(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
    if model_path is not None:
        kwargs={}
        if "vicuna" in model_base.lower() and model_path is not None:
            print("Loading Vicuna from base model...")
            model = WolfLlamaForCausalLM.from_pretrained(model_base, torch_dtype=torch.float16, **kwargs).cuda()
            tokenizer.add_special_tokens(
                {
                    "eos_token": DEFAULT_EOS_TOKEN,
                    "bos_token": DEFAULT_BOS_TOKEN,
                    "unk_token": DEFAULT_UNK_TOKEN,
                }
            )
            tokenizer.add_tokens(
                [
                    DEFAULT_IM_END_TOKEN,
                    DEFAULT_IMAGE_PATCH_TOKEN, 
                    DEFAULT_IM_START_TOKEN
                ]
            )
            model.resize_token_embeddings(len(tokenizer))
            tokenizer.pad_token = tokenizer.unk_token
        else:
            print("Loading Wolf from base model...")
            model = WolfLlamaForCausalLM.from_pretrained(model_base, torch_dtype=torch.float16, **kwargs).cuda()

    else:
        if "vicuna" in model_path.lower() and model_path is not None:
            kwargs={}
            print("Loading Vicuna from base model...")
            model = WolfLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, torch_dtype=torch.float16, **kwargs).cuda()
        else:
            kwargs={}
            print("Loading Wolf from base model...")
            model = WolfLlamaForCausalLM.from_pretrained(model_base, torch_dtype=torch.float16).cuda()

    image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=torch.float16)
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    # vision tower
    vision_tower = model.model.vision_tower[0]
    vision_tower.to(device="cuda", dtype=torch.float16)
    vision_config = vision_tower.config
    vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
    vision_config.use_im_start_end = mm_use_im_start_end
    if mm_use_im_start_end:
        vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
    image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2

    # mm_projector
    if args.mm_projector is not None:
        mm_projector = torch.nn.Linear(vision_config.hidden_size, model.config.hidden_size)
        mm_projector_weights = torch.load(args.mm_projector, map_location="cpu")
        mm_projector.load_state_dict({k.split(".")[-1]: v for k, v in mm_projector_weights.items()})

        model.model.mm_projector = mm_projector.cuda().half()
        model.model.vision_tower = [vision_tower]
        print("mm_projector state_dict is loaded...")

    # lora ; peft
    if args.lora_enable:
        from peft import PeftModel
        if os.path.exists(os.path.join(model_path, "non_lora_trainables.bin")):
            print("Loading additional non-LoRA weights...")
            non_lora_trainables = torch.load(os.path.join(model_path, "non_lora_trainables.bin"), map_location="cpu")
            non_lora_trainables = {(k[11:] if k.startswith("base_model.") else k): v for k, v in non_lora_trainables.items()}

            if any(k.startswith("model.model.") for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith("model.") else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

        print("Loading LoRA weights...")
        model = PeftModel.from_pretrained(model, model_path)
        print("Merging LoRA weights...")
        model = model.merge_and_unload()
        print("Model is loaded...")

    else:
        # in case of using a pretrained model with only a MLP projector weights
        model = WolfLlamaForCausalLM.from_pretrained(model_base, torch_dtype=torch.float16).cuda()
        vision_tower = CLIPVisionModel.from_pretrained(args.vision_tower, torch_dtype=torch.float16).cuda()
        image_processor = CLIPImageProcessor.from_pretrained(args.vision_tower, torch_dtype=torch.float16)

        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

        vision_config = vision_tower.config
        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
        vision_config.use_im_start_end = mm_use_im_start_end
        if mm_use_im_start_end:
            vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])

        image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2

        mm_projector = torch.nn.Linear(vision_config.hidden_size, model.config.hidden_size)
        mm_projector_weights = torch.load(args.mm_projector, map_location="cpu")
        mm_projector.load_state_dict({k.split(".")[-1]: v for k, v in mm_projector_weights.items()})

        model.model.mm_projector = mm_projector.cuda().half()
        model.model.vision_tower = [vision_tower]

    # answering process
    if 'ours' in args.conv_mode :
        cdatas = []
        canswer_file_path = "materials/answers/your-true-answers-for-questions.jsonl"
        with jsonlines.open(canswer_file_path, "r") as f:
            for line in f.iter():
                cdatas.append(line)

    else: 
        cdatas = None

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    # resume
    mode='w'
    start_idx=0
    if os.path.exists(answers_file):
        with jsonlines.open(answers_file, "r") as f:
            answered = [line["question_id"] for line in f.iter()]
        start_idx = answered[-1] + 1
        mode='a'
        print(f"Resuming from {start_idx}th question...")

    ans_file = open(answers_file, mode)
    for qid, line in enumerate(tqdm(questions[start_idx:]), start=start_idx):
        patient_id, study_id, dicom_id = line["image"].split("/")[-3:]
        idx = line["question_id"]
        assert idx == qid
        image_file = os.path.join(args.image_root, line['image'])
        qs = line["text"]
        cur_prompt = qs
        if mm_use_im_start_end:
            if DEFAULT_IMAGE_TOKEN in qs:
                qs = qs.replace(DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN)
            else:
                qs = qs + "\n" + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN
        else:
            if DEFAULT_IMAGE_TOKEN in qs:
                qs = qs.replace(DEFAULT_IMAGE_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN * image_token_len)
            else:
                qs = qs + "\n" + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len

        response_sep = "ASSISTANT: " if 'vicuna' in model_base else "### Response: "
        qs = qs + response_sep
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        prompt = conv.get_prompt()

        # prompt to input_ids
        inputs = tokenizer([prompt])
        input_ids = torch.as_tensor(inputs["input_ids"]).cuda()
        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], image_processor, model.config)[0]

        # new stopping implementation
        class KeywordsStoppingCriteria(StoppingCriteria):
            def __init__(self, keywords, tokenizer, input_ids):
                self.keywords = keywords
                self.tokenizer = tokenizer
                self.start_len = None
                self.input_ids = input_ids

            def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                if self.start_len is None:
                    self.start_len = self.input_ids.shape[1]
                else:
                    outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len :], skip_special_tokens=True)[0]
                    for keyword in self.keywords:
                        if keyword in outputs:
                            return True
                return False

        keywords = ["</s>"] if 'vicuna' in model_base else ["###"]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        model_response = ""
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                temperature=args.temperature,
                images=image_tensor.unsqueeze(0).cuda(),
                do_sample=True,
                max_new_tokens=args.max_new_tokens,
                stopping_criteria=[stopping_criteria],
                # no_repeat_ngram_size=5,
            )

        raw_outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        model_response = raw_outputs.split(response_sep)[-1].strip()

        while True:
            cur_len = len(outputs)
            outputs = outputs.strip()
            for pattern in ['###', 'Assistant:', 'Response:']:
                if outputs.startswith(pattern):
                    outputs = outputs[len(pattern):].strip()
            if len(outputs) == cur_len:
                break

        ans_id = shortuuid.uuid()
        ans_file.write(
            json.dumps(
                {
                    "question_id": idx,
                    "text": model_response,
                    "prompt": cur_prompt,
                    "answer_id": ans_id,
                    "model_id": model_path,
                    "metadata": {"patient_id": patient_id, "study_id": study_id, "dicom_id": dicom_id},
                }
            )
            + "\n"
        )
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", type=str, default="")
    parser.add_argument("--lora-enable", type=int, default=0)
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default="facebook/opt-350m")
    parser.add_argument("--mm-projector", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="simple_legacy")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=17)
    
    args = parser.parse_args()

    eval_model(args)
