# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from hmac import new
import os
import os.path as osp
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence

import torch

import transformers
from torch.utils.data import Dataset
from wolf_trainer import WolfTrainer

from src import conversation as conversation_lib
from src import WolfLlamaForCausalLM
from src.model.wolf import WolfConfig
from PIL import Image
import torch.nn as nn
import math
from preprocess import preprocess, preprocess_multimodal, preprocess_vicuna
# TODO: import and use code from ../data/dataset.py

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
LOCAL_ORGAN_NAMES = sorted(["pleural", "lung", "heart", "spine", "bone", "mediastinum", "airspace"])
local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def setup_for_distributed(is_master):
    """This function disables printing when not in master process."""
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="lmsys/vicuna-7b-v1.5")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_use_im_start_end: bool = field(default=False)
    model_base: Optional[str] = field(default="lmsys/vicuna-7b-v1.5")
    mm_projector: Optional[str] = field(default=None)


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_token_len: int = 0
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"
    stage2: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    force_fsdp: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    double_quant: bool = field(default=True, metadata={"help": "Compress the quantization statistics through double quantization."})
    quant_type: str = field(default="nf4", metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    # to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=False):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["mm_projector", "vision_tower", "vision_resampler"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
        if getattr(trainer.args, "tune_mm_mlp_adapter", False):
            # Only save Adapter
            keys_to_match = ['mm_projector']
            if getattr(trainer.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
            trainer.model.config.save_pretrained(output_dir)

            current_folder = output_dir.split('/')[-1]
            parent_folder = os.path.dirname(output_dir)
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        
        if trainer.deepspeed:
            torch.cuda.synchronize()
            trainer.save_model(output_dir)
            return
        
        state_dict = trainer.model.state_dict()
        if trainer.args.should_save:
            cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
            del state_dict
            trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = json.load(open(data_path, "r"))

        logging.warning("Formatting inputs...")
        sources = [example["conversations"] for example in list_data_dict]
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

class LazySupervisedDatasetStage2(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, multimodal_cfg: dict):
        super(LazySupervisedDatasetStage2, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = json.load(open(data_path, "r"))
        logging.warning("Formatting inputs...Skip in lazy mode")
        self.root_path = data_path
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.multimodal_cfg = multimodal_cfg
        self.local_names = LOCAL_ORGAN_NAMES
    def __len__(self):
        return len(self.list_data_dict)

    # batch
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        import random
        sources = self.list_data_dict[i]
        local_labels = {name : sources[name] for name in self.local_names}
        local_labels_answer = ""
        # for organ-specific labels
        # [TODO] Masking for organ name ?
        for key, val in local_labels.items():
            val_to_string = ", ".join(val)
            local_labels_answer += f"{key}: {val_to_string}\n"
        sources['conversations'] = [{'from': 'human', 'value': '<image>'},
                                    {'from': 'gpt', 'value': local_labels_answer}]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if "image" in sources[0]:
            dicom_idx = random.randint(0, len(self.list_data_dict[i]["dicom_id"]) - 1)
            if self.list_data_dict[i]["dicom_id"][dicom_idx][-4:] == ".jpg":
                image_file = osp.join(self.list_data_dict[i]["image"], self.list_data_dict[i]["dicom_id"][dicom_idx])
            else:
                image_file = osp.join(self.list_data_dict[i]["image"], self.list_data_dict[i]["dicom_id"][dicom_idx] + ".jpg")
            image_folder = self.multimodal_cfg["image_folder"]
            processor = self.multimodal_cfg["image_processor"]
            try:
                image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")
            except Exception as exn:
                print(exn)
                import random

                return random.choice(self)
            
            if self.multimodal_cfg["image_aspect_ratio"] == "keep":
                max_hw, min_hw = max(image.size), min(image.size)
                aspect_ratio = max_hw / min_hw
                max_len, min_len = 448, 224
                shortest_edge = int(min(max_len / aspect_ratio, min_len))
                image = processor.preprocess(image, return_tensors="pt", do_center_crop=False, size={"shortest_edge": shortest_edge})["pixel_values"][
                    0
                ]
            elif self.multimodal_cfg["image_aspect_ratio"] == "pad":

                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result

                image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            else:
                image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

            # import pdb; pdb.set_trace()
            image_token_len = self.multimodal_cfg["image_token_len"]
            patch_size = int(image.shape[1] // math.sqrt(image_token_len))
            cur_token_len = (image.shape[1] // patch_size) * (image.shape[2] // patch_size)

            try:
                sources = copy.deepcopy([e["conversations"] for e in sources])
            except:
                sources = copy.deepcopy([e["conversations"] for e in sources])

            sources = preprocess_multimodal(sources, self.multimodal_cfg, cur_token_len)
        else:
            try:
                sources = copy.deepcopy([e["conversations"] for e in sources])
            except:
                sources = copy.deepcopy([e["conversatons"] for e in sources])

        data_dict = preprocess(sources, self.tokenizer, stage2=True, organs=self.local_names)  # [input_ids, target_labels_masked]
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], 
                             labels=data_dict["labels"][0],
                             attention_mask_point_list=data_dict["attention_mask_point_list"])

        # image exist in the data
        if "image" in self.list_data_dict[i]:
            data_dict["image"] = image
        elif self.multimodal_cfg["is_multimodal"]:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.multimodal_cfg["image_processor"].crop_size
            data_dict["image"] = torch.zeros(3, crop_size["height"], crop_size["width"])
        # data_dict -> {input_ids[torch.size(token_length)], input_ids[torch.size(token_length)], image[torch.size(3, 224, 224)]}
        return data_dict

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, multimodal_cfg: dict):
        super(LazySupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = json.load(open(data_path, "r"))
        logging.warning("Formatting inputs...Skip in lazy mode")
        self.root_path = data_path
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.multimodal_cfg = multimodal_cfg

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        import random
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if "image" in sources[0]:
            dicom_idx = random.randint(0, len(self.list_data_dict[i]["dicom_id"]) - 1)
            if self.list_data_dict[i]["dicom_id"][dicom_idx][-4:] == ".jpg":
                image_file = osp.join(self.list_data_dict[i]["image"], self.list_data_dict[i]["dicom_id"][dicom_idx])
            else:
                image_file = osp.join(self.list_data_dict[i]["image"], self.list_data_dict[i]["dicom_id"][dicom_idx] + ".jpg")
            image_folder = self.multimodal_cfg["image_folder"]
            processor = self.multimodal_cfg["image_processor"]
            try:
                image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")
            except Exception as exn:
                print(exn)
                import random
                return random.choice(self)

            if self.multimodal_cfg["image_aspect_ratio"] == "keep":
                max_hw, min_hw = max(image.size), min(image.size)
                aspect_ratio = max_hw / min_hw
                max_len, min_len = 448, 224
                shortest_edge = int(min(max_len / aspect_ratio, min_len))
                image = processor.preprocess(image, return_tensors="pt", do_center_crop=False, size={"shortest_edge": shortest_edge})["pixel_values"][
                    0
                ]
            elif self.multimodal_cfg["image_aspect_ratio"] == "pad":

                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result

                image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            else:
                image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

            # import pdb; pdb.set_trace()
            image_token_len = self.multimodal_cfg["image_token_len"]
            patch_size = int(image.shape[1] // math.sqrt(image_token_len))
            cur_token_len = (image.shape[1] // patch_size) * (image.shape[2] // patch_size) 

            try:
                sources = copy.deepcopy([e["conversations"] for e in sources])
            except:
                sources = copy.deepcopy([e["conversations"] for e in sources])

            sources = preprocess_multimodal(sources, self.multimodal_cfg, cur_token_len)
        else:
            try:
                sources = copy.deepcopy([e["conversations"] for e in sources])
            except:
                sources = copy.deepcopy([e["conversatons"] for e in sources])

        data_dict = preprocess(sources, self.tokenizer) # {"input_ids": input_ids, "labels": masked_targets}
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # image exist in the data
        if "image" in self.list_data_dict[i]:
            data_dict["image"] = image
        elif self.multimodal_cfg["is_multimodal"]:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.multimodal_cfg["image_processor"].crop_size
            data_dict["image"] = torch.zeros(3, crop_size["height"], crop_size["width"])

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch["images"] = torch.stack(images)
            else:
                batch["images"] = images

        return batch

@dataclass
class DataCollatorForSupervisedDatasetStage2(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, attention_mask_point_list = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels", "attention_mask_point_list")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        # local_labels = [torch.nn.utils.rnn.pad_sequence(local_label, batch_first=True, padding_value=IGNORE_INDEX) for local_label in local_labels]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            attention_mask_point_list=attention_mask_point_list,
        )

        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch["images"] = torch.stack(images)
            else:
                batch["images"] = images

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    if data_args.stage2:
        dataset_cls = LazySupervisedDatasetStage2 if data_args.lazy_preprocess else SupervisedDatasetStage2
        data_collator = DataCollatorForSupervisedDatasetStage2(tokenizer=tokenizer)
    else:
        dataset_cls = LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset # lazy
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
        
    train_dataset = dataset_cls(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        multimodal_cfg=dict(
            is_multimodal=data_args.is_multimodal,
            image_token_len=data_args.image_token_len,
            image_folder=data_args.image_folder,
            image_aspect_ratio=data_args.image_aspect_ratio,
            use_im_start_end=getattr(data_args, "mm_use_im_start_end", False),
            image_processor=getattr(data_args, "image_processor", None),
        ),
    )
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

def train():
    global local_rank 

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    setup_for_distributed(local_rank < 1)
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    model_name = model_args.model_base if data_args.stage2 else model_args.model_name_or_path
    if model_args.vision_tower is not None:
        model = LlavaLlamaForCausalLM.from_pretrained(model_name, cache_dir=training_args.cache_dir, torch_dtype=compute_dtype).cuda()
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(model_name, cache_dir=training_args.cache_dir, torch_dtype=compute_dtype).cuda()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        training_args.gradient_checkpointing_kwargs = dict(
            use_reentrant=False
        )  # when lora, it should be False, and you don't need to call model.gradient_checkpointing_kwargs <- it has in the Trainer

    # Tokenizer and Conversation Template
    if "vicuna" in model_args.model_name_or_path:
        tokenizer = transformers.LlamaTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
        conversation_lib.default_conversation = conversation_lib.conv_templates["ours_vicuna"]
        if data_args.stage2:
            conversation_lib.default_conversation = conversation_lib.conv_templates["ours_vicuna_stage2"]
            print("STAGE2 Conversation Setting done...")
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
        conversation_lib.default_conversation = conversation_lib.conv_templates["ours_llava"]
        if data_args.stage2:
            conversation_lib.default_conversation = conversation_lib.conv_templates["ours_llava_stage2"]
            print("STAGE2 Conversation Setting done...")

    if "llama" in model_args.model_name_or_path or "vicuna" in model_args.model_name_or_path:
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

    elif "llava" in model_args.model_name_or_path:
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=tokenizer,
                model=model,
            )

    # Low-Rank Adaptation
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model, PeftModel

        if data_args.stage2:
            print(">>>>> STAGE2 <<<<<")
            # Load STAGE1 model: LoRA
            if os.path.exists(os.path.join(model_args.model_name_or_path, "non_lora_trainables.bin")):
                print("Loading additional non-LoRA weights...")
                non_lora_trainables = torch.load(os.path.join(model_args.model_name_or_path, "non_lora_trainables.bin"), map_location="cpu")
                non_lora_trainables = {(k[11:] if k.startswith("base_model.") else k): v for k, v in non_lora_trainables.items()}

                if any(k.startswith("model.model.") for k in non_lora_trainables):
                    non_lora_trainables = {(k[6:] if k.startswith("model.") else k): v for k, v in non_lora_trainables.items()}
                model.load_state_dict(non_lora_trainables, strict=False)

            print("Loading LoRA weights...")
            model = PeftModel.from_pretrained(model, model_args.model_name_or_path)
            print("Merging LoRA weights...")
            print("Model is loaded from STAGE1...!")
            # model = model.merge_and_unload()

        else:
            lora_config = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                target_modules=find_all_linear_names(model),
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                task_type="CAUSAL_LM",
            )
            if training_args.bits == 16:
                if training_args.bf16:
                    model.to(torch.bfloat16)
                if training_args.fp16:
                    model.to(torch.float16)
            rank0_print("Adding LoRA adapters...")
            model = get_peft_model(model, lora_config)
            rank0_print("Added LoRA adapters!")

    # vision tower
    if model_args.vision_tower is not None:
        model_vision_dict = model.get_model().initialize_vision_modules(
            vision_tower=model_args.vision_tower,
            mm_vision_select_layer=model_args.mm_vision_select_layer,
            pretrain_mm_mlp_adapter=model_args.pretrain_mm_mlp_adapter,
        )
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=compute_dtype, device=training_args.device)

        vision_config = model_vision_dict["vision_config"]

        data_args.image_token_len = model_vision_dict["image_token_len"]
        data_args.image_processor = model_vision_dict["image_processor"]
        data_args.is_multimodal = True

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        vision_config.use_im_start_end = training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.initialize_vision_tokenizer(
            mm_use_im_start_end=model_args.mm_use_im_start_end,
            tokenizer=tokenizer,
            device=training_args.device,
            tune_mm_mlp_adapter=model_args.tune_mm_mlp_adapter,
            pretrain_mm_mlp_adapter=model_args.pretrain_mm_mlp_adapter,
        )

    # Load STAGE1 model: mm_projector
    if data_args.stage2:
        if model_args.mm_projector is not None: # stage2
            print(">>>>> STAGE2 <<<<<")
            mm_projector = torch.nn.Linear(vision_config.hidden_size, model.config.hidden_size)
            mm_projector_weights = torch.load(model_args.mm_projector, map_location="cpu")
            mm_projector.load_state_dict({k.split(".")[-1]: v for k, v in mm_projector_weights.items()})

            model.model.mm_projector = mm_projector.cuda().half()
            model.model.vision_tower = [vision_tower]
            print("mm_projector state_dict is loaded...")

    # Model Freeze and Active
    if not data_args.stage2: # stage1
        for name, param in model.named_parameters():
            if 'mm_projector' in name:
                param.requires_grad_(True)
        model.lm_head.requires_grad_(True)

    else : # stage2
        for name, param in model.named_parameters():
            if "mm_projector" in name: # stage2: freeze adapter
                param.requires_grad_(False)
            if "lora_" in name:
                param.requires_grad_(True)
        model.lm_head.requires_grad_(True)

    if model_args.freeze_backbone:  # just for debugging with single gpu
        model.model.requires_grad_(False)

    ### Find require_grad=False parameters ###
    # Filter out parameters with require_grad=False
    non_trainable_params = {name: param for name, param in model.named_parameters() if not param.requires_grad}
    trainable_params = {name: param for name, param in model.named_parameters() if param.requires_grad}
    # Display the non-trainable parameters (or layers)
    print(model)
    print('\n----\n')
    for name in non_trainable_params:
        print("Non-trainable parameter:", name, end=" ### ")
    print('\n----\n')
    for name in trainable_params:
        print("Trainable parameter:", name, end="###")
    # print([name for name, param in model.model.model.vision_tower[0].named_parameters() if not param.requires_grad])

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = LLaVATrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    # model.print_trainable_parameters()
    # Training
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save model
    trainer.save_state()
    model.config.use_cache = True
    if training_args.lora_enable:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
        # state_dict = get_peft_state_maybe_zero_3(
        #     model.named_parameters(), training_args.lora_bias
        # )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
