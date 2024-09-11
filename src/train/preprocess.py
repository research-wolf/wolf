from typing import Dict, Optional, Sequence

from numpy import add
from src import conversation as conversation_lib
from _tokenize import _tokenize_fn, _mask_targets_stage2, _mask_targets, _add_speaker_and_signal
import torch
import transformers
import copy

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


def preprocess_vicuna(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        # Skip the first one if it is not from human
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    sep = conv.sep + conv.roles[1] + ": "
    rounds = conversations[0].split(conv.sep2)
    speakers = []
    new_input = []
    for i, rou in enumerate(rounds):
        for j, r in enumerate(rou.split(sep)):
            if r.startswith(roles["human"] + ": ") or (i == j == 0):
                new_input.append(r + sep)
                speakers.append("human")
            else:
                new_input.append(r)
                speakers.append("gpt")

    new_input.pop()
    speakers.pop()
    
    tokenized_list = [
        tokenizer(
            text.strip(), 
            return_tensors="pt", 
            padding="longest", 
            max_length=tokenizer.model_max_length, 
            truncation=True,
            add_special_tokens=False
        ).input_ids[0]
        for text in new_input
    ]
    # input_ids_lens = [tokenized.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list]
    input_ids_lens = []
    for i, tokenized in enumerate(tokenized_list):
        if i == 0:
            input_ids_lens.append(tokenized.ne(tokenizer.pad_token_id).sum().item() + 1)
        else:
            input_ids_lens.append(tokenized.ne(tokenizer.pad_token_id).sum().item())

    input_ids = tokenizer(
        conversations, return_tensors="pt", padding="longest", max_length=tokenizer.model_max_length, truncation=True
    ).input_ids[0]
    target = copy.deepcopy(input_ids)
    k = input_ids_lens[0]
    input_ids_lens = input_ids_lens[1:]
    speakers = speakers[1:]

    # Masking
    target[:k] = IGNORE_INDEX
    for tokenized_len, speaker in zip(input_ids_lens, speakers):
        if speaker == "human":
            target[k+1 : k+tokenized_len] = IGNORE_INDEX
        k += tokenized_len

    return dict(
        input_ids=[input_ids],
        labels=[target],
    )


def preprocess(sources: Sequence[str], tokenizer: transformers.PreTrainedTokenizer, stage2: bool = False, organs: list = None) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if not stage2 and conversation_lib.default_conversation.version == "vicuna":
        return preprocess_vicuna(sources, tokenizer)

    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.get_header}\n\n"
        conversation = _add_speaker_and_signal(header, source, version=conversation_lib.default_conversation.version)
        if stage2:
            new_conversation = ""
            seperator = " Assistant: " if "llava" in conversation_lib.default_conversation.version else "ASSISTANT: "
            new_conversation_starter, organ_and_explain = conversation.split(seperator)
            new_conversation += new_conversation_starter + f" {seperator}\n"
            head, human_conversation = new_conversation_starter.split("\n\n")
            new_sources = [{"from": "human", "value": human_conversation + f" {seperator}\n"}]

            stop_sign = "### " 
            if "vicuna" in conversation_lib.default_conversation.version :
                stop_sign = "</s>"

            organs_and_explains = organ_and_explain.split("\n")
            for state in organs_and_explains:
                if any([(organ in state) for organ in organs]):
                    organ, *explain = state.split(": ") # HACK: if there is a colon in the explain, it will be split
                    explain = ": ".join(explain)
                    new_organ_explain = organ + ": " + explain + "\n"
                    new_conversation += new_organ_explain
                    new_sources.append({"from": organ, "value": new_organ_explain})
            if "llava" in conversation_lib.default_conversation.version:
                conversation = new_conversation + conversation_lib.default_conversation.sep
            elif "vicuna" in conversation_lib.default_conversation.version: 
                conversation = new_conversation + conversation_lib.default_conversation.sep2
            sources = [new_sources]  # packing to list
        conversations.append(conversation)
    sources[0][-1].update({"value": sources[0][-1]["value"] + stop_sign})
    conversations_tokenized = _tokenize_fn(conversations, tokenizer)  # tokenize whole conversations
    input_ids = conversations_tokenized["input_ids"]
    targets = copy.deepcopy(input_ids)
    if stage2:
        skip_mask_len = [
            tokenizer(
                organ_name + ": ",
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
                add_special_tokens=False
            )["input_ids"].ne(tokenizer.pad_token_id).sum().item()
            for organ_name in organs
        ]
        skip_mask_len = [skip_mask_len[i]-1 for i in range(len(skip_mask_len))]
        skip_mask_len[0] += 1 # airspace
        for target, source in zip(targets, sources):
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer, add_special_tokens=False)["input_ids_lens"]
            speakers = [sentence["from"] for sentence in source]
            _mask_targets_stage2(target, tokenized_lens, speakers, skip_mask_len=skip_mask_len, tokenizer=tokenizer)
        attention_mask_point_list = []
        a = tokenized_lens[0] + tokenized_lens[1] # header + human
        for i, tl in enumerate(tokenized_lens[2:], start=2):
            b = sum(tokenized_lens[:i])
            c = sum(tokenized_lens[: i + 1])
            attention_mask_point_list.append(torch.Tensor([a, b, c]))  # a<= <b , c<= <end: attention mask
        attention_mask_point_list = torch.stack(attention_mask_point_list).long()

    else:
        assist_role_name = "Assistant: "
        skip_mask_len = (
            tokenizer(
                assist_role_name,
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
                # add_special_tokens=False, [TODO]
            )["input_ids"]
            .ne(tokenizer.pad_token_id).sum().item()
        )
        for target, source in zip(targets, sources):
            # compute length of each sentence
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
            speakers = [sentence["from"] for sentence in source]
            _mask_targets(target, tokenized_lens, speakers, skip_mask_len=skip_mask_len)

    if not stage2:
        try:
            assert sum(tokenized_lens) == len(target), f"tokenized_lens!=len(target): sum{tokenized_lens}!={len(target)}"
        except AssertionError as e:
            print(e)

    return (
        dict(input_ids=input_ids, labels=targets)
        if not stage2
        else dict(input_ids=input_ids, labels=targets, attention_mask_point_list=attention_mask_point_list)
    )


def preprocess_multimodal(
    sources: Sequence[str],
    multimodal_cfg: dict,
    cur_token_len: int,
) -> Dict:
    is_multimodal = multimodal_cfg["is_multimodal"]
    image_token_len = cur_token_len
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            replace_token = DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
            if multimodal_cfg["use_im_start_end"]:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN

            if isinstance(sentence["value"], int):
                sentence["value"] = str(sentence["value"])
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources
