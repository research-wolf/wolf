from typing import Dict, Optional, Sequence
from src import conversation as conversation_lib
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

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer, add_special_tokens: bool=True) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
            add_special_tokens=add_special_tokens,
        )
        for text in strings
    ]  # whole conversation
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets_stage2(target, tokenized_lens, speakers, skip_mask_len: list = None, tokenizer=None):
    # cur_idx = 0
    raw_target = copy.deepcopy(target)
    f = tokenizer.convert_ids_to_tokens

    cur_idx = tokenized_lens[0]  # Header
    # print(f"MASKED TOKENS: {f(raw_target[:cur_idx].tolist())}")
    target[:cur_idx] = IGNORE_INDEX  # ignore token of "System: "
    tokenized_lens = tokenized_lens[1:]  # Masking Header
    k = 0
    for idx, (tokenized_len, speaker) in enumerate(zip(tokenized_lens, speakers)):
        if speaker == "human":
            # cur_idx + 2 <<-- ignore token of "###"
            # print(f"MASKED TOKENS: {f(raw_target[cur_idx:cur_idx+tokenized_len].tolist())}")
            target[cur_idx : cur_idx + tokenized_len] = IGNORE_INDEX
        else:
            if idx == 2: # airxspace
                target[cur_idx+1: cur_idx+skip_mask_len[k]] = IGNORE_INDEX
            # print(f"MASKED TOKENS: {f(raw_target[cur_idx+1 : cur_idx + skip_mask_len[k]].tolist())}")
            else:
                target[cur_idx : cur_idx + skip_mask_len[k]] = IGNORE_INDEX
            k += 1
        cur_idx += tokenized_len
    
def _mask_targets(target, tokenized_lens, speakers, skip_mask_len=0):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]  # Header
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            # cur_idx + 2 <<-- ignore token of "###"
            target[cur_idx + 2 : cur_idx + tokenized_len] = IGNORE_INDEX
        if speaker == "gpt":
            target[cur_idx + 2 : cur_idx + skip_mask_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True, version='llava'):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### " if 'llava' in version else "</s>"
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = "unknown"
        sentence["value"] = BEGIN_SIGNAL + from_str + ": " + sentence["value"] + END_SIGNAL
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation
