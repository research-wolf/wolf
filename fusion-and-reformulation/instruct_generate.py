import sys
import time
import json
import argparse
import asyncio
import itertools
from pprint import pprint
from const import SYSTEM_MSG, FEW_SHOT
from attrs import define, field
from mixtral_vllm_api import call_mixtral

conv_to_str = lambda conv: "\n\n".join([("User: " if x["from"] == "human" else "Assistant: ") + x["value"] for x in conv])
conv_to_ehr = lambda conv: "\n\n".join([("EHR: " if conv["from"] == "ehr" else "EHR: None") + conv["value"]])
ehr_template = "{{{eq}, {ea}}}\n\n"
qa_template = "{{{q}, {a}}}\n\n"
'''
SYSTEM_MSG_FOR_EHR = """You are an AI assistant specialized in biomedical topics.
You will be presented with short question and answering conversations related to Electronic Health Records (EHR) called EHRQA between users and assistants.
- Your task is to describe the patient's condition in a single sentence from the dialogue provided.
- Ensure that your sentence does not use expressions like "based on" or "according to," and it must be a standalone sentence that clearly explains the content of the conversation.
- You have to complete a sentence that explains EHR of the patient, but cannot use delimiters like ":".
- You are given a few shot samples and generate dialogs based on the responses of the "assistant" in the few shot samples.
- Create only a pair of User, Assistant conversations that follow the instructions.
"""

Original
"""You are an AI assistant specialized in biomedical topics.

  You are provided with a text description (Figure Caption) of a figure image from a biomedical research paper. In some cases, you may have additional text (Figure Context) that mentions the image. Unfortunately, you don't have access to the actual image.

  Your task is to generate a conversation between a person (User) inquiring about the image and you (Assistant) responding to their questions. The conversation should proceed as though both the User and Assistant are viewing the image, while not referring to the text information (Figure Caption and Figure Context).


  Below are requirements for generating the questions and answers in the conversation:
  - Avoid quoting or referring to specific facts, terms, abbreviations, dates, numbers, or names, as these may reveal the conversation is based on the text information, rather than the image itself. Focus on the visual aspects of the image that can be inferred without the text information.
  - Do not use phrases like "mentioned", "caption", "context" in the conversation. Instead, refer to the information as being "in the image."
  - Ensure that questions are diverse and cover a range of visual aspects of the image.
  - The conversation should include at least 2-3 turns of questions and answers about the visual aspects of the image.
  - Answer responsibly, avoiding overconfidence, and do not provide medical advice or diagnostic information. Encourage the user to consult a healthcare professional for advice.
  """
'''
@define
class PromptGenerator:
    @staticmethod
    def few_shot_messages_gen(query_context, use_inline_mentions=True):
        messages = [
            {"role": "system", "content": SYSTEM_MSG},
        ]
        for ex in FEW_SHOT:
            messages += [
                {"role": "user", "content": PromptGenerator.context_gen(ex, use_inline_mentions)},
                {"role": "assistant", "content": conv_to_ehr(ex["conversations"][0]) + conv_to_str(ex["conversations"][1:])},
            ]
        messages.append({"role": "user", "content": query_context})
        return messages

    @staticmethod
    def context_gen(sample, use_inline_mentions=True):
        ctx = []
        ret = "EHR:\n"
        if not sample["q-ehr"]:
            ret += "None\n\n"
        else:
            for eq, ea in zip(sample["q-ehr"], sample["a-ehr"]):
                ret += ehr_template.format(eq=eq, ea=eq)

        ret += "VQA:\n"
        for q, a in zip(sample["q-vqa"], sample["a-vqa"]):
            ret += qa_template.format(q=q, a=a)
        return ret

    @staticmethod
    def wrap_gen_message(sample, use_inline_mentions=True):
        text = PromptGenerator.context_gen(sample, use_inline_mentions=use_inline_mentions)
        context = PromptGenerator.few_shot_messages_gen(text, use_inline_mentions=use_inline_mentions)
        return context


def main(args):
    with open(args.input_path) as f:
        domain_dict = json.load(f)

    results = []
    for i in range(3):
        print(f"round {i}")
        result_pair_ids = set(result["pair_id"] for result in results)

        batch = []
        counter = 0
        for cycle_idx, samples in enumerate(itertools.zip_longest(*domain_dict.values())):
            if counter >= args.max_size:
                break
            for domain_idx, sample in enumerate(samples):
                if not sample:
                    continue
                counter += 1
                if counter >= args.max_size:
                    break
                if sample["pair_id"] in result_pair_ids:
                    continue
                batch.append(sample)
                if len(batch) >= args.batch_size:
                    async_results = call_mixtral(batch, lambda x: PromptGenerator.wrap_gen_message(x, use_inline_mentions=args.use_inline_mentions))
                    results.extend(async_results)

                    print(f"Result Size: {len(results)}")
                    batch = []
        async_results = call_mixtral(batch, lambda x: PromptGenerator.wrap_gen_message(x, use_inline_mentions=args.use_inline_mentions))
        results.extend(async_results)
    print(f"Result Size: {len(results)}")

    with open(args.output_path, "w") as f:
        for line in results:
            f.write(json.dumps(line) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="data/instruct/llava_med_instruct_fig_captions.json")
    parser.add_argument("--output_path", type=str, default="data/instruct/llava_med_instruct_fig_captions_gen.json")
    parser.add_argument("--use_inline_mentions", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--max_size", type=int, default=60000)
    args = parser.parse_args()
    main(args)
