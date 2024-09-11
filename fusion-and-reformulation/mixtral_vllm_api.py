import time
import asyncio
import os


def dispatch_mixtral_requests(
    client,
    messages_list,
):
    chat_response = [client.chat.completions.create(model="mistralai/Mixtral-8x7B-Instruct-v0.1", messages=x, max_tokens=512) for x in messages_list]
    # chat_response = [client.chat.completions.create(model="mistralai/Mixtral-8x7B-Instruct-v0.1", messages=x, max_tokens=512) for x in messages_list]

    gen_answers = [chat_response.choices[0].message.content for chat_response in chat_response]
    new_gen_answers = []
    for g in gen_answers:
        new_gen_answers.append(g.split("<|im_end|>")[0])
    return new_gen_answers


def call_mixtral(client, samples, wrap_gen_message, print_result=False):
    message_list = []
    for sample in samples:
        input_msg = wrap_gen_message(sample)
        message_list.append(input_msg)

    try:
        predictions = dispatch_mixtral_requests(client=client, messages_list=message_list)

    except Exception as e:
        print(f"Error in call: {e}")
        time.sleep(3)
        raise e
        return []

    results = []
    for sample, prediction in zip(samples, predictions):
        if prediction:
            # sample["result"] = prediction.choices[0].message.content
            sample["result"] = prediction
            if print_result:
                print(sample["result"])
            results.append(sample)
    return results
