import dataclasses
from enum import auto, Enum
import random
from typing import List, Optional, Tuple, Union


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: Union[str, List[str]]
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE # constant; sepatator is only one or two
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    skip_next: bool = False
    @property
    def get_header(self):
        if type(self.system) is list:
            return random.choice(self.system)
        return self.system
    
    def get_prompt(self):
        if type(self.system) is list:
            system_message = random.choice(self.system)
        else:
            system_message = self.system
            
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = system_message + self.sep
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def get_images(self, return_pil=False):
        images = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    import base64
                    from io import BytesIO
                    from PIL import Image
                    msg, image, image_process_mode = msg
                    if image_process_mode == "Pad":
                        def expand2square(pil_img, background_color=(122, 116, 104)):
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
                        image = expand2square(image)
                    elif image_process_mode == "Crop":
                        pass
                    elif image_process_mode == "Resize":
                        image = image.resize((224, 224))
                    else:
                        raise ValueError(f"Invalid image_process_mode: {image_process_mode}")
                    max_hw, min_hw = max(image.size), min(image.size)
                    aspect_ratio = max_hw / min_hw
                    max_len, min_len = 800, 400
                    shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                    longest_edge = int(shortest_edge * aspect_ratio)
                    W, H = image.size
                    if H > W:
                        H, W = longest_edge, shortest_edge
                    else:
                        H, W = shortest_edge, longest_edge
                    image = image.resize((W, H))
                    if return_pil:
                        images.append(image)
                    else:
                        buffered = BytesIO()
                        image.save(buffered, format="JPEG")
                        img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                        images.append(img_b64_str)
        return images

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    import base64
                    from io import BytesIO
                    msg, image, image_process_mode = msg
                    max_hw, min_hw = max(image.size), min(image.size)
                    aspect_ratio = max_hw / min_hw
                    max_len, min_len = 800, 400
                    shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                    longest_edge = int(shortest_edge * aspect_ratio)
                    W, H = image.size
                    if H > W:
                        H, W = longest_edge, shortest_edge
                    else:
                        H, W = shortest_edge, longest_edge
                    image = image.resize((W, H))
                    # image = image.resize((224, 224))
                    buffered = BytesIO()
                    image.save(buffered, format="JPEG")
                    img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                    img_str = f'<img src="data:image/png;base64,{img_b64_str}" alt="user upload image" />'
                    msg = msg.replace('<image>', img_str)
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2)

    def dict(self):
        if len(self.get_images()) > 0:
            return {
                "system": self.system,
                "roles": self.roles,
                "messages": [[x, y[0] if type(y) is tuple else y] for x, y in self.messages],
                "offset": self.offset,
                "sep": self.sep,
                "sep2": self.sep2,
            }
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }

conv_stage2 = Conversation(
    system="You are a physician competent in reading radiology. Look at the given Chest X-ray Image and write a correct report."
    "The assistant gives helpful, detailed, accurate, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)
wolf_conv = Conversation(
    system="You are a large language and vision assistant trained by a group of researchers.\n"
    "You are designed to assist human with a Chest-X-Ray(CXR) research tasks using natural language.\n"
    "You don't have access to the actual report of chest x-ray. You don't have to mention report directly.\n"
    "You are able to understand the visual content that the user provides, and assist the user with a variety of medical and clinical tasks using natural language.\n"
    "Answer the given images and questions.\n",
    roles=("Human", "Assistant"),
    messages=(),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

wolf_conv_train = Conversation(
    system="You are given a chat between a curious human and an artificial intelligence assistant."
    "The assistant gives helpful, detailed, accurate, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(("Human", "Hi!\n\n### Response:"), ("Assistant", "Hi there!  How can I help you today?\n")),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
    version="llava"
)

wolf_conv_vicuna = Conversation(
    system="You are given a chat between a curious user and an artificial intelligence assistant."
    "The assistant gives helpful, detailed, accurate, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
    version="vicuna"
)

wolf_conv_vicuna_test = Conversation(
    system="You are a large language and vision assistant trained by a group of researchers.\n"
    "You are designed to assist human with a Chest-X-Ray(CXR) research tasks using natural language.\n"
    "You don't have access to the actual report of chest x-ray. You don't have to mention report directly.\n"
    "You are able to understand the visual content that the user provides, and assist the user with a variety of medical and clinical tasks using natural language.\n"
    "Answer the given images and questions.\n",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
    version="vicuna",
)

report_gen_instruction_list = [
    "Generate free-text radiology reports for the entered chest X-ray images.",
    "Use the entered chest X-ray images to create corresponding free-text radiology reports.",
    "Based on the entered chest X-ray images, produce free-text radiology reports.",
    "Create free-text radiology reports that correspond to the entered chest X-ray images.",
    "Utilize the entered chest X-ray images to generate corresponding free-text radiology reports.",
    "Generate free-text radiology reports in accordance with the entered chest X-ray images.",
    "Use the entered chest X-ray images to create accurate free-text radiology reports.",
    "Produce free-text radiology reports that match the entered chest X-ray images.",
    "Create free-text radiology reports that are consistent with the entered chest X-ray images.",
    "Utilize the entered chest X-ray images to generate comprehensive free-text radiology reports.",
]
stage2_system_message = "You are given an image and a report for each of the following organs: 'pleural', 'lung', 'heart', 'spine', 'bone', 'mediastinum', and 'airspace'.\nFor each organ, write any observable findings or impressions from the given image and"
wolf_conv_vicuna_stage2 = Conversation(
    system=[stage2_system_message + " " + x for x in report_gen_instruction_list],
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
    version="vicuna",
)

wolf_vicuna_test_for_report_gen = Conversation(
    system=[stage2_system_message + " " + x for x in report_gen_instruction_list],
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="\n",
    version="vicuna",
)

wolf_conv_llava_stage2 = Conversation(
    system="You are a great Chest-X-Ray analyst.\n"
    "You are given an image and a report for each of the following organs: 'pleural', 'lung', 'heart', 'spine', 'bone', 'mediastinum', and 'airspace'.\n"
    "For each organ, write any observable findings or impressions from the given image.\n",
    messages=(("Human", "Hi!\n\n### Response:"), ("Assistant", "Hi there!  How can I help you today?\n")),
    roles=("Human", "Assistant"),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
    version="llava"
)


default_conversation = wolf_conv_train
conv_templates = {
    "wolf_vicuna_test_for_report_gen": wolf_vicuna_test_for_report_gen,
    "wolf_llava": wolf_conv,
    "wolf_vicuna_stage2": wolf_conv_vicuna_stage2,
    "wolf_llava_stage2": wolf_conv_llava_stage2,
    "wolf_vicuna": wolf_conv_vicuna,
    "wolf_vicuna_test": wolf_conv_vicuna_test,
}


if __name__ == "__main__":
    print(default_conversation.get_prompt())
