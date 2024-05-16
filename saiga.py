import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


class Conversation:
    def __init__(
            self,
            system_prompt="Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."
    ):
        self.message_template = "<s>{role}\n{content}</s>"
        self.response_template = "<s>bot\n"
        self.messages = [{
            "role": "system",
            "content": system_prompt
        }]

    def add_user_message(self, message):
        self.messages.append({
            "role": "user",
            "content": message
        })

    def add_bot_message(self, message):
        self.messages.append({
            "role": "bot",
            "content": message
        })

    def get_prompt(self, tokenizer):
        final_text = ""
        for message in self.messages:
            message_text = self.message_template.format(**message)
            final_text += message_text
        final_text += self.response_template
        return final_text.strip()


class Saiga:
    def __init__(self):
        self.MODEL_NAME = "IlyaGusev/saiga2_7b_lora"
        self.DEFAULT_MESSAGE_TEMPLATE = "<s>{role}\n{content}</s>\n"
        self.DEFAULT_SYSTEM_PROMPT = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME, use_fast=False)
        self.config = PeftConfig.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME,
            load_in_4bit=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        self.model = PeftModel.from_pretrained(
            self.model,
            self.MODEL_NAME,
            torch_dtype=torch.float16
        ).to(torch.device('cuda:0'))
        self.generation_config = GenerationConfig.from_pretrained(self.MODEL_NAME, max_new_tokens=256, early_stopping=True, max_time=15, do_sample=True)
        self.conservations = {}

    def generate(self, prompt):
        data = self.tokenizer(prompt, return_tensors="pt")
        data = {k: v.to(self.model.device) for k, v in data.items()}
        output_ids = self.model.generate(
            **data,
            generation_config=self.generation_config
        )[0]
        output_ids = output_ids[len(data["input_ids"][0]):]
        output = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return output.strip()

    def process_message(self, user_id, message):
        inp = f'''Ты профессиональный психолог, который консультирует людей по переписке, а я твой клиент.
        Тебе пришло следующее сообщение: "{message}".
        Ответь на него и постарайся мне помочь. Это не первое моё сообщения, так что писать ПРИВЕТ или ЗДРАВСТВУЙТЕ НЕ НАДО.'''

        if user_id not in self.conservations.keys():
            print('new_conversation')
            self.conservations[user_id] = Conversation()
            inp = f'''Ты профессиональный психолог, который консультирует людей по переписке, а я твой клиент.
        Тебе пришло следующее сообщение: "{message}".
        Ответь на него и постарайся мне помочь.'''

        self.conservations[user_id].add_user_message(inp)
        prompt = self.conservations[user_id].get_prompt(self.tokenizer)
        output = self.generate(prompt)

        punctuation_marks = ['.', '!', '?']
        max_index = max((output.rfind(mark) for mark in punctuation_marks), default=-1)
        if max_index != -1:
            output = output[:max_index + 1]
        return output

    def support_message(self, message):
        inp = f'''Ты профессиональный психотерапевт, я твой клиент. Мне только что поставили диагноз {message}. 
        Утешь меня и дай совет, что лучше делать дальше.'''
        conversation = Conversation()
        conversation.add_user_message(inp)
        prompt = conversation.get_prompt(self.tokenizer)
        output = self.generate(prompt)

        punctuation_marks = ['.', '!', '?']
        max_index = max((output.rfind(mark) for mark in punctuation_marks), default=-1)
        if max_index != -1:
            output = output[:max_index + 1]
        return output
