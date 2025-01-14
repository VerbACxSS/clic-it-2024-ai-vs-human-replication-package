import torch
import pandas as pd

from .AbstractTransformersModel import AbstractTransformersModel


class Phi3Model(AbstractTransformersModel):
    HUGGING_FACE_MODEL_ID = 'e-palmisano/Phi3-ITA-mini-4K-instruct'

    def __init__(self):
        super().__init__(Phi3Model.HUGGING_FACE_MODEL_ID, torch.bfloat16)
        self.tokenizer.eos_token = '<|end|>'
        self.tokenizer.eos_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def build_prompt(self, _text_to_simplify):
        messages = [
            {'role': 'user', 'content': AbstractTransformersModel.PROMPT},
            {'role': 'assistant', 'content': 'Quale testo devo semplificare?'},
            {'role': 'user', 'content': _text_to_simplify},
        ]
        return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    def decode(self, _decoded):
        return _decoded.split('<|end|> \n<|assistant|> \n')[-1].split('<|end|>')[0].strip()