import torch
import pandas as pd

from .AbstractTransformersModel import AbstractTransformersModel


class Llama3Model(AbstractTransformersModel):
    HUGGING_FACE_MODEL_ID = 'DeepMount00/Llama-3-8b-Ita'

    def __init__(self):
        super().__init__(Llama3Model.HUGGING_FACE_MODEL_ID, torch.bfloat16)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def build_prompt(self, _text_to_simplify):
        messages = [
            {'role': 'system', 'content': AbstractTransformersModel.PROMPT},
            {'role': 'user', 'content': _text_to_simplify},
        ]
        return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    def decode(self, _decoded):
        return _decoded.split('<|start_header_id|>assistant<|end_header_id|>\n\n')[-1].split('<|eot_id|>')[0].strip()