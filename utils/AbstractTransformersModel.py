import os
from abc import ABC, abstractmethod

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)


class AbstractTransformersModel(ABC):
    PROMPT = 'Sei un dipendente pubblico che deve riscrivere dei documenti istituzionali italiani per renderli semplici e comprensibili per i cittadini. Ti verrà fornito un documento pubblico e il tuo compito sarà quello di riscriverlo applicando regole di semplificazione senza però modificare il significato del documento originale. Ad esempio potresti rendere le frasi più brevi, eliminare le perifrasi, esplicitare sempre il soggetto, utilizzare parole più semplici, trasformare i verbi passivi in verbi di forma attiva, spostare le frasi parentetiche alla fine del periodo.'

    def __init__(self, hugging_face_model_id: str, torch_dtype=torch.bfloat16, quantization_config=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Using {self.device} for inference')

        self.tokenizer = AutoTokenizer.from_pretrained(hugging_face_model_id, token=os.getenv('HF_TOKEN'))
        self.model = AutoModelForCausalLM.from_pretrained(hugging_face_model_id,
                                                          trust_remote_code=True,
                                                          device_map=self.device,
                                                          torch_dtype=torch_dtype,
                                                          token=os.getenv('HF_TOKEN'),
                                                          attn_implementation='eager',
                                                          quantization_config=quantization_config).eval()
        print('Model loaded')

    @abstractmethod
    def build_prompt(self, _text_to_simplify):
        pass

    @abstractmethod
    def decode(self, _decoded):
        pass

    def predict(self, _texts_to_simplify):
        prompts = [self.build_prompt(_text) for _text in _texts_to_simplify]
        outputs = []
        for prompt in tqdm(prompts):
            x = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            y = self.model.generate(x, max_new_tokens=512, temperature=0.2, top_p=0.1, do_sample=True, eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.eos_token_id)
            decoded = self.tokenizer.batch_decode(y, skip_special_tokens=False)
            decoded = [self.decode(d) for d in decoded]
            print(decoded)
            outputs.extend(decoded)
        return outputs
