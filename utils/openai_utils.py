import os

from tqdm import tqdm
from openai import OpenAI


CLIENT = OpenAI(api_key=os.os.getenv('OPENAI_API_KEY'))
PROMPT = "Sei un dipendente pubblico che deve scrivere dei documenti istituzionali italiani per renderli semplici e comprensibili per i cittadini. Ti verrà fornito un documento pubblico e il tuo compito sarà quello di riscriverlo applicando regole di semplificazione senza però modificare il significato del documento originale. Ad esempio potresti rendere le frasi più brevi, eliminare le perifrasi, esplicitare sempre il soggetto, utilizzare parole più semplicii, trasformare i verbi passivi in verbi di forma attiva, spostare le frasi parentetiche alla fine del periodo."


def build_prompt(_text_to_simplify):
    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": _text_to_simplify},
    ]
    return messages


def predict(_texts_to_simplify, _model='gpt-3.5-turbo-0125'):
    prompts = [build_prompt(_text) for _text in _texts_to_simplify]

    outputs = []
    for prompt in tqdm(prompts):
        response = CLIENT.chat.completions.create(
            model=_model,
            messages=prompt,
            stream=False,
            temperature=0.2,
            top_p=0.1
        )
        outputs.append(response.choices[0].message.content)
    return outputs