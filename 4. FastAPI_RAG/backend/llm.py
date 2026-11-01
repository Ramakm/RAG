from threading import Thread
from typing import Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from .config import LLM_MODEL, MAX_NEW_TOKENS

SYSTEM_PROMPT = (
    "You answer questions using ONLY the provided context. "
    "If the answer is not in the context, say you could not find it in the documents. "
    "Be concise and cite sources like [1], [2]."
)


class LocalLLM:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, torch_dtype="auto", device_map="auto")

    def stream(self, question: str, contexts: list[dict]) -> Iterator[str]:
        context = "\n\n".join(f"[{i}] ({c['source']})\n{c['text']}" for i, c in enumerate(contexts, 1))
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(**inputs, streamer=streamer, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

        def run():
            with torch.inference_mode():
                self.model.generate(**kwargs)

        Thread(target=run, daemon=True).start()
        yield from streamer
