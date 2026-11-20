"""Answering a question from report pages given as text or as images, with Qwen3.5-4B."""

import time
from functools import lru_cache

import pymupdf
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

from financebench_rag.data import FB, REPO
from financebench_rag.grading import final_answer

IMAGE_DIR = REPO / "data" / "processed" / "images"

PROMPT = """{question}

Work through the question step by step and show any calculation.
If the answer cannot be found, end with the line: Final answer: not found
Otherwise end with one line in the form: Final answer: <value>"""


@lru_cache(maxsize=None)
def open_pdf(doc_name: str) -> pymupdf.Document:
    return pymupdf.open(FB / "pdfs" / f"{doc_name}.pdf")


def page_text(doc_name: str, page_num: int) -> str:
    return open_pdf(doc_name)[page_num].get_text()


def page_image(doc_name: str, page_num: int, dpi: int = 150) -> str:
    """Render a page to a PNG file and return its path."""
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    path = IMAGE_DIR / f"{doc_name}_{page_num}.png"
    open_pdf(doc_name)[page_num].get_pixmap(dpi=dpi).save(path)
    return str(path)


class Reader:
    """The same model reads the pages either as extracted text or as page images."""

    def __init__(self, model_id: str = "Qwen/Qwen3.5-4B", max_new_tokens: int = 1024, dpi: int = 150):
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForMultimodalLM.from_pretrained(model_id, dtype=torch.float16, device_map="auto")
        self.max_new_tokens = max_new_tokens
        self.dpi = dpi

    def ask(self, question: str, pages: list[tuple[str, int]], mode: str) -> dict:
        """Answer from the pages, given as "text" or "image". With no pages, the model answers from memory."""
        content = []
        if pages:
            content.append({"type": "text", "text": "Here are pages from financial reports."})
        for i, (doc_name, page_num) in enumerate(pages, start=1):
            content.append({"type": "text", "text": f"\nPage {i}, from {doc_name}:\n"})
            if mode == "image":
                content.append({"type": "image", "path": page_image(doc_name, page_num, self.dpi)})
            else:
                content.append({"type": "text", "text": page_text(doc_name, page_num)})
        content.append({"type": "text", "text": "\n" + PROMPT.format(question=question)})

        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(self.model.device)
        start = time.time()
        out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        new_tokens = out[0][inputs["input_ids"].shape[-1]:]
        response = self.processor.decode(new_tokens, skip_special_tokens=True).strip()
        return {
            "input_tokens": inputs["input_ids"].shape[-1],
            "response": response,
            "final": final_answer(response),
            "tokens": len(new_tokens),
            "seconds": round(time.time() - start, 1),
        }
