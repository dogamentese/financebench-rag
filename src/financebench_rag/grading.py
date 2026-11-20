"""Reading the final answer out of a response and checking numbers against the reference answer."""

import re

import numpy as np


def final_answer(response: str) -> str | None:
    """The text after the last "Final answer:" in a response, or None if there is no such line."""
    for line in reversed(response.splitlines()):
        low = line.lower()
        if "final answer:" in low:
            return line[low.rindex("final answer:") + len("final answer:"):].strip(" *")
    return None


def is_refusal(final: str | None) -> bool:
    """True if the final answer is exactly "not found"."""
    return final is not None and final.strip(" .").lower() == "not found"


def numbers(text: str) -> list[float]:
    """All numbers in a text, after removing commas and dollar signs."""
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", "").replace("$", ""))]


def numeric_match(prediction: str, reference: str) -> bool:
    """True if a number in the prediction is within 3% of a number in the reference (Kobeissi and Langlais, 2026)."""
    return any(np.isclose(p, r, atol=0.03, rtol=0.03) for p in numbers(prediction) for r in numbers(reference))
