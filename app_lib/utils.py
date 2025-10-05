import re
from typing import List, Iterable
import numpy as np


def normalize_brand(b: str) -> str:
    return re.sub(r"\s+", " ", b.strip().lower())


def detect_brands_in_text(text: str, brands: List[str]) -> List[str]:
    text_norm = f" {re.sub(r'\s+', ' ', text.lower())} "
    detected = []
    for b in brands:
        bn = normalize_brand(b)
        # word-boundary like match to avoid substrings in other words
        if re.search(rf"(^|\W){re.escape(bn)}(\W|$)", text_norm):
            detected.append(b)
    return detected


def detect_brands_in_image(prompts: Iterable[str], top_indices: Iterable[int], brands: List[str]) -> List[str]:
    prompt_list = list(prompts)
    selected = [prompt_list[i] for i in top_indices]
    detected = set()
    for s in selected:
        for b in brands:
            if normalize_brand(b) in s.lower():
                detected.add(b)
    return list(sorted(detected))


def score_from_label(label: str, probability: float) -> float:
    # Map typical labels to signed score [-1, 1]
    l = label.lower()
    if 'positive' in l:
        return float(probability)
    if 'negative' in l:
        return float(-probability)
    # neutral or unknown
    return 0.0


def compute_overall_sentiment(text_score: float, image_score: float) -> float:
    # Simple average; could be weighted by confidence or modality presence
    scores = [s for s in [text_score, image_score] if s is not None]
    if not scores:
        return 0.0
    return float(np.mean(scores))
