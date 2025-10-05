from typing import Tuple, Dict, List
import numpy as np

EMOTIONS = [
    "joy",
    "surprise",
    "neutral",
    "sadness",
    "anger",
    "fear",
    "disgust",
]

EMOTION_PROMPTS = [
    "a photo expressing joy",
    "a photo expressing surprise",
    "a neutral emotion photo",
    "a photo expressing sadness",
    "a photo expressing anger",
    "a photo expressing fear",
    "a photo expressing disgust",
]

POSITIVE_EMOTIONS = {"joy", "surprise"}
NEGATIVE_EMOTIONS = {"sadness", "anger", "fear", "disgust"}


def classify_image_emotion(clip_model, clip_processor, image) -> Tuple[str, float, Dict[str, float]]:
    """Return (top_emotion, top_prob, probs_by_emotion)."""
    import torch
    clip_model.eval()
    with torch.no_grad():
        inputs = clip_processor(text=EMOTION_PROMPTS, images=image, return_tensors="pt", padding=True)
        outputs = clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy().flatten()
    idx = int(np.argmax(probs))
    probs_map = {emo: float(p) for emo, p in zip(EMOTIONS, probs)}
    return EMOTIONS[idx], float(probs[idx]), probs_map


def emotion_to_sentiment_score(emotion_label: str, probability: float) -> float:
    e = emotion_label.lower()
    if e in POSITIVE_EMOTIONS:
        return float(probability)
    if e in NEGATIVE_EMOTIONS:
        return float(-probability)
    return 0.0
