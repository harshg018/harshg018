from typing import Dict, Optional

import numpy as np

try:
    from deepface import DeepFace
    _DEEPFACE = True
except Exception:
    _DEEPFACE = False


EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


class ImageEmotionAnalyzer:
    def __init__(self) -> None:
        self.available = _DEEPFACE

    def analyze(self, image_path_or_url: str) -> Dict[str, float | str]:
        if not image_path_or_url:
            return {"label": "neutral", "score": 0.0}

        if self.available:
            try:
                # DeepFace returns dict with dominant_emotion and scores
                res = DeepFace.analyze(img_path=image_path_or_url, actions=["emotion"], enforce_detection=False)
                if isinstance(res, list) and res:
                    res = res[0]
                label = res.get("dominant_emotion", "neutral")
                scores = res.get("emotion", {})
                score = float(scores.get(label, 0.0))
                return {"label": str(label), "score": score}
            except Exception:
                pass

        # Heuristic fallback: if DeepFace not available, return neutral
        return {"label": "neutral", "score": 0.0}
