from typing import Dict, Optional

import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    _TRANSFORMERS_AVAILABLE = True
except Exception:  # pragma: no cover
    _TRANSFORMERS_AVAILABLE = False


class TextSentimentAnalyzer:
    def __init__(self, hf_model: Optional[str] = None, device: str = "cpu") -> None:
        self.vader = SentimentIntensityAnalyzer()
        self.hf_model_name = hf_model
        self.device = device
        self.hf_tokenizer = None
        self.hf_model = None
        if _TRANSFORMERS_AVAILABLE and hf_model:
            try:
                self.hf_tokenizer = AutoTokenizer.from_pretrained(hf_model)
                self.hf_model = AutoModelForSequenceClassification.from_pretrained(hf_model)
                if device == "cuda":
                    self.hf_model.to("cuda")
            except Exception:
                self.hf_model = None
                self.hf_tokenizer = None

    def analyze(self, text: str) -> Dict[str, float | str]:
        if not text:
            return {"label": "neutral", "score": 0.0}
        vader_scores = self.vader.polarity_scores(text)
        label = (
            "positive" if vader_scores["compound"] >= 0.05
            else "negative" if vader_scores["compound"] <= -0.05
            else "neutral"
        )
        score = float(vader_scores["compound"])

        if self.hf_model and self.hf_tokenizer:
            try:
                inputs = self.hf_tokenizer(text, return_tensors="pt", truncation=True)
                if self.device == "cuda":
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                with torch.no_grad():
                    logits = self.hf_model(**inputs).logits
                probs = logits.detach().softmax(dim=-1).cpu().numpy().flatten()
                # cardiffnlp twitter-roberta-base-sentiment labels: [negative, neutral, positive]
                idx = int(np.argmax(probs))
                label = ["negative", "neutral", "positive"][idx]
                score = float(probs[idx])
            except Exception:
                pass

        return {"label": label, "score": score}
