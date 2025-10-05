from functools import lru_cache
from transformers import pipeline, CLIPModel, CLIPProcessor, AutoProcessor, VisionEncoderDecoderModel


@lru_cache(maxsize=1)
def get_text_sentiment_pipeline():
    # DistilBERT SST-2
    return pipeline("sentiment-analysis")


@lru_cache(maxsize=1)
def get_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor


@lru_cache(maxsize=1)
def get_image_captioner():
    # ViT-GPT2 captioning
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    processor = AutoProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    return model, processor
