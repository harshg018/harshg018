import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from typing import List, Dict, Any

from app_lib.models import get_text_sentiment_pipeline, get_clip_model, get_image_captioner
from app_lib.utils import (
    detect_brands_in_text,
    detect_brands_in_image,
    compute_overall_sentiment,
    score_from_label,
)

st.set_page_config(page_title="Brand Multimodal Sentiment Dashboard", layout="wide")

@st.cache_resource(show_spinner=False)
def load_models():
    text_sentiment = get_text_sentiment_pipeline()
    clip, clip_processor = get_clip_model()
    captioner, caption_processor = get_image_captioner()
    return text_sentiment, (clip, clip_processor), (captioner, caption_processor)


def analyze_text_records(texts: List[str], brands: List[str]):
    text_sentiment = st.session_state.models[0]
    records = []
    if not texts:
        return pd.DataFrame(records)

    preds = text_sentiment(texts, truncation=True)
    for t, p in zip(texts, preds):
        label = p[0]['label'] if isinstance(p, list) else p['label']
        score = p[0]['score'] if isinstance(p, list) else p['score']
        score = score_from_label(label, score)
        mentioned = detect_brands_in_text(t, brands)
        for brand in mentioned:
            records.append({
                'modality': 'text',
                'brand': brand,
                'content': t,
                'sentiment_label': label,
                'sentiment_score': score,
            })
    return pd.DataFrame(records)


def analyze_images(images: List[Image.Image], brands: List[str]):
    (clip_model, clip_processor) = st.session_state.models[1]
    (captioner, caption_processor) = st.session_state.models[2]

    from transformers import CLIPProcessor

    records = []
    if not images:
        return pd.DataFrame(records)

    # Zero-shot brand detection with CLIP text prompts
    brand_prompts = [f"logo of {b}" for b in brands] + [f"product of {b}" for b in brands]
    unique_prompts = list(dict.fromkeys(brand_prompts))

    import torch
    clip_model.eval()

    for img in images:
        with torch.no_grad():
            inputs = clip_processor(text=unique_prompts, images=img, return_tensors="pt", padding=True)
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image  # (1, num_text)
            probs = logits_per_image.softmax(dim=1).cpu().numpy().flatten()

        # Pick top-N matching prompts and map back to brands
        top_idx = np.argsort(probs)[-5:][::-1]
        detected = detect_brands_in_image(unique_prompts, top_idx, brands)

        # Image caption, then sentiment of caption via text pipeline to approximate image sentiment
        caption_inputs = caption_processor(images=img, return_tensors="pt")
        caption_ids = captioner.generate(**caption_inputs, max_new_tokens=32)
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        caption_text = tok.decode(caption_ids[0], skip_special_tokens=True)

        # Sentiment on caption
        text_sentiment = st.session_state.models[0]
        sp = text_sentiment(caption_text)[0]
        sentiment_label = sp['label']
        sentiment_score = score_from_label(sentiment_label, sp['score'])

        for brand in detected:
            records.append({
                'modality': 'image',
                'brand': brand,
                'content': caption_text,
                'sentiment_label': sentiment_label,
                'sentiment_score': sentiment_score,
            })

    return pd.DataFrame(records)


def render_dashboard(df: pd.DataFrame):
    if df.empty:
        st.info("Upload content and provide brands to see results.")
        return

    # Aggregate
    agg = df.groupby(['brand']).agg(
        avg_score=('sentiment_score', 'mean'),
        pos_rate=(lambda x: np.mean(np.array(x) > 0)),
        count=('brand', 'count'),
    ).reset_index()

    left, right = st.columns([2, 3])

    with left:
        st.subheader("Brand sentiment summary")
        st.dataframe(agg.sort_values('avg_score', ascending=False), use_container_width=True)
        st.bar_chart(agg.set_index('brand')[['avg_score']])

    with right:
        st.subheader("Sentiment distribution by brand")
        # Bucketize scores
        df['bucket'] = pd.cut(df['sentiment_score'], bins=[-1.01, -0.2, 0.2, 1.01], labels=['negative', 'neutral', 'positive'])
        dist = df.groupby(['brand', 'bucket']).size().reset_index(name='count')
        import altair as alt
        chart = alt.Chart(dist).mark_bar().encode(
            x='brand:N',
            y='count:Q',
            color='bucket:N'
        )
        st.altair_chart(chart, use_container_width=True)


# UI
st.title("Multimodal Brand Sentiment Analysis Dashboard")

with st.sidebar:
    st.header("Inputs")
    brands_input = st.text_input("Brands (comma-separated)", value="Apple, Samsung, Nike")
    brands = [b.strip() for b in brands_input.split(',') if b.strip()]

    text_items = st.text_area("Texts (one per line)", height=150)
    texts = [t.strip() for t in text_items.split('\n') if t.strip()]

    uploaded_images = st.file_uploader("Upload images", type=["png","jpg","jpeg"], accept_multiple_files=True)
    pil_images = []
    if uploaded_images:
        for f in uploaded_images:
            pil_images.append(Image.open(f).convert('RGB'))

if 'models' not in st.session_state:
    with st.spinner('Loading models... first time can take 1-2 minutes'):
        st.session_state.models = load_models()
        # Mark scaffold done

run_btn = st.button("Run Analysis")

if run_btn:
    with st.spinner('Analyzing text...'):
        df_text = analyze_text_records(texts, brands)
    with st.spinner('Analyzing images...'):
        df_images = analyze_images(pil_images, brands)

    df_all = pd.concat([df_text, df_images], ignore_index=True) if not df_text.empty or not df_images.empty else pd.DataFrame([])

    st.subheader("Results")
    if df_all.empty:
        st.warning("No brand mentions detected.")
    else:
        st.dataframe(df_all, use_container_width=True, hide_index=True)
        render_dashboard(df_all)
