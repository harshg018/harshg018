import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional

from app_lib.models import get_text_sentiment_pipeline, get_clip_model, get_image_captioner
from app_lib.utils import (
    detect_brands_in_text,
    detect_brands_in_image,
    compute_overall_sentiment,
    score_from_label,
)
from app_lib.ingest import read_events_csv, read_inbox_csvs
from app_lib.analytics import aggregate_sentiment, bucketize_sentiment, timeseries_sentiment, correlate_with_metrics
from app_lib.emotions import classify_image_emotion, emotion_to_sentiment_score
from app_lib.recommend import generate_recommendations
import pydeck as pdk

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

        # Emotion classification (CLIP zero-shot)
        emo_label, emo_prob, _ = classify_image_emotion(clip_model, clip_processor, img)
        emo_score = emotion_to_sentiment_score(emo_label, emo_prob)
        # Combine caption sentiment and emotion score (simple average)
        sentiment_score = float(np.mean([sentiment_score, emo_score]))

        for brand in detected:
            records.append({
                'modality': 'image',
                'brand': brand,
                'content': caption_text,
                'sentiment_label': sentiment_label,
                'sentiment_score': sentiment_score,
                'emotion_label': emo_label,
                'emotion_confidence': emo_prob,
            })

    return pd.DataFrame(records)


def analyze_events(df_events: pd.DataFrame, brands: List[str]) -> pd.DataFrame:
    """Analyze a normalized events dataframe (see app_lib.ingest) and return scored rows.
    Supports text and optional image_path per row.
    """
    if df_events is None or df_events.empty:
        return pd.DataFrame([])

    text_sentiment = st.session_state.models[0]
    (clip_model, clip_processor) = st.session_state.models[1]
    (captioner, caption_processor) = st.session_state.models[2]

    # Build prompts once
    brand_prompts = [f"logo of {b}" for b in brands] + [f"product of {b}" for b in brands]
    unique_prompts = list(dict.fromkeys(brand_prompts))

    rows: List[Dict[str, Any]] = []
    for _, r in df_events.iterrows():
        text_val: str = str(r.get('text') or '')
        explicit_brand: Optional[str] = r.get('brand') if pd.notna(r.get('brand')) else None
        mentioned = [explicit_brand] if explicit_brand else detect_brands_in_text(text_val, brands)

        # Text scoring
        if text_val.strip():
            sp = text_sentiment(text_val)[0]
            s_label = sp['label']
            s_score = score_from_label(s_label, sp['score'])
            for b in mentioned:
                rows.append({
                    'modality': 'text',
                    'brand': b,
                    'content': text_val,
                    'sentiment_label': s_label,
                    'sentiment_score': s_score,
                    'timestamp': r.get('timestamp'),
                    'lat': r.get('lat'),
                    'lon': r.get('lon'),
                    'source': r.get('source'),
                })

        # Image scoring
        img_path = r.get('image_path')
        if pd.notna(img_path) and str(img_path).strip():
            try:
                img = Image.open(str(img_path)).convert('RGB')
            except Exception:
                img = None
            if img is not None:
                import torch
                clip_model.eval()
                with torch.no_grad():
                    inputs = clip_processor(text=unique_prompts, images=img, return_tensors="pt", padding=True)
                    outputs = clip_model(**inputs)
                    probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy().flatten()
                top_idx = np.argsort(probs)[-5:][::-1]
                detected_img_brands = detect_brands_in_image(unique_prompts, top_idx, brands)

                caption_inputs = caption_processor(images=img, return_tensors="pt")
                caption_ids = captioner.generate(**caption_inputs, max_new_tokens=32)
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
                caption_text = tok.decode(caption_ids[0], skip_special_tokens=True)

                sp = text_sentiment(caption_text)[0]
                s_label = sp['label']
                s_score = score_from_label(s_label, sp['score'])
                emo_label, emo_prob, _ = classify_image_emotion(clip_model, clip_processor, img)
                emo_score = emotion_to_sentiment_score(emo_label, emo_prob)
                final_score = float(np.mean([s_score, emo_score]))

                for b in (detected_img_brands or mentioned):
                    rows.append({
                        'modality': 'image',
                        'brand': b,
                        'content': caption_text,
                        'sentiment_label': s_label,
                        'sentiment_score': final_score,
                        'emotion_label': emo_label,
                        'emotion_confidence': emo_prob,
                        'timestamp': r.get('timestamp'),
                        'lat': r.get('lat'),
                        'lon': r.get('lon'),
                        'source': r.get('source'),
                    })

    return pd.DataFrame(rows)


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


def render_map(df: pd.DataFrame):
    if df is None or df.empty or not set(['lat','lon']).issubset(df.columns):
        return
    m = df.copy()
    m = m.dropna(subset=['lat','lon'])
    if m.empty:
        return
    # Ensure numeric
    m['lat'] = pd.to_numeric(m['lat'], errors='coerce')
    m['lon'] = pd.to_numeric(m['lon'], errors='coerce')
    m = m.dropna(subset=['lat','lon'])
    if m.empty:
        return
    m['weight'] = m['sentiment_score'].abs().clip(0, 1)
    # Color mapping: red for negative, green for positive
    def to_rgb(score: float):
        return [int(255 * max(-score, 0)), int(255 * max(score, 0)), 60]
    colors = m['sentiment_score'].apply(to_rgb)
    m[['r','g','b']] = pd.DataFrame(colors.tolist(), index=m.index)

    midpoint = (m['lat'].mean(), m['lon'].mean())
    layer = pdk.Layer(
        'ScatterplotLayer',
        data=m,
        get_position='[lon, lat]',
        get_radius=20000,
        get_weight='weight',
        get_fill_color='[r, g, b, 180]'
    )
    view_state = pdk.ViewState(latitude=float(midpoint[0]), longitude=float(midpoint[1]), zoom=2)
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{brand}: {sentiment_score}"}))


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

    st.divider()
    st.caption("Realtime ingestion")
    inbox_dir = st.text_input("Inbox folder (CSV files)", value="/workspace/inbox")
    uploaded_events = st.file_uploader("Or upload events CSV", type=["csv"], accept_multiple_files=False)

    st.divider()
    st.caption("Correlation metrics (CSV with timestamp and metrics columns)")
    metrics_file = st.file_uploader("Upload metrics CSV", type=["csv"], accept_multiple_files=False, key="metrics")

if 'models' not in st.session_state:
    with st.spinner('Loading models... first time can take 1-2 minutes'):
        st.session_state.models = load_models()
        # Mark scaffold done

run_btn = st.button("Run Analysis")

if run_btn:
    # Base manual inputs
    with st.spinner('Analyzing text...'):
        df_text = analyze_text_records(texts, brands)
    with st.spinner('Analyzing images...'):
        df_images = analyze_images(pil_images, brands)

    frames = [df_text, df_images]

    # Ingested events
    df_events = pd.DataFrame([])
    if uploaded_events is not None:
        try:
            df_events = read_events_csv(uploaded_events)
        except Exception as e:
            st.error(f"Failed to read uploaded events CSV: {e}")
    elif inbox_dir:
        try:
            df_events = read_inbox_csvs(inbox_dir)
        except Exception as e:
            st.error(f"Failed to read inbox: {e}")

    if not df_events.empty:
        with st.spinner('Analyzing ingested events...'):
            df_stream = analyze_events(df_events, brands)
            frames.append(df_stream)

    df_all = pd.concat([f for f in frames if f is not None and not f.empty], ignore_index=True) if any((f is not None and not f.empty) for f in frames) else pd.DataFrame([])

    st.subheader("Results")
    if df_all.empty:
        st.warning("No brand mentions detected.")
    else:
        st.dataframe(df_all, use_container_width=True, hide_index=True)
        render_dashboard(df_all)

        # Geospatial heatmap
        st.subheader("Geospatial sentiment heatmap")
        render_map(df_all)

        # Time series and correlations
        st.subheader("Trends and correlations")
        sent_ts = timeseries_sentiment(df_all, freq="D")
        if not sent_ts.empty:
            st.line_chart(sent_ts.pivot(index='timestamp', columns='brand', values='avg_score'))
        if metrics_file is not None:
            try:
                m = pd.read_csv(metrics_file)
                merged, corr = correlate_with_metrics(sent_ts, m)
                if not corr.empty:
                    st.write("Top correlations (avg_score vs metric):")
                    st.dataframe(corr, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Failed to read metrics CSV: {e}")

        # Recommendations
        st.subheader("Recommendations")
        agg = aggregate_sentiment(df_all)
        recs = generate_recommendations(sent_ts, agg)
        if not recs.empty:
            st.dataframe(recs, use_container_width=True, hide_index=True)
