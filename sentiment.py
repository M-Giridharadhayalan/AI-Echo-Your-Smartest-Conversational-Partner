import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="AI ECHO Sentiment Analyzer",
    page_icon="💬",
    layout="wide"
)

# Download NLTK resources
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

@st.cache_resource
def load_models():
    """Load trained models"""
    try:
        model = joblib.load('.venv\Include\sentiment_model.pkl')
        vectorizer = joblib.load('.venv\Include/tfidf_vectorizer.pkl')
        class_names = joblib.load('.venv\Include\class_names.pkl')
        return model, vectorizer, class_names, True
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, False

def preprocess_text(text):
    """Preprocess text for prediction"""
    if not text or text.strip() == '':
        return ''
    
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words and len(word) > 2]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

def predict_sentiment(text, model, vectorizer):
    """Predict sentiment for input text"""
    processed_text = preprocess_text(text)
    if not processed_text:
        return "neutral", [0.33, 0.33, 0.33], "Text too short"
    
    text_vector = vectorizer.transform([processed_text])
    prediction = model.predict(text_vector)[0]
    probabilities = model.predict_proba(text_vector)[0]
    return prediction, probabilities, processed_text

def main():
    st.title("💬 AI ECHO Sentiment Analyzer")
    st.markdown("Analyze the sentiment of ChatGPT reviews using Machine Learning")
    
    # Load models
    model, vectorizer, class_names, models_loaded = load_models()
    
    if not models_loaded:
        st.error("""
        ❌ Model files not found! 
        
        Please run the Jupyter Notebook training first to generate:
        - `sentiment_model.pkl`
        - `tfidf_vectorizer.pkl`
        - `class_names.pkl`
        """)
        return
    
    # Create sidebar
    st.sidebar.header("Settings")
    show_processed = st.sidebar.checkbox("Show processed text", value=False)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Enter Review Text")
        user_input = st.text_area(
            "Type or paste your ChatGPT review:",
            height=150,
            placeholder="Example: 'Great tool for generating content quickly...'",
            help="Enter any text related to ChatGPT usage or experience"
        )
        
        if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
            if user_input.strip():
                with st.spinner("Analyzing sentiment..."):
                    prediction, probabilities, processed_text = predict_sentiment(
                        user_input, model, vectorizer
                    )
                    
                    # Display results
                    st.subheader("📊 Analysis Results")
                    
                    # Sentiment card
                    sentiment_colors = {
                        'positive': '#00cc96',
                        'neutral': '#636efa',
                        'negative': '#ef553b'
                    }
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Predicted Sentiment", prediction.title(), 
                                 delta=None, delta_color="normal")
                    
                    with col2:
                        st.metric("Confidence", f"{max(probabilities)*100:.1f}%")
                    
                    with col3:
                        sentiment_emoji = {
                            'positive': '😊',
                            'neutral': '😐',
                            'negative': '😞'
                        }
                        st.write(f"**Emotion:** {sentiment_emoji[prediction]}")
                    
                    # Probability chart
                    st.subheader("📈 Confidence Distribution")
                    
                    prob_df = pd.DataFrame({
                        'Sentiment': class_names,
                        'Probability': probabilities
                    })
                    
                    fig = px.bar(prob_df, x='Sentiment', y='Probability',
                                color='Sentiment',
                                color_discrete_map=sentiment_colors,
                                text_auto='.1%')
                    fig.update_layout(showlegend=False)
                    fig.update_traces(textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Radar chart for probabilities
                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(
                        r=probabilities,
                        theta=class_names,
                        fill='toself',
                        name='Confidence'
                    ))
                    fig_radar.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                        showlegend=False
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)
                    
                    if show_processed:
                        with st.expander("🔍 View Processed Text"):
                            st.code(processed_text)
                    
            else:
                st.warning("⚠️ Please enter some text to analyze.")
    
if __name__ == "__main__":
    main()