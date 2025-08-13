import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

st.set_page_config(page_title="Sentiment Insights", layout="wide")

# Load model and tokenizer
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(".venv/lstm_sentiment_model.h5")
    with open(".venv/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_model()

# Load and clean data
@st.cache_data
def load_data():
    ec = pd.read_csv(".venv/chatgpt_review.csv")
    ec.columns = ec.columns.str.strip().str.lower()
    ec.rename(columns={
        'review': 'review_text',
        'rating': 'rating',
        'verified': 'verified_purchase',
        'platform': 'platform',
        'location': 'location',
        'version': 'version',
        'date': 'date'
    }, inplace=True)
    ec["date"] = pd.to_datetime(ec["date"], errors='coerce')
    ec["review_text"] = ec["review_text"].fillna("").astype(str).str.strip()
    ec["length"] = ec["review_text"].apply(len)
    return ec

ec = load_data()

# Predict sentiment
@st.cache_data
def classify_sentiment(text):
    seq = tokenizer.texts_to_sequences([str(text)])
    padded = pad_sequences(seq, maxlen=100)
    prob = model.predict(padded)[0][0]
    if prob > 0.6:
        return "Positive"
    elif prob < 0.4:
        return "Negative"
    else:
        return "Neutral"

with st.spinner("Classifying reviews..."):
    ec["sentiment"] = ec["review_text"].apply(classify_sentiment)

st.title("🧠 ChatGPT Review Sentiment Insights")

questions = [
    "1️⃣ What is the overall sentiment of user reviews?",
    "2️⃣ How does sentiment vary by rating?",
    "3️⃣ Which keywords or phrases are most associated with each sentiment class?",
    "4️⃣ How has sentiment changed over time?",
    "5️⃣ Do verified users tend to leave more positive or negative reviews?",
    "6️⃣ Are longer reviews more likely to be negative or positive?",
    "7️⃣ Which locations show the most positive or negative sentiment?",
    "8️⃣ Is there a difference in sentiment across platforms (Web vs Mobile)?",
    "9️⃣ Which ChatGPT versions are associated with higher/lower sentiment?",
    "🔟 What are the most common negative feedback themes?"
]

for i, q in enumerate(questions):
    with st.expander(q):
        if i == 0:
            counts = ec["sentiment"].value_counts(normalize=True).round(2)
            st.bar_chart(counts)
            for sentiment in ["Positive", "Neutral", "Negative"]:
                st.write(f"{sentiment}: {counts.get(sentiment, 0.0)}")

        elif i == 1:
            mismatch = ec.groupby("rating")["sentiment"].value_counts().unstack().fillna(0)
            st.write("📊 Sentiment by Rating")
            st.bar_chart(mismatch)

        elif i == 2:
            for sentiment in ["Positive", "Neutral", "Negative"]:
                filtered = ec[ec["sentiment"] == sentiment]["review_text"]
                filtered = filtered[filtered.str.strip().astype(bool)]
                if len(filtered) < 5:
                    st.write(f"Not enough valid reviews for {sentiment} sentiment.")
                    continue
                vectorizer = CountVectorizer(stop_words='english', max_features=30)
                try:
                    X = vectorizer.fit_transform(filtered)
                    keywords = vectorizer.get_feature_names_out()
                    st.write(f"**{sentiment} keywords:** {', '.join(keywords)}")
                except ValueError:
                    st.write(f"Could not extract keywords for {sentiment} sentiment.")

        elif i == 3:
            trend = ec.dropna(subset=["date"]).groupby(ec["date"].dt.to_period("M"))["sentiment"].value_counts().unstack().fillna(0)
            st.line_chart(trend)

        elif i == 4:
            verified = ec.groupby("verified_purchase")["sentiment"].value_counts(normalize=True).unstack().fillna(0).round(2)
            st.write("📊 Sentiment by Verified Purchase")
            st.bar_chart(verified)

        elif i == 5:
            avg_length = ec.groupby("sentiment")["length"].mean().round(1)
            st.write("📏 Average Review Length by Sentiment")
            st.bar_chart(avg_length)

        elif i == 6:
            loc_sent = ec.groupby("location")["sentiment"].value_counts().unstack().fillna(0)
            if "Positive" in loc_sent.columns:
             top_pos = loc_sent["Positive"].sort_values(ascending=False).head(5)
             st.write("🌍 Top Positive Locations")
             st.bar_chart(top_pos)
            else:
               st.warning("No positive reviews found for any location.")

            if "Negative" in loc_sent.columns:
              top_neg = loc_sent["Negative"].sort_values(ascending=False).head(5)
              st.write("🌍 Top Negative Locations")
              st.bar_chart(top_neg)
            else:
               st.warning("No negative reviews found for any location.")

        elif i == 7:
            plat_sent = ec.groupby("platform")["sentiment"].value_counts(normalize=True).unstack().fillna(0).round(2)
            st.write("📱 Sentiment by Platform")
            st.bar_chart(plat_sent)

        elif i == 8:
            version_sent = ec.groupby("version")["sentiment"].value_counts(normalize=True).unstack().fillna(0).round(2)
            st.write("🧬 Sentiment by Version")
            st.bar_chart(version_sent)

        elif i == 9:
            neg_reviews = ec[ec["sentiment"] == "Negative"]["review_text"].dropna()
            neg_reviews = neg_reviews[neg_reviews.str.strip().astype(bool)]
            if len(neg_reviews) < 5:
                st.write("Not enough negative reviews for topic modeling.")
            else:
                vectorizer = CountVectorizer(stop_words='english', max_features=1000)
                X = vectorizer.fit_transform(neg_reviews)
                lda = LatentDirichletAllocation(n_components=5, random_state=42)
                lda.fit(X)
                st.write("🧠 Common Negative Feedback Themes")
                for i, topic in enumerate(lda.components_):
                    top_words = [vectorizer.get_feature_names_out()[j] for j in topic.argsort()[-10:]]
                    st.write(f"**Topic {i+1}:** {', '.join(top_words)}")