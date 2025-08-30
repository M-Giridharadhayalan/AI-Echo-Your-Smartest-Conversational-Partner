import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud
from collections import Counter
import re
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="AI ECHO Review Analyzer",
    page_icon="🤖",
    layout="wide"
)

# Load and preprocess data
@st.cache_data
def load_data():
    bf = pd.read_csv('D:\Data Science\Project-5\AI_ECHO\.venv\Include\chatgpt_review.csv')
    bf['date'] = pd.to_datetime(bf['date'])
    bf['review_length'] = bf['review'].str.len()
    bf['rating'] = pd.to_numeric(bf['rating'], errors='coerce')
    
    # Sentiment analysis
    def get_sentiment(text):
        analysis = TextBlob(str(text))
        if analysis.sentiment.polarity > 0.1:
            return 'positive'
        elif analysis.sentiment.polarity < -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    bf['sentiment'] = bf['review'].apply(get_sentiment)
    return bf

bf = load_data()

# Title
st.title("🤖 AI ECHO Review Analysis")
st.markdown("This chatbot analyzes user reviews and answers specific questions about ChatGPT's performance.")

# Questions and answers
st.header("Analysis Questions & Answers")

# 1. Overall sentiment
with st.expander("1. What is the overall sentiment of user reviews?"):
    sentiment_counts = bf['sentiment'].value_counts()
    total_reviews = len(bf)
    positive_pct = (sentiment_counts.get('positive', 0) / total_reviews) * 100
    negative_pct = (sentiment_counts.get('negative', 0) / total_reviews) * 100
    neutral_pct = (sentiment_counts.get('neutral', 0) / total_reviews) * 100
    
    st.write(f"Based on our analysis of {total_reviews} reviews:")
    st.write(f"- **Positive sentiment:** {positive_pct:.1f}%")
    st.write(f"- **Negative sentiment:** {negative_pct:.1f}%")
    st.write(f"- **Neutral sentiment:** {neutral_pct:.1f}%")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', 
           colors=['#4CAF50', '#F44336', '#FFC107'])
    ax.set_title('Overall Sentiment Distribution')
    st.pyplot(fig)

# 2. Sentiment by rating
with st.expander("2. How does sentiment vary by rating?"):
    rating_sentiment = pd.crosstab(bf['rating'], bf['sentiment'])
    st.write("Sentiment distribution across different ratings:")
    st.dataframe(rating_sentiment)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rating_sentiment.plot(kind='bar', ax=ax)
    ax.set_title('Sentiment Distribution by Rating')
    ax.set_xlabel('Rating')
    ax.set_ylabel('Count')
    ax.legend(title='Sentiment')
    st.pyplot(fig)

# 3. Keywords by sentiment
with st.expander("3. Which keywords or phrases are most associated with each sentiment class?"):
    def get_top_keywords(texts, n=10):
        words = []
        for text in texts:
            words.extend([word.lower() for word in re.findall(r'\b\w+\b', str(text)) 
                        if len(word) > 3 and word not in ['chatgpt', 'review', 'the', 'and', 'for']])
        return Counter(words).most_common(n)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Positive Reviews")
        positive_keywords = get_top_keywords(bf[bf['sentiment'] == 'positive']['review'])
        for word, count in positive_keywords:
            st.write(f"- {word} ({count})")
    
    with col2:
        st.subheader("Negative Reviews")
        negative_keywords = get_top_keywords(bf[bf['sentiment'] == 'negative']['review'])
        for word, count in negative_keywords:
            st.write(f"- {word} ({count})")
    
    with col3:
        st.subheader("Neutral Reviews")
        neutral_keywords = get_top_keywords(bf[bf['sentiment'] == 'neutral']['review'])
        for word, count in neutral_keywords:
            st.write(f"- {word} ({count})")

# 4. Sentiment over time
with st.expander("4. How has sentiment changed over time?"):
    bf['month'] = bf['date'].dt.to_period('M').astype(str)
    time_sentiment = bf.groupby(['month', 'sentiment']).size().unstack(fill_value=0)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    time_sentiment.plot(ax=ax)
    ax.set_title('Sentiment Trends Over Time')
    ax.set_xlabel('Month')
    ax.set_ylabel('Number of Reviews')
    ax.legend(title='Sentiment')
    plt.xticks(rotation=45)
    st.pyplot(fig)

# 5. Verified users sentiment
with st.expander("5. Do verified users tend to leave more positive or negative reviews?"):
    verified_sentiment = pd.crosstab(bf['verified_purchase'], bf['sentiment'])
    st.dataframe(verified_sentiment)
    
    # Calculate percentages
    verified_pos = verified_sentiment.loc['Yes', 'positive'] / verified_sentiment.loc['Yes'].sum() * 100
    non_verified_pos = verified_sentiment.loc['No', 'positive'] / verified_sentiment.loc['No'].sum() * 100
    
    st.write(f"- Verified users: {verified_pos:.1f}% positive reviews")
    st.write(f"- Non-verified users: {non_verified_pos:.1f}% positive reviews")
    
    if verified_pos > non_verified_pos:
        st.write("**Verified users tend to leave more positive reviews.**")
    else:
        st.write("**Non-verified users tend to leave more positive reviews.**")

# 6. Review length vs sentiment
with st.expander("6. Are longer reviews more likely to be negative or positive?"):
    sentiment_length = bf.groupby('sentiment')['review_length'].mean()
    st.write("Average review length by sentiment:")
    st.write(sentiment_length)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bf.boxplot(column='review_length', by='sentiment', ax=ax)
    ax.set_title('Review Length by Sentiment')
    ax.set_ylabel('Review Length (characters)')
    st.pyplot(fig)

# 7. Location-based sentiment
with st.expander("7. Which locations show the most positive or negative sentiment?"):
    # Calculate sentiment score for each location (positive % - negative %)
    location_sentiment = bf.groupby('location')['sentiment'].apply(
        lambda x: (x == 'positive').mean() - (x == 'negative').mean()
    ).sort_values(ascending=False)
    
    st.write("Top 5 locations with most positive sentiment:")
    st.write(location_sentiment.head(5))
    
    st.write("Top 5 locations with most negative sentiment:")
    st.write(location_sentiment.tail(5))

# 8. Platform sentiment
with st.expander("8. Is there a difference in sentiment across platforms?"):
    platform_sentiment = pd.crosstab(bf['platform'], bf['sentiment'])
    st.dataframe(platform_sentiment)
    
    # Calculate positive percentage for each platform
    platform_pos = {}
    for platform in bf['platform'].unique():
        platform_reviews = bf[bf['platform'] == platform]
        platform_pos[platform] = (platform_reviews['sentiment'] == 'positive').mean() * 100
    
    st.write("Positive sentiment by platform:")
    for platform, pos_pct in platform_pos.items():
        st.write(f"- {platform}: {pos_pct:.1f}% positive")

# 9. Version sentiment
with st.expander("9. Which ChatGPT versions are associated with higher/lower sentiment?"):
    version_rating = bf.groupby('version')['rating'].mean().sort_values(ascending=False)
    st.write("Average rating by version:")
    st.write(version_rating)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    version_rating.plot(kind='bar', ax=ax)
    ax.set_title('Average Rating by Version')
    ax.set_xlabel('Version')
    ax.set_ylabel('Average Rating')
    plt.xticks(rotation=45)
    st.pyplot(fig)

# 10. Negative feedback themes
with st.expander("10. What are the most common negative feedback themes?"):
    negative_reviews = bf[bf['sentiment'] == 'negative']
    
    # Common themes
    themes = {
        'Wrong answers': negative_reviews['review'].str.contains('wrong answer', case=False).sum(),
        'Outdated information': negative_reviews['review'].str.contains('outdated', case=False).sum(),
        'Slow response': negative_reviews['review'].str.contains('slow|response time', case=False).sum(),
        'Technical issues': negative_reviews['review'].str.contains('technical|bug|error', case=False).sum(),
        'Context loss': negative_reviews['review'].str.contains('context|conversation', case=False).sum(),
        'Limited functionality': negative_reviews['review'].str.contains('limit|cannot|can\'t', case=False).sum()
    }
    
    themes_df = pd.DataFrame(list(themes.items()), columns=['Theme', 'Count'])
    themes_df = themes_df.sort_values('Count', ascending=False)
    
    st.write("Most common negative feedback themes:")
    st.dataframe(themes_df)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    themes_df.plot(x='Theme', y='Count', kind='bar', ax=ax)
    ax.set_title('Common Negative Feedback Themes')
    ax.set_ylabel('Count')
    plt.xticks(rotation=45)
    st.pyplot(fig)
