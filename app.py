import streamlit as st
import pandas as pd
import requests
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time

# Set up Streamlit
st.title("Real-Time News Sentiment Dashboard")

# Initialize Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to fetch news headlines
def fetch_headlines():
    url = "https://newsapi.org/v2/top-headlines?country=us&apiKey=f4a079b6fa9a4e5491222c9cd8302429"
    response = requests.get(url)
    articles = response.json().get("articles", [])
    return [article["title"] for article in articles]

# Function for sentiment analysis using TextBlob and VADER
def get_sentiment(headline):
    # TextBlob Sentiment
    textblob_sentiment = TextBlob(headline).sentiment.polarity
    # VADER Sentiment
    vader_sentiment = analyzer.polarity_scores(headline)["compound"]
    
    # Combine both sentiment scores (optional)
    avg_sentiment = (textblob_sentiment + vader_sentiment) / 2
    
    if avg_sentiment > 0:
        sentiment = "Positive"
    elif avg_sentiment < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    
    return sentiment, avg_sentiment

# Display live headlines and their sentiment
st.subheader("Live News Headlines and Sentiment")

# Store headlines and sentiments in a DataFrame
if "headlines_df" not in st.session_state:
    st.session_state.headlines_df = pd.DataFrame(columns=["Headline", "Sentiment", "Score"])

while True:
    headlines = fetch_headlines()
    
    # Analyze sentiment of each headline
    for headline in headlines:
        sentiment, score = get_sentiment(headline)
        new_data = pd.DataFrame({"Headline": [headline], "Sentiment": [sentiment], "Score": [score]})
        st.session_state.headlines_df = pd.concat([st.session_state.headlines_df, new_data], ignore_index=True)
    
    # Display headlines with sentiment
    st.write(st.session_state.headlines_df)
    
    # Show sentiment distribution
    sentiment_counts = st.session_state.headlines_df["Sentiment"].value_counts()
    st.bar_chart(sentiment_counts)
    
    time.sleep(60)  # Fetch new headlines every minute
