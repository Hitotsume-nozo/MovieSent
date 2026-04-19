import pandas as pd
import joblib
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Setup NLTK
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

def clean_text(text):
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    tokens = [ps.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def predict_sentiment(review):
    model_path = os.path.expanduser('~/SentimentAnalysisProject/models/sentiment_model.pkl')
    vectorizer_path = os.path.expanduser('~/SentimentAnalysisProject/models/tfidf_vectorizer.pkl')
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    cleaned = clean_text(review)
    tfidf = vectorizer.transform([cleaned])
    prediction = model.predict(tfidf)[0]
    probability = model.predict_proba(tfidf)[0]
    
    sentiment = 'positive' if prediction == 1 else 'negative'
    confidence = probability[prediction]
    
    return sentiment, confidence

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        sentiment, conf = predict_sentiment(text)
        print(f"Review: {text}")
        print(f"Sentiment: {sentiment} (Confidence: {conf:.2%})")
    else:
        print("Please provide a review text as an argument.")
