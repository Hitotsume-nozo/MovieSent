import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    # Remove non-alphabet characters and lowercase
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Remove stop-words and stem
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    tokens = [ps.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def main():
    input_path = os.path.expanduser('~/Sentiment/IMDB Dataset.csv')
    output_path = os.path.expanduser('~/SentimentAnalysisProject/data/processed_reviews.csv')
    
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    print("Cleaning and preprocessing text... (this may take a few minutes)")
    df['review'] = df['review'].apply(clean_text)
    
    print(f"Saving processed data to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Preprocessing complete.")

if __name__ == '__main__':
    main()
