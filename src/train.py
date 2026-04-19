import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    data_path = os.path.expanduser('~/SentimentAnalysisProject/data/processed_reviews.csv')
    model_path = os.path.expanduser('~/SentimentAnalysisProject/models/sentiment_model.pkl')
    vectorizer_path = os.path.expanduser('~/SentimentAnalysisProject/models/tfidf_vectorizer.pkl')
    results_path = os.path.expanduser('~/SentimentAnalysisProject/results/metrics.txt')

    print(f"Loading processed data from {data_path}...")
    df = pd.read_csv(data_path)

    # Prepare features and labels
    X = df['review']
    y = df['sentiment'].map({'positive': 1, 'negative': 0})

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Vectorizing text using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    # Evaluation
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"Model Training Complete. Accuracy: {acc:.4f}")

    # Save model and vectorizer
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    
    # Save results
    with open(results_path, 'w') as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(conf_matrix))

    print(f"Model and vectorizer saved to models/ directory.")
    print(f"Evaluation metrics saved to {results_path}.")

if __name__ == '__main__':
    main()
