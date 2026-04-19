# IMDB Sentiment Analysis Project

A professional implementation of a sentiment analysis pipeline to classify movie reviews as positive or negative using Natural Language Processing (NLP) and Machine Learning.

## Project Overview

This project implements a highly efficient text classification pipeline. It transforms raw, noisy movie review data into cleaned tokens, vectorizes them using TF-IDF, and employs a Logistic Regression model to predict sentiment with high accuracy.

## Methodology & Pipeline

### 1. Data Preprocessing

To reduce noise and dimensionality, the following steps are applied to every review:

- **HTML Removal**: Stripping `<br />` and other HTML tags using regular expressions.
- **Normalization**: Converting all text to lowercase and removing non-alphabetical characters.
- **Tokenization**: Breaking sentences into individual words.
- **Stop-word Removal**: Filtering out common English words (e.g., "the", "is", "in") that do not contribute to sentiment.
- **Stemming**: Using the Porter Stemmer to reduce words to their root form (e.g., "acting" -> "act").

### 2. Feature Engineering

The project uses **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization.

- **TF**: Measures how frequently a term occurs in a document.
- **IDF**: Reduces the weight of terms that occur very frequently across all documents.
- **Configuration**: The model is limited to the top 5,000 most significant features to optimize speed and prevent overfitting.

### 3. Modeling

A **Logistic Regression** classifier was chosen for this task because:

- It is computationally efficient for high-dimensional sparse data.
- It provides probabilistic outputs (confidence scores).
- It serves as a strong baseline for binary text classification.

## Installation & Running

### Setup

The project uses a Python virtual environment to ensure reproducibility.

```bash
# Navigate to project root
cd ~/SentimentAnalysisProject

# Activate the virtual environment
source venv/bin/activate
```

### Execution Flow

1. **Preprocessing**:

   ```bash
   python src/preprocess.py
   ```

   _Cleans the raw data in ~/Sentiment/ and saves it to data/processed_reviews.csv._

2. **Training**:

   ```bash
   python src/train.py
   ```

   _Trains the model and saves artifacts to models/ and metrics to results/._

3. **Inference (Prediction)**:
   ```bash
   python src/predict.py "This movie was an absolute masterpiece!"
   ```

## Results

The model was evaluated on a hold-out test set of 10,000 reviews.

| Metric              | Value  |
| :------------------ | :----- |
| **Accuracy**        | 88.65% |
| **Precision (Pos)** | 88%    |
| **Recall (Pos)**    | 90%    |
| **F1-Score**        | 0.89   |

### Confusion Matrix

- **True Negatives**: 4,320
- **True Positives**: 4,545
- **False Positives**: 641
- **False Negatives**: 494

## Project Structure

- `data/`: Processed datasets.
- `models/`: Saved `.pkl` files (model and vectorizer).
- `results/`: Evaluation reports.
- `src/`: Source code for preprocessing, training, and prediction.
- `venv/`: Python virtual environment.
