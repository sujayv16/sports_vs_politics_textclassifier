"""
sports_politics_classifier.py

Assignment Problem 4: Sports vs Politics Text Classifier.
Author: B22CS063

Description:
This script trains and compares three different Machine Learning models 
(Multinomial Naive Bayes, Logistic Regression, Linear SVC) across three 
feature extraction techniques (Bag of Words, TF-IDF, TF-IDF + Bigrams).

The goal is to classify text documents from the 20 Newsgroups dataset into 
two categories: 'Sports' or 'Politics'.

Usage:
    python sports_politics_classifier.py --out output_folder
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
import joblib

# Scikit-learn imports for data, splitting, and metrics
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)

# ---------------------------------------------------------
# Data Cleaning & Preprocessing
# ---------------------------------------------------------
def basic_clean(text):
    """
    Performs minimal text preprocessing to reduce noise while preserving
    semantic meaning.
    
    Steps:
    1. Convert to lowercase to ensure 'Politics' and 'politics' are treated strictly as the same token.
    2. Remove URLs and email addresses using Regex (these are metadata, not content).
    3. Remove non-alphanumeric characters but keep whitespace.
    4. Collapse multiple spaces into one.
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercasing
    text = text.lower()
    
    # Remove URLs (http/https/www)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    
    # Remove email patterns (user@domain)
    text = re.sub(r'\S+@\S+', ' ', text)
    
    # Remove special chars/punctuation (keep letters, numbers, spaces)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # Collapse extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# ---------------------------------------------------------
# Experiment Execution Logic
# ---------------------------------------------------------
def run_experiments(output_dir="output"):
    """
    Main driver function. 
    1. Fetches data.
    2. Splits into Train/Test.
    3. Iterates through 9 combinations of Vectorizers and Classifiers.
    4. Saves results and models.
    """
    
    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    # -----------------------------------------------------
    # 1. Dataset Collection
    # -----------------------------------------------------
    # We select specific sub-categories to define our binary problem.
    sports_cats = ['rec.sport.baseball', 'rec.sport.hockey']
    politics_cats = ['talk.politics.guns', 'talk.politics.misc', 'talk.politics.mideast']
    categories = sports_cats + politics_cats

    print("Fetching 20 Newsgroups subset...")
    # 'remove' argument strips metadata headers to prevent the model 
    # from cheating by learning from email headers instead of body text.
    data = fetch_20newsgroups(subset='all', categories=categories,
                              remove=('headers', 'footers', 'quotes'),
                              shuffle=True, random_state=42)

    texts = data.data
    targets = data.target
    target_names = data.target_names

    # Convert the multi-class labels (0..4) into binary labels (0 or 1).
    # 0 = Sports, 1 = Politics
    cat_to_binary = {}
    for idx, name in enumerate(target_names):
        if name in sports_cats:
            cat_to_binary[idx] = 0  # Label 0 for Sports
        else:
            cat_to_binary[idx] = 1  # Label 1 for Politics
            
    binary_labels = np.array([cat_to_binary[t] for t in targets])

    print(f"Total documents: {len(texts)}")
    print(f"Distribution: Sports={np.sum(binary_labels==0)}, Politics={np.sum(binary_labels==1)}")

    # -----------------------------------------------------
    # 2. Preprocessing
    # -----------------------------------------------------
    print("Preprocessing text data...")
    cleaned_texts = [basic_clean(t) for t in texts]

    # Stratified Split: Ensures the ratio of Sports/Politics is maintained in both sets.
    # 80% Training, 20% Testing.
    X_train, X_test, y_train, y_test = train_test_split(
        cleaned_texts, binary_labels, test_size=0.2, stratify=binary_labels, random_state=42
    )

    # -----------------------------------------------------
    # 3. Model Definition
    # -----------------------------------------------------
    # We compare three feature representations:
    # 1. Bag of Words (BoW): Simple counts.
    # 2. TF-IDF Unigram: Weighted importance of words.
    # 3. TF-IDF Bigram: Weighted importance of word pairs (e.g., "white house").
    vectorizers = {
        "bow_unigram": CountVectorizer(ngram_range=(1,1), min_df=2),
        "tfidf_unigram": TfidfVectorizer(ngram_range=(1,1), min_df=2),
        "tfidf_unigram_bigram": TfidfVectorizer(ngram_range=(1,2), min_df=2),
    }

    # We compare three classifiers:
    # 1. MultinomialNB: Standard baseline for text.
    # 2. LogisticRegression: Good probabilistic baseline.
    # 3. LinearSVC: SVMs usually work best on high-dimensional text data.
    classifiers = {
        "MultinomialNB": MultinomialNB(),
        "LogisticRegression": LogisticRegression(max_iter=2000, solver='liblinear', random_state=42),
        "LinearSVC": LinearSVC(max_iter=20000, random_state=42, dual='auto'),
    }

    results = []

    # -----------------------------------------------------
    # 4. Training Loop
    # -----------------------------------------------------
    for vec_name, vectorizer in vectorizers.items():
        for clf_name, clf in classifiers.items():
            run_name = f"{vec_name}__{clf_name}"
            print(f"\n--- Running Experiment: {run_name} ---")

            # Create a pipeline: Raw Text -> Vectorizer -> Classifier
            pipe = Pipeline([
                ("vectorizer", vectorizer),
                ("clf", clf)
            ])

            # Train
            pipe.fit(X_train, y_train)

            # Predict
            y_pred = pipe.predict(X_test)

            # Calculate ROC-AUC if possible (requires probability or decision function)
            roc_auc = np.nan
            try:
                if hasattr(pipe.named_steps['clf'], "predict_proba"):
                    y_score = pipe.predict_proba(X_test)[:, 1]
                    roc_auc = roc_auc_score(y_test, y_score)
                elif hasattr(pipe.named_steps['clf'], "decision_function"):
                    y_score = pipe.decision_function(X_test)
                    roc_auc = roc_auc_score(y_test, y_score)
            except Exception:
                pass

            # Compute standard metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)

            print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")
            
            # Save the trained model for future inference
            model_path = os.path.join(model_dir, f"{run_name}.joblib")
            joblib.dump(pipe, model_path)

            # Log results
            results.append({
                "run_id": run_name,
                "vectorizer": vec_name,
                "classifier": clf_name,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "roc_auc": roc_auc
            })

    # -----------------------------------------------------
    # 5. Save Results
    # -----------------------------------------------------
    results_df = pd.DataFrame(results).sort_values(by='f1', ascending=False)
    csv_path = os.path.join(output_dir, "results.csv")
    results_df.to_csv(csv_path, index=False)
    
    print("\n" + "="*50)
    print("Top 3 Models by F1 Score:")
    print(results_df[['run_id', 'f1', 'accuracy']].head(3))
    print(f"\nFull results saved to: {csv_path}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sports vs Politics Classifier (Roll No: B22CS063)")
    parser.add_argument("--out", type=str, default="output", help="Directory to save outputs")
    args = parser.parse_args()

    run_experiments(output_dir=args.out)