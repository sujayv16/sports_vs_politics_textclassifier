# Sports vs. Politics Text Classifier

**Course:** CSL 7640: Natural Language Understanding  
**Assignment:** 1 (Problem 4)  
**Student Name:** Viswanadhapalli Sujay  
**Roll No:** B22CS063

## 📌 Project Overview
This project implements a binary text classifier to distinguish between **Sports** and **Politics** documents. It serves as a comparative study of different Machine Learning algorithms and Feature Extraction techniques to identify the most robust pipeline for text classification.

The system is built using Python and `scikit-learn` and evaluates performance on a subset of the **20 Newsgroups** dataset.

## 📂 Dataset
The dataset is a filtered subset of the 20 Newsgroups corpus, aggregating specific categories into two binary classes:

| Class | Label | Source Categories |
|-------|-------|-------------------|
| **Sports** | `0` | `rec.sport.baseball`, `rec.sport.hockey` |
| **Politics** | `1` | `talk.politics.guns`, `talk.politics.mideast`, `talk.politics.misc` |

* **Total Documents:** 4,618
* **Split:** 80% Training / 20% Testing (Stratified)

## 🛠️ Methodology

### Preprocessing
* **Metadata Removal:** Stripped headers, footers, and quotes to prevent data leakage.
* **Cleaning:** Lowercasing, removal of URLs, email addresses, and special characters.
* **Normalization:** Whitespace collapse.

### Feature Extraction
We compared three vectorization techniques:
1. **Bag of Words (BoW):** Simple word counts.
2. **TF-IDF (Unigram):** Weighted term importance.
3. **TF-IDF (Bigram):** Weighted importance of word pairs.

### Models Evaluated
1. **Multinomial Naive Bayes (MNB)**
2. **Logistic Regression (LR)**
3. **Linear Support Vector Machine (LinearSVC)**

## 🚀 How to Run

### 1. Prerequisites
Ensure you have Python installed. Install the dependencies using:
```bash
pip install -r requirements.txt

## 2. Execution

Run the main script to fetch data, train models, and generate the results:

```bash
python sports_politics_classifier.py --out output
```

### Arguments

* `--out`
  Directory to save the results CSV and trained models
  *(default: output)*

* `--no-save-models`
  Flag to skip saving the trained model files

---

## 3. Output

The script will generate:

**Console Output:**
A summary of accuracy and F1 scores for all models.

**output/results.csv:**
A detailed CSV file containing Precision, Recall, F1, and ROC-AUC for every experiment run.

**output/models/:**
Saved `.joblib` files for the trained pipelines.

---

## 📊 Results Summary

The best performing model was the Linear SVM using TF-IDF (Unigram) features.

| Model       | Feature         | Accuracy | F1 Score |
| ----------- | --------------- | -------- | -------- |
| Linear SVM  | TF-IDF          | 95.99%   | 0.9653   |
| Linear SVM  | TF-IDF (Bigram) | 95.56%   | 0.9619   |
| Naive Bayes | Bag of Words    | 95.45%   | 0.9613   |

For a complete theoretical analysis and error breakdown, please refer to the project report (`report.pdf`).

---

## 📜 notes

This project is submitted for academic evaluation for CSL 7640.
