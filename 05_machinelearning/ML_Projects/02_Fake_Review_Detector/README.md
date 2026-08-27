# Fake Review Detector

A Machine Learning and Natural Language Processing project that classifies hotel reviews as **Truthful** or **Potentially Deceptive** based on patterns learned from labeled review data.

---

## Project Overview

Online reviews influence customer decisions, but not every review can be trusted. This project uses **Natural Language Processing (NLP)** and **Machine Learning** to analyze review text and predict whether a review is likely to be truthful or deceptive.

The final model takes a review as input and provides:

- Predicted class: **Truthful** or **Potentially Deceptive**
- Model confidence score

---

## Problem Statement

The goal of this project is to build a text classification system that can analyze a hotel review and classify it as:

- **Truthful**
- **Potentially Deceptive**

The model does not prove that a review is fake. It makes predictions based on language patterns learned from the training data.

---

## Dataset

The dataset contains **1,600 hotel reviews** with the following columns:

| Column | Description |
|---|---|
| `deceptive` | Target label indicating whether the review is truthful or deceptive |
| `hotel` | Name of the hotel |
| `polarity` | Sentiment of the review |
| `source` | Source of the review |
| `text` | Actual review text |

### Class Distribution

- Deceptive reviews: **800**
- Truthful reviews: **800**

The dataset is balanced.

---

## Project Workflow

```text
Raw Review Data
        ↓
Data Understanding
        ↓
Data Cleaning
        ↓
Duplicate Analysis
        ↓
Train-Test Split
        ↓
TF-IDF Vectorization
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Model Comparison
        ↓
Final Model Selection
        ↓
Prediction Application

---

## Model Results

Two models were trained and compared:

| Model | Accuracy |
|---|---:|
| Logistic Regression | **87.5%** |
| Multinomial Naive Bayes | ~87% |

Logistic Regression was selected as the final model because it achieved slightly higher accuracy and more balanced performance across both classes.

---

## Application

The project includes a command-line application in `app.py`.

The application:

- Accepts a hotel review as input
- Validates the input
- Converts the review using the saved TF-IDF vectorizer
- Predicts whether the review is likely truthful or potentially deceptive
- Displays the model confidence
- Allows multiple predictions until the user types `exit`

---

## Project Structure

```text
01_Fake_Review_Detector/
│
├── Data/
├── Models/
│   ├── fake_review_model.pkl
│   └── tfidf_vectorizer.pkl
├── Notebook/
├── Outputs/
├── app.py
├── README.md
└── requirements.txt