# Amazon Product Review Sentiment Analysis using NLP and Machine Learning

## Project Overview

This project focuses on analyzing customer reviews from Amazon products using **Natural Language Processing (NLP)** and **Machine Learning** techniques. The objective is to classify customer reviews into **Positive**, **Neutral**, and **Negative** sentiments based on the review text.

The project demonstrates the complete machine learning workflow, including data preprocessing, feature engineering, model training, evaluation, visualization, and sentiment prediction.

---

## Objectives

* Analyze customer reviews from an Amazon reviews dataset.
* Convert product ratings into sentiment labels.
* Build a machine learning model to classify review sentiments.
* Evaluate the model using standard classification metrics.
* Visualize customer sentiment and review patterns.

---

## Dataset

The project uses an Amazon Product Reviews dataset containing customer reviews and ratings.

### Features Used

* **reviews.text** – Customer review text
* **reviews.rating** – Product rating (1–5)

### Sentiment Labeling

| Rating | Sentiment |
| ------ | --------- |
| 1–2    | Negative  |
| 3      | Neutral   |
| 4–5    | Positive  |

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* WordCloud
* Joblib

---

## Machine Learning Workflow

1. Load the dataset
2. Data cleaning and preprocessing
3. Handle missing values
4. Generate sentiment labels from ratings
5. Split data into training and testing sets
6. Convert text into numerical features using **TF-IDF Vectorization**
7. Train a **Logistic Regression** classifier
8. Evaluate model performance
9. Predict sentiment for new reviews
10. Generate visualizations

---

## Exploratory Data Analysis (EDA)

The project includes several visualizations to understand customer feedback:

* Sentiment Distribution
* Rating Distribution
* Confusion Matrix
* Positive Review Word Cloud
* Most Common Words in Negative Reviews

---

## Model

**Algorithm Used**

* Logistic Regression

**Feature Extraction**

* TF-IDF Vectorizer

---

## Model Evaluation

The model is evaluated using:

* Accuracy Score
* Precision
* Recall
* F1-Score
* Classification Report
* Confusion Matrix

---

## Sample Prediction

**Input Review**

```
This product is amazing and works perfectly.
```

**Predicted Output**

```
Positive
```

---

## Project Structure

```
Amazon-Review-Sentiment-Analysis/
│
├── data.py
├── amazon reviews.csv
├── sentiment_model.pkl
├── requirements.txt
├── README.md
│
├── sentiment_distribution.png
├── rating_distribution.png
├── confusion_matrix.png
├── positive_wordcloud.png
└── negative_keywords.png
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/pragyaguptastats/amazon-review-sentiment-analysis.git
```

Navigate to the project directory:

```bash
cd amazon-review-sentiment-analysis
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python data.py
```

---

## Future Improvements

* Compare multiple machine learning algorithms such as Naïve Bayes, Support Vector Machine, and Random Forest.
* Deploy the model using Streamlit for an interactive web interface.
* Integrate real-time customer review retrieval from online sources.
* Improve performance using transformer-based NLP models such as BERT.
* Develop an interactive dashboard for sentiment analytics.

---

## Skills Demonstrated

* Data Cleaning
* Exploratory Data Analysis (EDA)
* Natural Language Processing (NLP)
* Text Preprocessing
* Feature Engineering
* Machine Learning
* Model Evaluation
* Data Visualization
* Python Programming

---

## Author

**Pragya Gupta**

M.Sc. Statistics | University of Delhi

Interested in Data Analytics, Machine Learning, Statistical Modeling, and Business Intelligence.
