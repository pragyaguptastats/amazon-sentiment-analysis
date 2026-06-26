import pandas as pd

df = pd.read_csv("amazon reviews.csv", low_memory=False)

# Keep required columns
df = df[['reviews.text', 'reviews.rating']]

# Remove missing values
df.dropna(inplace=True)

# Convert ratings to sentiments
def get_sentiment(rating):
    if rating <= 2:
        return "negative"
    elif rating == 3:
        return "neutral"
    else:
        return "positive"

df['sentiment'] = df['reviews.rating'].apply(get_sentiment)

print(df.head())
from sklearn.model_selection import train_test_split

X = df['reviews.text']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('classifier', LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

from sklearn.metrics import classification_report

predictions = model.predict(X_test)

print(classification_report(y_test, predictions))

review = "This product is amazing and works perfectly"

prediction = model.predict([review])

print("Sentiment:", prediction[0])

import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))

df['sentiment'].value_counts().plot(kind='bar')

plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Number of Reviews')

plt.show()

plt.figure(figsize=(6,4))

df['reviews.rating'].value_counts().sort_index().plot(kind='bar')

plt.title('Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Count')

plt.show()

from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions
)

plt.title("Confusion Matrix")
plt.show()

from wordcloud import WordCloud

positive_text = " ".join(
    df[df['sentiment']=='positive']['reviews.text']
)

wordcloud = WordCloud(
    width=800,
    height=400
).generate(positive_text)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Most Common Positive Words")
plt.show()
from sklearn.feature_extraction.text import CountVectorizer

negative_reviews = df[
    df['sentiment']=='negative'
]['reviews.text']

vectorizer = CountVectorizer(
    stop_words='english',
    max_features=10
)

X_neg = vectorizer.fit_transform(negative_reviews)

word_counts = X_neg.sum(axis=0)

words = vectorizer.get_feature_names_out()

counts = word_counts.A1

plt.figure(figsize=(8,4))

plt.bar(words, counts)

plt.title("Most Common Words in Negative Reviews")
plt.xticks(rotation=45)

plt.show()

from sklearn.metrics import accuracy_score, classification_report

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))

print(classification_report(y_test, pred))

print("\n" + "="*60)
print("LIVE SENTIMENT PREDICTION")
print("="*60)

while True:

    review = input("\nEnter a Review (type 'exit' to quit): ")

    if review.lower() == "exit":
        break

    prediction = model.predict([review])

    print("\nPredicted Sentiment:", prediction[0].upper())

print("\nThank you for using the Sentiment Analyzer!")
