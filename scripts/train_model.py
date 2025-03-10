import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 🔹 1. Data Load karna
df = pd.read_csv("data/clean_news_data.csv")

# 🔹 2. Features (X) aur Labels (y) define karna
X = df["text"]  # News text
y = df["label"] # 0 = Fake, 1 = Real

# 🔹 3. Text ko Vectorize karna (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

# 🔹 4. Train-Test Split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# 🔹 5. Logistic Regression Model Train karna
model = LogisticRegression()
model.fit(X_train, y_train)

# 🔹 6. Model Accuracy Check
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# 🔹 7. Model aur Vectorizer ko Save Karna
with open("models/fake_news_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model training complete & saved in models/ folder")
