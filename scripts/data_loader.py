import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# Stopwords download karna
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# 🔹 Text Cleaning Function
def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'\W', ' ', text)  # Special characters remove
    text = re.sub(r'\s+', ' ', text)  # Extra spaces remove
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Stopwords remove
    return text

# 🔹 Load Dataset
def load_and_preprocess_data():
    # Fake aur Real news ka CSV load karna
    df_fake = pd.read_csv("data/Fake.csv")
    df_real = pd.read_csv("data/True.csv")

    # Labels assign karna
    df_fake["label"] = 0  # 0 = Fake News
    df_real["label"] = 1  # 1 = Real News

    # Datasets combine karna
    df = pd.concat([df_fake, df_real], axis=0)

    # Data shuffle karna
    df = df.sample(frac=1).reset_index(drop=True)

    # Title aur Text combine karke clean text banana
    df["text"] = df["title"] + " " + df["text"]
    df["text"] = df["text"].apply(clean_text)

    # Cleaned dataset save karna
    df.to_csv("data/clean_news_data.csv", index=False)
    print("✅ Data cleaned and saved as clean_news_data.csv")

# 🏁 Run when script executes
if __name__ == "__main__":
    load_and_preprocess_data()
