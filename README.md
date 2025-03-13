# News Classification using Naïve Bayes 📊📜✨

---

## Project Overview 🚀📰📌
This project focuses on classifying news articles using the Naïve Bayes algorithm. The model is trained and evaluated under two scenarios:

1. **Without Text Preprocessing:** The raw text data is used directly for training.
2. **With Text Preprocessing:** Stop words are removed using spaCy to analyze the impact of preprocessing on classification performance. 🎯🔍📖

---

## Dataset 📂📑📊
The dataset consists of labeled news articles belonging to multiple categories. Ensure the dataset is cleaned and properly formatted before training the model. 🛠️📝✅

---

## Preprocessing Function ✂️🧼🔡
The preprocessing function removes stop words from the text using spaCy:

```python
import spacy

nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    doc = nlp(text)
    no_stop_words = [token.text for token in doc if not token.is_stop]
    return " ".join(no_stop_words)
```

---

## Implementation Steps 🏗️📊⚙️
1. **Data Loading:** Load the news dataset.
2. **Preprocessing:** Apply the `preprocess` function on the text data (for the preprocessed version).
3. **Feature Extraction:** Convert text data into numerical features using TF-IDF Vectorization.
4. **Model Training:** Train a Naïve Bayes classifier on both raw and preprocessed text data.
5. **Performance Evaluation:** Compare accuracy, precision, recall, and F1-score between both approaches. 📈🧐🔬

---

## Model Training 🏋️‍♂️🎓📡
The model is trained using the `MultinomialNB` classifier from `sklearn`:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load data (Assume 'text' and 'label' columns exist in the dataset)
data = load_news_data()  # Custom function to load the dataset
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Train Model Without Preprocessing
vectorizer = TfidfVectorizer()
nb_model = MultinomialNB()
pipeline = make_pipeline(vectorizer, nb_model)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print("Performance without Preprocessing:\n", classification_report(y_test, y_pred))

# Train Model With Preprocessing
X_train_preprocessed = X_train.apply(preprocess)
X_test_preprocessed = X_test.apply(preprocess)

pipeline.fit(X_train_preprocessed, y_train)
y_pred_preprocessed = pipeline.predict(X_test_preprocessed)
print("Performance with Preprocessing:\n", classification_report(y_test, y_pred_preprocessed))
```

---

## Performance Analysis 📊📉🔎
The impact of preprocessing is evaluated using:
- **Accuracy Score**
- **Precision, Recall, and F1-score**
- **Comparison of classification reports** 🎯📌📈

---

## Conclusion 🧐🔬✅
The analysis highlights whether removing stop words improves or degrades the model's performance. The results guide whether text preprocessing should be included in future text classification models. 📚🤔💡

---

## Dependencies 🔗📦🖥️
- Python 3.x
- `scikit-learn`
- `spaCy`
- `numpy`
- `pandas`

To install the dependencies, run:
```bash
pip install scikit-learn spacy numpy pandas
python -m spacy download en_core_web_sm
```
🎯🛠️🚀

---

## License 📜🛡️⚖️
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

```
MIT License

Copyright (c) 2025
---

