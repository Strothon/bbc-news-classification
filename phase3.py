import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Load or preprocess data
df = pd.DataFrame({
    'category': ['tech', 'sport', 'business'],
    'text': [
        "tv future in the hands of viewers with home theatre systems plasma high-definition tvs and digital video recorders moving into the living room",
        "football team wins the championship game with a last minute goal",
        "stock market prices fluctuate due to new economic policies"
    ]
})

def preprocess_text_stub(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words_stub = set(stopwords.words('english'))
    processed_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words_stub]
    return " ".join(processed_tokens)

if 'processed_text' not in df.columns:
    print("Running stub preprocessing for demonstration...")
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('omw-1.4')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    df['processed_text'] = df['text'].apply(preprocess_text_stub)
    print("Stub preprocessing complete.")
    print(df.head())

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words='english')
print("\nStarting TF-IDF vectorization...")
X_tfidf = tfidf_vectorizer.fit_transform(df['processed_text'])
print("TF-IDF vectorization completed.")

print(f"\nShape of the TF-IDF matrix: {X_tfidf.shape}")
feature_names = tfidf_vectorizer.get_feature_names_out()
print(f"Number of features (words/n-grams) found: {len(feature_names)}")
print("Some example features (words/n-grams):")
if len(feature_names) > 40:
    print(list(feature_names[:20]))
    print("...")
    print(list(feature_names[-20:]))
else:
    print(list(feature_names))

y = df['category']
print(f"\nTarget variable 'y' (categories) shape: {y.shape}")
print("Categories:", y.unique())