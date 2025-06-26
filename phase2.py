import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')



import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources if not already present
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4') # For WordNetLemmatizer
# nltk.download('punkt')   # For word_tokenize

# Load Data
try:
    df = pd.read_csv('bbc-text.csv') # Replace with your file name/path
    print("Data loaded successfully.")
    print("Original data sample:")
    print(df['text'].head(1)) # Show one original text sample
except FileNotFoundError:
    print("ERROR: 'bbc-text.csv' not found. Please check the file name and path.")
    exit()
# --------------------------------------------------------------------

# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # 1. Text Cleaning: Remove special characters and punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A) # Keep only letters and whitespace

    # 2. Text Normalization: Lowercasing
    text = text.lower()

    # 3. Tokenization: Splitting text into words
    tokens = word_tokenize(text)

    # 4. Remove Stopwords and Apply Lemmatization
    processed_tokens = []
    for token in tokens:
        if token not in stop_words:
            # Lemmatization
            lemma = lemmatizer.lemmatize(token)
            processed_tokens.append(lemma)
            
    return " ".join(processed_tokens)

# Apply the preprocessing function to the 'text' column
# This might take a few moments depending on the dataset size
print("\nStarting text preprocessing...")
df['processed_text'] = df['text'].apply(preprocess_text)
print("Text preprocessing completed.")

print("\nSample of original text vs. processed text:")
print("Original:\n", df['text'].iloc[0][:300]) # Show first 300 chars of an original text
print("\nProcessed:\n", df['processed_text'].iloc[0][:300]) # Show first 300 chars of its processed version

print("\nProcessed DataFrame head:")
print(df[['category', 'processed_text']].head())