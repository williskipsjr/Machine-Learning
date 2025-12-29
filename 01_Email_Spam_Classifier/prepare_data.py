import os
import re
import string
import numpy as np
import csv
from collections import Counter
from sklearn.model_selection import train_test_split

# --- CONFIGURABLE HYPERPARAMETERS ---
REMOVE_HEADERS = True
TO_LOWER = True
REMOVE_PUNCT = True
REPLACE_URLS = True
REPLACE_NUMBERS = True
USE_STEMMING = False  # Set to True to use stemming
BINARY_FEATURES = True  # True: presence/absence, False: word counts

# --- Optional: Stemming ---
if USE_STEMMING:
    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()

def preprocess(text):
    if REMOVE_HEADERS:
        # Remove headers (everything before first blank line)
        parts = text.split('\n\n', 1)
        text = parts[1] if len(parts) > 1 else text
    if TO_LOWER:
        text = text.lower()
    if REPLACE_URLS:
        text = re.sub(r'http[s]?://\S+', 'URL', text)
    if REPLACE_NUMBERS:
        text = re.sub(r'\b\d+\b', 'NUMBER', text)
    if REMOVE_PUNCT:
        text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    if USE_STEMMING:
        words = [stemmer.stem(w) for w in words]
    return words

def get_files(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

def load_emails(folder, label):
    emails = []
    labels = []
    for fname in get_files(folder):
        with open(fname, encoding='latin1') as f:
            text = f.read()
            words = preprocess(text)
            emails.append(words)
            labels.append(label)
    return emails, labels

# --- Load all emails ---
train_emails, train_labels = [], []
test_emails, test_labels = [], []

categories = [
    ('20021010_easy_ham/easy_ham/train', 0),
    ('20030228_hard_ham/hard_ham/train', 0),
    ('20030228_spam/spam/train', 1)
]
for folder, label in categories:
    emails, labels = load_emails(folder, label)
    train_emails.extend(emails)
    train_labels.extend(labels)

categories_test = [
    ('20021010_easy_ham/easy_ham/test', 0),
    ('20030228_hard_ham/hard_ham/test', 0),
    ('20030228_spam/spam/test', 1)
]
for folder, label in categories_test:
    emails, labels = load_emails(folder, label)
    test_emails.extend(emails)
    test_labels.extend(labels)

# --- Build vocabulary from training set ---
vocab_counter = Counter()
for words in train_emails:
    vocab_counter.update(set(words) if BINARY_FEATURES else words)
vocab = sorted(vocab_counter)
word_idx = {word: i for i, word in enumerate(vocab)}

# --- Vectorize emails ---
def vectorize(emails):
    X = np.zeros((len(emails), len(vocab)), dtype=int)
    for i, words in enumerate(emails):
        counts = Counter(words)
        for word in counts:
            if word in word_idx:
                X[i, word_idx[word]] = 1 if BINARY_FEATURES else counts[word]
    return X

X_train = vectorize(train_emails)
X_test = vectorize(test_emails)
y_train = np.array(train_labels)
y_test = np.array(test_labels)

# --- Save to CSV ---
np.savetxt('X_train.csv', X_train, delimiter=',', fmt='%d')
np.savetxt('X_test.csv', X_test, delimiter=',', fmt='%d')
np.savetxt('y_train.csv', y_train, delimiter=',', fmt='%d')
np.savetxt('y_test.csv', y_test, delimiter=',', fmt='%d')

print(f"Vocabulary size: {len(vocab)}")
print("Feature vectors and labels saved as CSV files.")
