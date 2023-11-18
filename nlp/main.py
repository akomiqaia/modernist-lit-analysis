import pandas as pd
import json
import glob
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from nltk.corpus import stopwords


def read_file(file_path):
    with open(file_path, 'r') as f:
        file_contents = f.read()
    return file_contents


def write_data(data, file):
    with open(file, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def remove_stops(text, stops):
    # uses regex to remove all AC numbers
    text = re.sub(r"AC\/\d{1,4}\/\d{1,4}", "", text)

    # removes all stop words, including months
    words = text.split()
    final = []
    for word in words:
        if word not in stops:
            final.append(word)

    # reassembles the text without stop words
    final = " ".join(final)

    # removes all punctuation
    final = final.translate(str.maketrans("", "", string.punctuation))

    # removes all numbers
    final = "".join([i for i in final if not i.isdigit()])

    # eliminates double white spaces
    while "  " in final:
        final = final.replace("  ", " ")
    return (final)


def clean_texts(text):
    # remove lines before the start of the book
    # remove parts after the end of the book

    # enable stop words
    stops = stopwords.words('english')
    stops.extend(["ethan", "frome", "edith", "wharton", "chapter",
                 "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"])
    return remove_stops(text, stops)


file_path = "./texts/Edith Wharton - Ethan Frome.txt"
cleaned_text = ""
try:
    cleaned_text = clean_texts(read_file(file_path))
except FileNotFoundError:
    print(f"The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {str(e)}")

vectorizer = TfidfVectorizer(
    lowercase=True,
    max_features=100,
    max_df=0.8,
    min_df=5,
    ngram_range=(1, 3),
    stop_words="english"

)

vectors = vectorizer.fit_transform(cleaned_text)
print(vectors)
