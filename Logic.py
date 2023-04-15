"""In this program we detect plagiarism between two text files by calculating the cosine similarity 
between them using the TF-IDF (Term Frequency-Inverse Document Frequency) technique."""


import string
from collections import Counter
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

nltk.download('stopwords')

"""This line downloads the NLTK (Natural Language Toolkit) stopwords, which are common words such as
"a", "an", "the", etc. that are often removed from text during text processing as they do not carry
much meaning."""

# Function to remove punctuation and convert to lowercase
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

"""This function takes a text string as input and performs text preprocessing tasks, including 
converting the text to lowercase and removing all punctuation characters."""

# Function to calculate cosine similarity between two texts
def my_cosine_similarity(text1, text2):
    text1 = preprocess(text1)
    text2 = preprocess(text2)

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    
    # Fit and transform the texts
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    
    # Calculate cosine similarity
    similarity = sklearn_cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    
    return similarity[0][0]

"""It creates a TfidfVectorizer object, which is used to convert the text into TF-IDF vectors. 
The fit_transform method of the vectorizer is used to fit the text data and transform it into 
TF-IDF vectors. Finally, it calculates the cosine similarity between the two TF-IDF vectors using
the cosine_similarity function"""

# Load files
file1 = open("file1.txt", "r")
file2 = open("file2.txt", "r")

text1 = file1.read()
text2 = file2.read()

# Calculate cosine similarity
similarity = my_cosine_similarity(text1, text2)
print("Cosine Similarity: {:.2f}".format(similarity * 100)+"%")


