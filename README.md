# Plagiarism-Checker
## Problem Statement
Plagiarism is a serious issue in academic and professional settings, where individuals may try to pass off someone else's work as their own. To combat this problem, a plagiarism detection system using TF-IDF (Term Frequency-Inverse Document Frequency) is needed to identify similarities between documents and detect potential cases of plagiarism.
## Description
In this program we detect plagiarism between two text files by calculating the cosine similarity between them using the TF-IDF (Term Frequency-Inverse Document Frequency) technique. It compares multiple text files and calculates the similarity between them to detect potential cases of plagiarism. The code reads text files, preprocesses the text by removing punctuation and converting to lowercase, calculates the TF-IDF values for each term in the documents, and then computes the cosine similarity between pairs of documents. The cosine similarity score indicates the similarity between the documents, with a higher score indicating higher similarity. The code is capable of handling multiple input files and provides similarity scores for each pair of documents.
## Run in Local PC
### Step 1: 
If you do not have Python installed on your PC, download and install the latest version of Python from the official Python website (https://www.python.org/).
### Step 2: 
Clone the directory:
```
git clone https://github.com/Siba-Prasad-Pattnaik/Plagiarism-Checker.git
```
### Step 3: 
Place the text files that you want to compare for plagiarism in the same directory where your code is located. Make sure the files have the appropriate file extensions (.txt).
### Step 4:
Install required libraries:
```
pip install nltk
pip install scikit-learn
```
Run The Code!!
