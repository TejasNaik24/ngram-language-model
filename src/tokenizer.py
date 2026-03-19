"""
Tokenizer used to lowercase, remove punctuation, and split.
"""

import string

def tokenize(text):
    text = text.lower()
    clean_text = "".join(char for char in text if char not in string.punctuation)
    return clean_text.split()