import os
import re
import pickle


def load_corpus(path, encoding="utf-8"):
    """
    Load a text corpus from a file.
    """
    with open(path, "r", encoding=encoding) as f:
        return f.read()


def save_text(path, text, encoding="utf-8"):
    """
    Save text to a file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "w", encoding=encoding) as f:
        f.write(text)


def clean_gutenberg_text(text):
    """
    Remove Project Gutenberg header/footer boilerplate if present.
    If the markers are not found, returns the original text.
    """
    start_marker = r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*"
    end_marker = r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*"

    start_match = re.search(start_marker, text, flags=re.IGNORECASE | re.DOTALL)
    end_match = re.search(end_marker, text, flags=re.IGNORECASE | re.DOTALL)

    if start_match and end_match:
        return text[start_match.end():end_match.start()].strip()

    return text


def add_sentence_tokens(tokens, n):
    """
    Add start and end tokens for n-gram training.

    Example:
        n=3 -> ['<s>', '<s>', token1, token2, ..., '</s>']
    """
    if n < 1:
        raise ValueError("n must be at least 1")

    return ["<s>"] * (n - 1) + tokens + ["</s>"]


def preview_tokens(tokens, count=20):
    """
    Print the first few tokens and total token count.
    """
    print(tokens[:count])
    print(f"Total tokens: {len(tokens)}")


def save_model(model, name, folder="models"):
    """
    Save a trained NGramModel to disk using pickle.

    Example:
        save_model(model, "bigram")  ->  saves to models/bigram.pkl
    """
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved: {path}")


def load_model(name, folder="models"):
    """
    Load a saved NGramModel from disk.

    Example:
        load_model("bigram")  ->  loads from models/bigram.pkl
    """
    path = os.path.join(folder, f"{name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved model found at '{path}'. Train and save it first.")
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"Loaded: {path}")
    return model