import os
import re


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