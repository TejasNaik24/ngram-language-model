from src.tokenizer import tokenize
from src.ngram_model import NGramModel
from src.utils import load_corpus, clean_gutenberg_text
from src.generator import generate_text

text = load_corpus("data/corpus.txt")
text = clean_gutenberg_text(text)
tokens = tokenize(text)

model = NGramModel(2)
model.train(tokens)

print(generate_text(model, max_words=20))