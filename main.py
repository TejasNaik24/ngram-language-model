from src.tokenizer import tokenize
from src.ngram_model import NGramModel
from src.utils import load_corpus, clean_gutenberg_text, save_model
from src.generator import generate_text

text = load_corpus("data/corpus.txt")
text = clean_gutenberg_text(text)
tokens = tokenize(text)

print(f"Total tokens: {len(tokens)}\n")

names = {1: "unigram", 2: "bigram", 3: "trigram"}

for n in [1, 2, 3]:
    print(f"=== {names[n].upper()} ===")
    model = NGramModel(n)
    model.train(tokens)
    save_model(model, names[n])
    print(generate_text(model, max_words=20))
    print()