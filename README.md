# N-Gram Language Model From Scratch

## Overview

This project implements a **unigram, bigram, and trigram language model from scratch** using only Python's standard library, and is deployed on **Streamlit** to provide an interactive web interface for exploring how n-gram models work.

The models are trained on two classic Jane Austen novels — **Pride and Prejudice** and **Sense and Sensibility** — sourced from Project Gutenberg, giving the model a rich literary corpus of ~230,000 tokens to learn from.

This project aims to demonstrate a deep understanding of statistical language modeling — including tokenization, n-gram counting, probability estimation, and perplexity evaluation — without relying on any NLP libraries like NLTK, spaCy, or HuggingFace.

The project is written with clean, modular code across separate files for tokenization, modeling, generation, and utilities. It was built as preparation for NLP research to understand the statistical foundations that modern neural language models are built on top of.

## Check out the website [here](https://ngram-language-model.streamlit.app)!

---

## Streamlit Web Interface

The n-gram language model is deployed with **Streamlit**, providing a clean and interactive UI to explore how each model behaves.

The web app allows users to:

- Switch between **unigram, bigram, and trigram** models from the sidebar.
- View **perplexity scores** for all three models side by side to quantitatively compare their quality.
- **Predict the next word** by typing any word or phrase and seeing what the model predicts comes next.
- **Generate full sentences** from an optional starting phrase, with adjustable output length.

---

## What are N-Grams?

An n-gram is a sequence of n consecutive tokens (words) from a text. N-gram language models predict the next word in a sequence based on the previous n-1 words. This is called the **Markov assumption** — the probability of the next word depends only on a fixed window of previous words, not the entire history.

| Model         | Context Used                     | Example                     |
| :------------ | :------------------------------- | :-------------------------- |
| Unigram (n=1) | No context — pure word frequency | P(word)                     |
| Bigram (n=2)  | Previous 1 word                  | P(word \| previous word)    |
| Trigram (n=3) | Previous 2 words                 | P(word \| previous 2 words) |

### Example

Take the sentence: `"she was not happy"`

- **Unigrams**: `["she", "was", "not", "happy"]`
- **Bigrams**: `["she was", "was not", "not happy"]`
- **Trigrams**: `["she was not", "was not happy"]`

---

## How Probabilities are Estimated

The probability of the next word given its context is estimated using **Maximum Likelihood Estimation (MLE)** — simply counting occurrences in the training corpus:

$$P(w_i \mid w_{i-n+1}, \ldots, w_{i-1}) = \frac{\text{count}(w_{i-n+1}, \ldots, w_{i-1}, w_i)}{\text{count}(w_{i-n+1}, \ldots, w_{i-1})}$$

For example, if "she was" appears 200 times in the corpus, and "she was not" appears 40 times:

$$P(\text{not} \mid \text{she was}) = \frac{40}{200} = 0.2$$

---

## Model Evaluation: Perplexity

Perplexity measures how "surprised" a model is by a sequence of text. **Lower perplexity = better model.**

$$PP(W) = \exp\left(-\frac{1}{N} \sum_{i=1}^{N} \log P(w_i \mid \text{context})\right)$$

Where:

- $W$ is the sequence of tokens being evaluated
- $N$ is the number of tokens scored
- $P(w_i \mid \text{context})$ is the model's predicted probability for each token

### Results on Training Corpus (~230k tokens)

| Model   | Perplexity | Context |
| :------ | :--------: | :-----: |
| Unigram |    ~676    |  none   |
| Bigram  |    ~52     | 1 word  |
| Trigram |    ~4.7    | 2 words |

The improvement from unigram to trigram represents a **143x reduction in perplexity**, showing how much context window size matters for language modeling quality.

> **Note:** Trigram perplexity appears low because unseen contexts are skipped rather than penalized (no smoothing). This is a known limitation — adding Laplace or Kneser-Ney smoothing would produce a fairer evaluation.

---

## Project Structure

```
ngram-language-model/
│
├── app.py                  # Streamlit web interface
├── main.py                 # Training script — trains and saves all 3 models
├── requirements.txt
├── README.md
│
├── data/
│   └── corpus.txt          # Training corpus (Pride and Prejudice + Sense and Sensibility)
│
├── models/
│   ├── unigram.pkl
│   ├── bigram.pkl
│   └── trigram.pkl
│
└── src/
    ├── tokenizer.py        # Lowercasing, punctuation removal, splitting
    ├── ngram_model.py      # NGramModel class — counting, probabilities, generation, perplexity
    ├── generator.py        # Text generation logic
    └── utils.py            # I/O helpers — load corpus, save/load models
```

---

## Implementation Details

### Tokenizer (`src/tokenizer.py`)

The tokenizer performs three steps:

1. Lowercase the entire text
2. Remove punctuation using Python's `string.punctuation`
3. Split on whitespace

```python
def tokenize(text):
    text = text.lower()
    clean_text = "".join(char for char in text if char not in string.punctuation)
    return clean_text.split()
```

---

### N-Gram Model (`src/ngram_model.py`)

The core of the project. The `NGramModel` class handles everything — training, probability estimation, prediction, generation, and perplexity.

#### Building N-Gram Counts

The key generalization that supports unigram, bigram, and trigram in one class:

```python
for i in range(len(tokens) - self.n + 1):
    context = tuple(tokens[i:i + self.n - 1])  # () for unigram, (w,) for bigram, (w1,w2) for trigram
    next_word = tokens[i + self.n - 1]
    self.counts[context][next_word] += 1
```

#### Converting Counts to Probabilities

```python
for context, counter in self.counts.items():
    total = sum(counter.values())
    self.probs[context] = {
        word: count / total
        for word, count in counter.items()
    }
```

#### Text Generation

Generation works by:

1. Starting from a known context (random or user-provided)
2. Sampling the next word from the probability distribution
3. Shifting the context window forward
4. Repeating until max length is reached or an unknown context is hit

```python
next_word = random.choices(words, weights=probs, k=1)[0]
```

---

### Perplexity (`src/ngram_model.py`)

```python
def perplexity(self, tokens):
    log_prob_sum = 0
    count = 0

    for i in range(len(tokens) - self.n + 1):
        context = tuple(tokens[i:i + self.n - 1])
        target = tokens[i + self.n - 1]
        if context in self.probs and target in self.probs[context]:
            prob = self.probs[context][target]
            log_prob_sum += math.log(prob)
            count += 1

    avg_log_prob = log_prob_sum / count
    return round(math.exp(-avg_log_prob), 2)
```

---

## Limitations and Research Observations

### Data Sparsity

As n increases, the number of possible n-grams grows exponentially. Most trigrams in the English language will never appear in any finite corpus, meaning the model returns no prediction for unseen contexts. This is the core motivation for **smoothing techniques** like Laplace smoothing and Kneser-Ney smoothing.

### No Long-Range Dependencies

N-gram models only capture local context. A sentence like:

> "The keys to the cabinet **are** missing"

requires understanding that "keys" (not "cabinet") determines "are" — a dependency the trigram model cannot capture. This limitation is what motivated the development of RNNs, LSTMs, and ultimately Transformers.

### Why This Still Matters in 2026

Even with transformers dominating NLP:

- N-gram models are fully interpretable — you can inspect every probability directly
- They are extremely fast to train and run
- They serve as clean baselines for evaluating more complex models
- Understanding their failures is the best way to understand why modern architectures look the way they do

---

## Training Data

Both books were sourced from [Project Gutenberg](https://www.gutenberg.org/) as plain text UTF-8 files. Gutenberg boilerplate headers and footers were stripped using regex before training.

| Book                  | Author      | Tokens (approx) |
| :-------------------- | :---------- | :-------------: |
| Pride and Prejudice   | Jane Austen |    ~124,000     |
| Sense and Sensibility | Jane Austen |    ~120,000     |
| **Total**             |             |  **~230,000**   |

---

## To Run the Code Locally

1. **Clone the repository:**

   ```bash
   git clone https://github.com/TejasNaik24/ngram-language-model.git
   cd ngram-language-model
   ```

2. **(Recommended) Create a virtual environment:**

   ```bash
   python -m venv .venv
   ```

   Then activate it:
   - **macOS/Linux:**
     ```bash
     source .venv/bin/activate
     ```
   - **Windows:**
     ```bash
     .venv\Scripts\activate
     ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Train the models:**

   ```bash
   python main.py
   ```

   This trains all three models on `data/corpus.txt` and saves them as `.pkl` files in the `models/` folder.

5. **Run the Streamlit app:**

   ```bash
   streamlit run app.py
   ```

6. Once launched, Streamlit will open automatically at:

   http://localhost:8501

---
