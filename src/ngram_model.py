from collections import defaultdict, Counter
import random
import math


class NGramModel:
    def __init__(self, n):
        if n < 1:
            raise ValueError("n must be at least 1")
        self.n = n
        self.counts = defaultdict(Counter)
        self.probs = {}

    def train(self, tokens):
        """
        Train the model on a list of tokens.
        For n=1, context is the empty tuple ().
        For n=2, context is 1 previous word.
        For n=3, context is 2 previous words.
        """
        if self.n == 1:
            for token in tokens:
                self.counts[()][token] += 1
        else:
            for i in range(len(tokens) - self.n + 1):
                context = tuple(tokens[i:i + self.n - 1])
                next_word = tokens[i + self.n - 1]
                self.counts[context][next_word] += 1

        self._compute_probabilities()

    def _compute_probabilities(self):
        """Convert counts into conditional probabilities."""
        self.probs = {}
        for context, counter in self.counts.items():
            total = sum(counter.values())
            self.probs[context] = {
                word: count / total
                for word, count in counter.items()
            }

    def predict_next(self, context):
        """
        Predict a next word from a context.
        context should be:
          - () for unigram
          - tuple of length n-1 for bigram/trigram
        """
        if self.n == 1:
            context = ()
        else:
            if len(context) != self.n - 1:
                raise ValueError(f"Context must have length {self.n - 1}")

        if context not in self.probs:
            return None

        words = list(self.probs[context].keys())
        probs = list(self.probs[context].values())
        return random.choices(words, weights=probs, k=1)[0]

    def generate(self, max_words=20, start_context=None):
        """
        Generate text from the model.
        """
        if self.n == 1:
            context = ()
            generated = []
        else:
            if start_context is None:
                context = random.choice(list(self.probs.keys()))
            else:
                if len(start_context) != self.n - 1:
                    raise ValueError(f"start_context must have length {self.n - 1}")
                context = tuple(start_context)
            generated = list(context)

        for _ in range(max_words):
            next_word = self.predict_next(context)
            if next_word is None:
                break

            generated.append(next_word)

            if self.n > 1:
                context = tuple(generated[-(self.n - 1):])

        return " ".join(generated)

    def get_contexts(self):
        """Optional helper to inspect learned contexts."""
        return self.probs

    def perplexity(self, tokens):
        """
        Compute perplexity of the model on a list of tokens.
        Lower is better. Skips unseen contexts (no smoothing).
        """
        log_prob_sum = 0
        count = 0

        if self.n == 1:
            for token in tokens:
                context = ()
                if context in self.probs and token in self.probs[context]:
                    prob = self.probs[context][token]
                    log_prob_sum += math.log(prob)
                    count += 1
        else:
            for i in range(len(tokens) - self.n + 1):
                context = tuple(tokens[i:i + self.n - 1])
                target = tokens[i + self.n - 1]
                if context in self.probs and target in self.probs[context]:
                    prob = self.probs[context][target]
                    log_prob_sum += math.log(prob)
                    count += 1

        if count == 0:
            return float("inf")

        avg_log_prob = log_prob_sum / count
        return round(math.exp(-avg_log_prob), 2)