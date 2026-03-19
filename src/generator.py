import random


def sample_next(prob_dict):
    """
    Sample the next word from a probability dictionary.

    prob_dict format:
        {
            "word1": 0.2,
            "word2": 0.5,
            "word3": 0.3
        }
    """
    if not prob_dict:
        return None

    words = list(prob_dict.keys())
    probs = list(prob_dict.values())

    return random.choices(words, weights=probs, k=1)[0]


def generate_text(model, max_words=20, start_context=None):
    """
    Generate text from a trained NGramModel.

    Args:
        model: trained NGramModel with attributes:
            - n
            - probs
        max_words: maximum number of words to generate
        start_context: optional tuple/list of starting words

    Returns:
        A generated string.
    """
    if not hasattr(model, "probs") or not model.probs:
        return ""

    n = model.n

    if n == 1:
        context = ()
        generated = []
    else:
        if start_context is None:
            context = random.choice(list(model.probs.keys()))
        else:
            if len(start_context) != n - 1:
                raise ValueError(f"start_context must have length {n - 1}")
            context = tuple(start_context)

        generated = list(context)

    for _ in range(max_words):
        if context not in model.probs:
            break

        next_word = sample_next(model.probs[context])
        if next_word is None:
            break

        generated.append(next_word)

        if n > 1:
            context = tuple(generated[-(n - 1):])

    return " ".join(generated)


def generate_from_random_context(model, max_words=20):
    """
    Generate text starting from a random known context.
    """
    return generate_text(model, max_words=max_words, start_context=None)