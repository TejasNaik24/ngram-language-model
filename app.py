import streamlit as st
from src.utils import load_model, load_corpus, clean_gutenberg_text
from src.tokenizer import tokenize

st.set_page_config(page_title="N-Gram Language Model", page_icon="🧠")

st.title("N-Gram Language Model")

# --- Sidebar: model selection ---
st.sidebar.header("Model Settings")
model_name = st.sidebar.radio(
    "Choose a model:",
    ["unigram", "bigram", "trigram"],
    index=1
)

st.sidebar.divider()
st.sidebar.markdown("**How each model works:**")
st.sidebar.markdown("**Unigram** — predicts based on word frequency alone, no context")
st.sidebar.markdown("**Bigram** — uses the previous 1 word as context")
st.sidebar.markdown("**Trigram** — uses the previous 2 words as context")


# --- Load model ---
@st.cache_resource
def get_model(name):
    return load_model(name)


@st.cache_resource
def get_tokens():
    text = load_corpus("data/corpus.txt")
    text = clean_gutenberg_text(text)
    return tokenize(text)


model = get_model(model_name)
tokens = get_tokens()

st.markdown(f"### Using: `{model_name}` model")
st.divider()

# --- Perplexity section ---
st.subheader("Model Perplexity")
st.caption("Lower is better — measures how surprised the model is by the text. Trigram wins because more context = better predictions.")

col_u, col_b, col_t = st.columns(3)

scores = {
    "unigram": get_model("unigram").perplexity(tokens),
    "bigram":  get_model("bigram").perplexity(tokens),
    "trigram": get_model("trigram").perplexity(tokens),
}

with col_u:
    st.metric(
        label="Unigram",
        value=f"{scores['unigram']:,}",
        delta="no context",
        delta_color="off"
    )
with col_b:
    st.metric(
        label="Bigram",
        value=f"{scores['bigram']:,}",
        delta="1 word context",
        delta_color="off"
    )
with col_t:
    st.metric(
        label="Trigram",
        value=f"{scores['trigram']:,}",
        delta="2 word context",
        delta_color="off"
    )

st.divider()

# --- Tabs: Predict / Generate ---
tab1, tab2 = st.tabs(["Predict Next Word", "Generate Sentence"])

with tab1:
    st.subheader("Predict the next word")
    user_input = st.text_input(
        "Type a word or phrase:",
        placeholder="e.g. mr darcy"
    )

    if user_input:
        input_tokens = tokenize(user_input)
        n = model.n

        if n == 1:
            context = ()
        else:
            context = tuple(input_tokens[-(n - 1):])

        prediction = model.predict_next(context)

        if prediction:
            st.success(f"{user_input.strip()} → {prediction}")
        else:
            st.warning("Unknown context — the model hasn't seen this before. Try different words.")

with tab2:
    st.subheader("Generate a full sentence")

    col1, col2 = st.columns(2)
    with col1:
        start_input = st.text_input(
            "Starting word(s) (optional):",
            placeholder="e.g. she was"
        )
    with col2:
        length = st.slider(
            "Number of words to generate:",
            min_value=5,
            max_value=50,
            value=20
        )

    if st.button("Generate"):
        start_context = None
        if start_input:
            start_tokens = tokenize(start_input)
            n = model.n
            if n > 1 and len(start_tokens) >= n - 1:
                start_context = start_tokens[-(n - 1):]

        result = model.generate(max_words=length, start_context=start_context)
        st.info(result)