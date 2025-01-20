import streamlit as st
import numpy as np
import pickle

# Load models
@st.cache_resource
def load_model(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

glove = load_model("glove.pkl")
skipgram_negative = load_model("skipgram_negative.pkl")
skipgram = load_model("skipgram.pkl")

# Dictionary of models
models = {
    "GloVe": glove,
    "Skipgram Negative": skipgram_negative,
    "Skipgram": skipgram,
}

def compute_dot_product(query_vec, corpus):
    """
    Compute dot products between the query vector and all vectors in the corpus.
    Returns indices of the top 10 most similar contexts.
    """
    dot_products = np.dot(corpus, query_vec)
    top_indices = np.argsort(dot_products)[::-1][:10]
    return top_indices, dot_products[top_indices]

# Streamlit App
st.markdown("<h1 style='text-align: center;'>Context Similarity Search Application</h1>", unsafe_allow_html=True)

# Center the text
st.markdown("<h3 style='text-align: center;'>Search similar contexts using pre-trained embeddings.</h3>", unsafe_allow_html=True)

# Input Query
query = st.text_input("Enter your search query:")

# Model Selection
model_choice = st.selectbox("Choose an embedding model:", list(models.keys()))

# Create the Enter button
enter_button = st.button("Enter")

# Load selected model
if model_choice:
    selected_model = models[model_choice]
    
    # Handle GloVe embeddings (assuming it's a dictionary of word vectors)
    if model_choice == "GloVe":
        if isinstance(selected_model, dict):
            corpus = np.array(list(selected_model.values()))  # Get all vectors in an array
            contexts = list(selected_model.keys())            # Get all words corresponding to vectors
        else:
            st.error("GloVe model structure not supported.")
    
    # Handle Skipgram Negative and Skipgram models (assuming they are dictionaries or arrays)
    elif model_choice in ["Skipgram Negative", "Skipgram"]:
        if isinstance(selected_model, dict):
            corpus = np.array(list(selected_model.values()))  # Get all vectors
            contexts = list(selected_model.keys())            # Get all words corresponding to vectors
        elif isinstance(selected_model, np.ndarray):
            corpus = selected_model  # Embeddings as a numpy array of vectors
            contexts = list_of_words  # You need to provide the list of words corresponding to vectors
        else:
            st.error(f"{model_choice} model does not have a supported format.")
    
    if enter_button and query:  # Trigger search when Enter button is clicked
        if query in contexts:
            # Vectorize the query
            query_vec = selected_model[query] if model_choice == "GloVe" else selected_model[query]

            # Compute the dot product
            top_indices, top_scores = compute_dot_product(query_vec, corpus)

            # Display results in two columns (5 per column)
            st.subheader("Top 10 Most Similar Contexts")
            
            # Create two columns for split-screen view
            col1, col2 = st.columns(2)

            with col1:
                for idx, (score, context) in enumerate(zip(top_scores[:5], np.array(contexts)[top_indices[:5]])):
                    st.write(f"{idx + 1}. **{context}** (Score: {score:.4f})")
            
            with col2:
                for idx, (score, context) in enumerate(zip(top_scores[5:], np.array(contexts)[top_indices[5:]])):
                    st.write(f"{idx + 6}. **{context}** (Score: {score:.4f})")

            # Add 1-inch space between the columns for extra padding
            st.markdown("<br><br><br><br>", unsafe_allow_html=True)
                
        else:
            st.error(f"'{query}' not found in the model's vocabulary.")
