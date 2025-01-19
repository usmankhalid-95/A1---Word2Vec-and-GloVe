import os
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load models
def load_model(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained models
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

@app.route("/", methods=["GET", "POST"])
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get the input query and model choice from the form
        query = request.form.get("query")
        model_choice = request.form.get("model")
        enter_button = request.form.get("enter")

        # Load the selected model
        selected_model = models.get(model_choice)
        
        if selected_model:
            # Handle GloVe embeddings (assuming it's a dictionary of word vectors)
            if model_choice == "GloVe":
                if isinstance(selected_model, dict):
                    corpus = np.array(list(selected_model.values()))  # Get all vectors in an array
                    contexts = list(selected_model.keys())            # Get all words corresponding to vectors
                else:
                    return "Error: GloVe model structure not supported."

            # Handle Skipgram Negative and Skipgram models
            elif model_choice in ["Skipgram Negative", "Skipgram"]:
                if isinstance(selected_model, dict):
                    corpus = np.array(list(selected_model.values()))  # Get all vectors
                    contexts = list(selected_model.keys())            # Get all words corresponding to vectors
                elif isinstance(selected_model, np.ndarray):
                    corpus = selected_model  # Embeddings as a numpy array of vectors
                    contexts = list_of_words  # You need to provide the list of words corresponding to vectors
                else:
                    return f"Error: {model_choice} model does not have a supported format."

            if query in contexts:
                query_vec = selected_model[query] if model_choice == "GloVe" else selected_model[query]
                top_indices, top_scores = compute_dot_product(query_vec, corpus)

                # Split results into two columns (5 per column)
                results_left = [(contexts[idx], top_scores[i]) for i, idx in enumerate(top_indices[:5])]
                results_right = [(contexts[idx], top_scores[i]) for i, idx in enumerate(top_indices[5:])]

                return render_template("index.html", results_left=results_left, results_right=results_right, query=query, model_choice=model_choice)
            else:
                return render_template("index.html", error=f"'{query}' not found in the model's vocabulary.", query=query, model_choice=model_choice)

    return render_template("index.html", results_left=None, results_right=None)

if __name__ == "__main__":
    app.run(debug=True)
