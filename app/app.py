# app/app.py

import pickle
import torch
import numpy as np
from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModel
import os

# --- 1. Initialize the Flask App ---
app = Flask(__name__)

# --- 2. Load All Models and Tokenizer (This happens only once at startup) ---
MODEL_PATH = '../models/'
PLM_MODEL_NAME = "Rostlab/prot_bert"
MAX_LENGTH = 512

print("➡️  Loading all models and tokenizer, please wait...")

# Load the trained scikit-learn classifier and the class labels
try:
    with open(os.path.join(MODEL_PATH, 'protein_classifier.pkl'), 'rb') as f:
        classifier = pickle.load(f)
    with open(os.path.join(MODEL_PATH, 'class_labels.pkl'), 'rb') as f:
        class_labels = pickle.load(f)
except FileNotFoundError:
    print("❌ ERROR: Model files not found. Make sure 'protein_classifier.pkl' and 'class_labels.pkl' are in the 'models' directory.")
    exit()

# Load the pre-trained Protein Language Model (ProtBERT)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(PLM_MODEL_NAME)
plm_model = AutoModel.from_pretrained(PLM_MODEL_NAME).to(device)

print("✅ Models and tokenizer loaded successfully.")
print(f"Running on device: {device}")


# --- 3. Helper Function to Generate Embeddings ---
def get_embedding(sequence: str) -> np.ndarray:
    """Converts a protein sequence into a numerical embedding."""
    sequence_spaced = " ".join(list(sequence))
    inputs = tokenizer(
        sequence_spaced,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = plm_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()


# --- 4. Define Web Application Routes ---

@app.route('/')
def home():
    """Renders the main input page (index.html)."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the prediction logic and renders the new result page.
    """
    try:
        sequence = request.form['protein_sequence']
        
        if not sequence.strip():
            # If the form is empty, just reload the home page with an error
            return render_template('index.html', prediction_result="Error: Please enter a protein sequence.")

        # 1. Generate the embedding (this is the slow part)
        embedding = get_embedding(sequence)
        
        # 2. Get prediction probabilities for ALL classes
        probabilities = classifier.predict_proba(embedding)[0]
        
        # 3. Find the highest probability and its corresponding class
        max_prob_index = np.argmax(probabilities)
        predicted_class = class_labels[max_prob_index]
        confidence_score = probabilities[max_prob_index]
        
        # 4. Render the NEW result.html page with the results
        return render_template(
            'result.html',
            prediction=predicted_class,
            confidence_score=f"{confidence_score*100:.2f}",
            sequence=sequence
        )

    except Exception as e:
        error_message = f"An error occurred during prediction: {str(e)}"
        return render_template('index.html', prediction_result=error_message)


# --- 5. Run the Flask App ---
if __name__ == '__main__':
    app.run(debug=True)