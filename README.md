# Protein-Infer 🧬

**A Tool for Predicting Protein Subcellular Localization Using Protein Language Models.**

---

## Abstract

Protein-Infer is a machine learning tool developed as a final year capstone project. It leverages the power of the pre-trained Protein Language Model, ProtBERT, to generate high-quality numerical embeddings from amino acid sequences. These embeddings are then used to train a Random Forest classifier to predict the subcellular location of a given protein (e.g., Nucleus, Cytoplasm, Mitochondrion). The project includes a complete pipeline for training and evaluation, as well as a simple web application for live predictions.

## 📊 Key Results

- **Model:** Random Forest Classifier
- **Feature Extractor:** ProtBERT
<<<<<<< HEAD
- **Final Accuracy:** 65%
- **Key Finding:** The model performed best on distinct classes like 'Plastid' but occasionally confused 'Cytoplasm' and 'Nucleus'.
=======
- **Final Accuracy:** [Enter your final accuracy here, e.g., 75.34%]
- **Key Finding:** [Enter a key finding, e.g., The model performed best on distinct classes like 'Plastid' but occasionally confused 'Cytoplasm' and 'Nucleus'.]
>>>>>>> 1134c4c (Initial project setup with data, notebooks, and app)

## ✨ Features

- **Sequence to Prediction:** End-to-end pipeline from a raw protein sequence to a location prediction.
- **PLM-Powered:** Uses state-of-the-art ProtBERT for feature extraction.
- **Web Interface:** A simple Flask application to make live predictions.
- **Reproducible:** Notebooks for data processing, training, and evaluation are included.

## ⚙️ Tech Stack

- **Backend:** Python, Flask
- **Machine Learning:** Scikit-learn, PyTorch, Hugging Face Transformers
- **Data Handling:** Pandas, NumPy
- **Frontend:** HTML, CSS

## 📂 Project Structure
Protein-Infer/
├── app/                # Contains the Flask web application
│   ├── templates/
│   │   ├── index.html      # The main input page
│   │   └── result.html     # The prediction result page
│   ├── app.py              # The backend logic for the web app
│   └── requirements.txt    # Python dependencies for the app
│
├── data/               # Stores the train.csv dataset
│
├── models/             # Stores the trained classifier and class labels
│   ├── protein_classifier.pkl
│   └── class_labels.pkl
│
├── notebooks/          # Colab notebooks for the ML workflow
│   ├── 01_Data_Exploration.ipynb
│   └── 02_Training_and_Evaluation.ipynb
│
└── README.md

## 🚀 Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/tararauzumaki/Protein-Infer.git
    cd Protein-Infer
    ```

2.  **Set up a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies for the web app:**
    ```bash
    pip install -r app/requirements.txt
    ```

##  kullanım

### Training the Model

To retrain the model or see the evaluation, open and run the notebook in `notebooks/`. Ensure you have downloaded the dataset into the `data/` folder.

### Running the Web Application

1.  **Navigate to the app directory:**
    ```bash
    cd app
    ```

2.  **Run the Flask application:**
    ```bash
    python app.py
    ```

3.  **Open your browser** and go to `http://127.0.0.1:5000`.

<<<<<<< HEAD
---
=======
---
>>>>>>> 1134c4c (Initial project setup with data, notebooks, and app)
