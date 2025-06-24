import streamlit as st
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
import joblib
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
torch.classes.__path__ = []
# Download stopwords and lemmatizer
nltk.download('stopwords')
nltk.download('wordnet')

# ----------------------------------------
# Load Model and Scaler
# ----------------------------------------
class SimpleNN(torch.nn.Module):
    def __init__(self, input_dim=386):  # 384 + 2 binary features
        super(SimpleNN, self).__init__()
        self.hidden = torch.nn.Linear(input_dim, 128)
        self.relu = torch.nn.ReLU()
        self.hidden1 = torch.nn.Linear(128, 64)
        self.output = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.relu(self.hidden1(x))
        x = self.output(x)
        return x

# Initialize and load model
model = SimpleNN()
model.load_state_dict(torch.load("model.pt", map_location=torch.device('cpu')))
model.eval()

# Load scaler (fit on log1p(y) then standardized)
scler = joblib.load("scaler.pkl")

# ----------------------------------------
# Text Preprocessing
# ----------------------------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    words = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return tokens

# Load sentence transformer
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# ----------------------------------------
# Streamlit UI
# ----------------------------------------
st.set_page_config(page_title="YouTube View Predictor", page_icon="📊")
st.title("📈 YouTube Video View Predictor")

channel_name = st.text_input("Channel Name")
title = st.text_input("Video Title")
tags = st.text_input("Video Tags (comma-separated)")
description = st.text_area("Video Description")
ratings_disabled = st.radio("Are Ratings Disabled?", ["No", "Yes"]) == "Yes"
comments_disabled = st.radio("Are Comments Disabled?", ["No", "Yes"]) == "Yes"

if st.button("Predict Views"):
    if not all([channel_name, title, tags, description]):
        st.warning("Please fill in all fields.")
    else:
        with st.spinner("🔍 Processing your input..."):
            # Preprocess
            text_fields = [channel_name, title, tags, description]
            processed_fields = [preprocess(text) for text in text_fields]
            combined_text = " ".join([" ".join(tokens) for tokens in processed_fields])

            # Encode
            embed1 = embed_model.encode([combined_text], show_progress_bar=True)  # Shape: (1, 384)

            # Combine with binary features
            binary_data = np.array([[int(comments_disabled), int(ratings_disabled)]])
            pred_data = np.hstack([embed1, binary_data])
            pred_tensor = torch.tensor(pred_data, dtype=torch.float32)

            # Predict
            with torch.no_grad():
                user_preds = model(pred_tensor).numpy()

            # Inverse transform
            user_preds_inv = scler.inverse_transform(user_preds.reshape(-1, 1))
            user_preds_final = np.exp(user_preds_inv) - 1
            predicted_views = int(np.clip(user_preds_final[0][0], 0, None))

        st.success(f"✅ Predicted View Count: **{predicted_views:,}**")
