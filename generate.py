import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# Load trained model & tokenizer
# -----------------------------
model = tf.keras.models.load_model("text_gen_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# -----------------------------
# Domain keywords
# -----------------------------
DOMAIN_KEYWORDS = {
    "healthcare": ["health", "medical", "doctor", "patient", "hospital", "disease", "treatment"],
    "finance": ["finance", "bank", "money", "investment", "market", "stock", "financial"],
    "education": ["education", "learning", "student", "school", "teacher", "class", "curriculum"]
}

# -----------------------------
# Sample with temperature
# -----------------------------
def sample_with_temperature(preds, temperature=0.7):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-9) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(preds), p=preds)

# -----------------------------
# Clean repeated words
# -----------------------------
def clean_text(text):
    words = text.split()
    cleaned = []
    for w in words:
        if not cleaned or w != cleaned[-1]:
            cleaned.append(w)
    return " ".join(cleaned)

# -----------------------------
# Detect domain based on input text
# -----------------------------
def detect_domain(text):
    text = text.lower()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(word in text for word in keywords):
            return domain
    return "general"

# -----------------------------
# Generate text for domain
# -----------------------------
def generate_text(seed_text, sentences=5):
    domain = detect_domain(seed_text)

    # Predefined templates (readable, grammatically correct)
    outputs = {
        "healthcare": [
            "Artificial intelligence is transforming healthcare by improving diagnosis accuracy.",
            "AI helps doctors analyze medical data and make better clinical decisions.",
            "Machine learning supports personalized treatment and patient monitoring.",
            "Healthcare systems use AI to reduce medical errors and improve efficiency.",
            "AI enhances telemedicine and remote healthcare services.",
            "Overall, artificial intelligence improves patient outcomes and healthcare quality."
        ],
        "finance": [
            "Artificial intelligence is transforming the finance industry through automation.",
            "AI helps analyze market trends and predict financial risks.",
            "Machine learning supports fraud detection and investment strategies.",
            "Financial institutions use AI to improve customer insights.",
            "AI enhances risk management and financial forecasting.",
            "Overall, AI improves efficiency and accuracy in financial services."
        ],
        "education": [
            "Artificial intelligence is transforming education through personalized learning.",
            "AI helps analyze student performance and learning patterns.",
            "Machine learning enables adaptive learning platforms.",
            "Educational institutions use AI to improve teaching effectiveness.",
            "AI supports digital classrooms and interactive learning.",
            "Overall, AI enhances learning experiences and educational outcomes."
        ],
        "general": [
            "Artificial intelligence is transforming multiple industries.",
            "AI supports data analysis and decision making.",
            "Machine learning enables intelligent systems.",
            "AI improves efficiency across various domains.",
            "AI enhances innovation and productivity.",
            "Overall, AI drives automation and progress."
        ]
    }

    # Choose only the number of sentences requested
    paragraph = " ".join(outputs[domain][:sentences])
    return paragraph

# -----------------------------
# Optional: Word-by-word generation (unused now but can be activated)
# -----------------------------
def generate_text_wordwise(seed_text, max_words=25, temperature=0.7):
    max_len = 10
    text = seed_text.lower()
    generated_words = []

    for _ in range(max_words):
        token_list = tokenizer.texts_to_sequences([text])[0]
        token_list = pad_sequences([token_list], maxlen=max_len-1, padding="pre")
        preds = model.predict(token_list, verbose=0)[0]
        predicted_idx = sample_with_temperature(preds, temperature)
        word = tokenizer.index_word.get(predicted_idx, "")
        if word:
            generated_words.append(word)
            text += " " + word

    return clean_text(" ".join(generated_words)).capitalize() + "."

