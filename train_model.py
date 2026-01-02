import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import pickle

# Load data
import os

texts = []
for file in os.listdir("datasets"):
    with open(f"datasets/{file}", encoding="utf-8") as f:
        texts.append(f.read().lower())

text = "\n".join(texts)


# Tokenizer (small vocab)
tokenizer = Tokenizer(num_words=3000, oov_token="<OOV>")
tokenizer.fit_on_texts([text])

sequences = []
for line in text.split("\n"):
    tokens = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(tokens)):
        sequences.append(tokens[:i+1])

max_len = 10
sequences = pad_sequences(sequences, maxlen=max_len, padding="pre")

X = sequences[:, :-1]
y = tf.keras.utils.to_categorical(sequences[:, -1], num_classes=3000)

# Model (LOW RAM)
model = Sequential([
    Embedding(3000, 64, input_length=max_len-1),
    LSTM(64),
    Dense(3000, activation="softmax")
])

model.compile(loss="categorical_crossentropy", optimizer="adam")

model.fit(X, y, epochs=25, batch_size=8)

model.save("text_gen_model.h5")

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("Model trained & saved")

