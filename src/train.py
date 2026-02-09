# Import necessary libraries and modules needed
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer  # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
import numpy as np
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
import matplotlib.pyplot as plt
from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import pickle

# Load dataset
df = pd.read_csv("WELFake_Dataset.csv")

# Check for missing values
df.head()
df.isnull().sum()  # sum of missing values

# Drop unnecessary columns
df.drop("Unnamed: 0", axis=1, inplace=True)

# Fill in missing values
df.fillna("", inplace=True)
df.isnull().sum()  # Confirm columns aren't missing again

# Combine useful text columns (feature engineering)
df["content"] = df["title"] + " " + df["text"]  # type: ignore

# Encode labels
X = df["content"].values
y = df["label"].values

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=28
)

# Apply tokenizer and padding
tokenizer = Tokenizer(num_words=50000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)  # Encode each text input

train_sequences = tokenizer.texts_to_sequences(X_train)
test_sequences = tokenizer.texts_to_sequences(X_test)

max_len = 300  # Maximum sequence length
X_train_pad = pad_sequences(train_sequences, maxlen=max_len, padding="post")
X_test_pad = pad_sequences(test_sequences, maxlen=max_len, padding="post")

# Convert labels to an array
y_train = np.array(y_train)
y_test = np.array(y_test)

# Build the LSTM Bi-directional model
vocab_size = 50000
embedding_dim = 128
max_len = 300

model = Sequential(
    [
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        Bidirectional(LSTM(128, return_sequences=False)),
        Dropout(0.5),
        Dense(64, activation="relu"),
        Dropout(0.4),
        Dense(1, activation="sigmoid"),  # binary classification
    ]
)

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

# Add EarlyStopping to prevent overfitting
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=2,  # Stop if val_loss doesn't improve for 2 epochs
    restore_best_weights=True,
)

# Train the model with EarlyStopping
history = model.fit(
    X_train_pad,
    y_train,
    batch_size=64,
    epochs=10,  # Increased epochs; EarlyStopping will stop training early if needed
    validation_split=0.2,
    callbacks=[early_stop],
)

# Plot training & validation loss
plt.figure(figsize=(15, 7))
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Plot training & validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.show()

# Evaluate tthe model's performance
loss, acc = model.evaluate(X_test_pad, y_test)
print(f"Accuracy: {acc:.3f} \n Loss: {loss:.4f}")

# Saving the model
model.save("fake_news_model.keras")

# Save the tokenizer for later use in the API
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# CSV Files to test the model
true = pd.read_csv("True.csv")
fake = pd.read_csv("Fake.csv")

true["content"] = true["title"] + ":" + true["text"]
fake["content"] = fake["title"] + ":" + fake["text"]
