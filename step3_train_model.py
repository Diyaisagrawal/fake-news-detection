import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

X_train_pad = np.load("X_train.npy")
X_test_pad = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

VOCAB_SIZE = 20000   # same as tokenizer
EMBED_DIM = 100
MAX_LEN = 300

# Build the Bi-LSTM model
model = Sequential([
    Embedding(VOCAB_SIZE, EMBED_DIM, input_length=MAX_LEN),
    Bidirectional(LSTM(64)),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.compile(
    loss="binary_crossentropy",
    optimizer=Adam(learning_rate=0.001),
    metrics=["accuracy"]
)

model.summary()

# Train the model
history = model.fit(
    X_train_pad,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1
)

# Evaluation on test data
test_loss, test_acc = model.evaluate(X_test_pad, y_test)
print("Test Accuracy:", test_acc)

# Detailled metrics
y_pred = (model.predict(X_test_pad) > 0.5).astype("int32")

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Save the model
model.save("fake_news_bilstm_model.h5")
