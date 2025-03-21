import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
columns = ["feature" + str(i) for i in range(561)]  # Adjust if needed
columns.append("activity")

# Replace with actual file paths
train_data = pd.read_csv("train_data.csv", names=columns)
test_data = pd.read_csv("test_data.csv", names=columns)

# Extract features and labels
X_train = train_data.iloc[:, :-1].values  # Features
y_train = train_data.iloc[:, -1].values   # Labels

X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape to fit CNN-LSTM (samples, time_steps, features)
X_train = X_train.reshape(-1, 100, 9)  # Adjust based on dataset
X_test = X_test.reshape(-1, 100, 9)

# Convert labels to one-hot encoding
num_classes = len(set(y_train))
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Define model
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(100, 9)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    LSTM(50, return_sequences=False),
    Dense(50, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save the model
model.save("cnn_lstm_har_model.h5")

print("Model saved successfully.")
