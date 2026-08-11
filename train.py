import numpy as np
import pandas as pd
import pickle
import sklearn.datasets

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(3)

# ------------------------------------------------------------
# Load Data (built into sklearn - no CSV needed)
# ------------------------------------------------------------
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()

data_frame = pd.DataFrame(
    breast_cancer_dataset.data,
    columns=breast_cancer_dataset.feature_names
)
data_frame['label'] = breast_cancer_dataset.target

X = data_frame.drop(columns='label')
Y = data_frame['label']

# ------------------------------------------------------------
# Train-Test Split
# ------------------------------------------------------------
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2
)

# ------------------------------------------------------------
# Standardize the Data
# ------------------------------------------------------------
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# ------------------------------------------------------------
# Build & Train the Neural Network
# ------------------------------------------------------------
model = keras.Sequential([
    keras.layers.Input(shape=(30,)),
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(2, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train_std, Y_train, validation_split=0.1, epochs=10)

loss, accuracy = model.evaluate(X_test_std, Y_test)
print("Test Accuracy:", accuracy)

# ------------------------------------------------------------
# Save Model and Scaler for the Streamlit App
# ------------------------------------------------------------
model.save("breast_cancer_model.keras")
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("Saved breast_cancer_model.keras and scaler.pkl")
