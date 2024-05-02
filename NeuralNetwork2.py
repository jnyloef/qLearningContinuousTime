import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Generate training data (exponential function)
X_train = np.linspace(0, 2*np.pi, 100).reshape(-1, 1)
y_train = np.sin(X_train)

# Define neural network architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(20, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=5000, verbose=0)

# Test the model
X_test = np.linspace(0, 1, 100).reshape(-1, 1)
y_pred = model.predict(X_test)

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(X_train, y_train, label='True Exponential Function')
plt.plot(X_test, y_pred, label='Predicted Function', linestyle='--', color='red')
plt.title('Approximation of Exponential Function by Neural Network')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.show()