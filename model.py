import tensorflow as tf
import numpy as np

import data

mockX_train = np.array([
    [1500, 3, 1],
    [2500, 4, 2],
    [800,  1, 0],
    [3200, 5, 2],
    [1200, 2, 1]
], dtype='float32')

mockY_train = np.array([400, 650, 250, 800, 350], dtype='float32')

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(data.X_train)

model = tf.keras.Sequential([
    normalizer,

    # 2 layers, each with 64 neurons and ReLU activation
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
   
    # Output layer, one neuron
    tf.keras.layers.Dense(1) 
])

model.compile(
    optimizer='adam',
    loss='mean_squared_error'
)

print("Starting training...")
history = model.fit(
    data.X_train, 
    data.Y_train,
    epochs=3,
)
print("Training finished.")


# Single test case to see what the model thinks of a new house
new_house_features = np.array([[2000, 3, 2]])
actual_market_price = 500000

predicted_price = model.predict(new_house_features)[0][0]


print(f"Predicted Price: ${predicted_price}")
print(f"Listed at: ${actual_market_price}")

threshold = predicted_price * 1.02
if actual_market_price <= threshold:
    print("DECISION: YES (Buy)")
else:
    print("DECISION: NO (Overpriced)")


