import tensorflow as tf
import numpy as np

import data


# Normalize the input data, map it between 0 and 1
input_normalizer = tf.keras.layers.Normalization(axis=-1)
input_normalizer.adapt(data.X_train)

# Normalize the output data, map it between 0 and 1
output_normalizer = tf.keras.layers.Normalization(axis=None)
output_normalizer.adapt(data.Y_train)

# Define the model architecture
model = tf.keras.Sequential([
    input_normalizer,

    # 2 layers, each with 64 neurons and ReLU activation
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
   
    # Output layer, one neuron
    tf.keras.layers.Dense(1)
])

# 'Compile' the model, use Adam optimizer and mean squared error loss function
model.compile(
    optimizer='Adam',
    loss='mean_squared_error'
)

# Train the model on our data (100 rotations)
print("Starting training...")
model.fit(
    data.X_train,
    output_normalizer(data.Y_train),
    epochs=100,
)
print("Training finished.")

'''
# Single test case to see what the model thinks of a new house
new_house_features = np.array([[2000, 3, 1]])
actual_market_price = 500000
'''

size = input("Enter the size of the house in square feet: ")
bedrooms = input("Enter the number of bedrooms: ")
parking = input("Does the house have good parking?: [1=\"Yes\", 0=\"No\"] ")
new_house_features = np.array([[float(size), float(bedrooms), float(parking)]])

actual_market_price = float(input("Enter the actual market price of the house: "))

# predicted_price = model.predict(new_house_features)[0][0]
predicted_price = model.predict(new_house_features)

# Convert the normalized predicted price back to the original scale (millions || thousands)
variance = output_normalizer.variance.numpy()
mean = output_normalizer.mean.numpy()
normalized_predicted_price = predicted_price * np.sqrt(variance) + mean

print(f"Predicted Price: ${normalized_predicted_price[0][0]:.2f}")
print(f"Listed at: ${actual_market_price}")

# logic to decide whether to buy or not based on the predicted price compared to the actual market price
threshold = normalized_predicted_price * 1.02
if actual_market_price <= threshold:
    print("DECISION: YES (Buy)")
else:
    print("DECISION: NO (Overpriced)")


