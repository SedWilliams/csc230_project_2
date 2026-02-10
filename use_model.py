import numpy as np
import tensorflow as tf

import data
import get_new_test_case
import save_and_load_model as save_or_load

"""
Model Training

'model.fit(...)' performs the training with our model

X_train is our features array [size, rooms, parking?]

'output_normalizer' was defined above. It gives us our
normalized Y values (prices)
"""
def train_model(model, data):
    # Normalize the Y data manually before training
    output_normalizer = tf.keras.layers.Normalization(axis=None)
    output_normalizer.adapt(data.Y_train)
    Y_train_normalized = output_normalizer(data.Y_train)

    print("Starting training...")
    model.fit(
        data.X_train,
        Y_train_normalized,
        epochs=100,
        verbose=1,
    )
    print("Training finished.")

"""
This takes the normalized predicted price 
back to the original scale (millions || thousands)
"""
def denormalize_price_output(predicted_price):
    #get the normalizers that were used to normalize the output data (prices)
    output_normalizer = tf.keras.layers.Normalization(axis=None)
    output_normalizer.adapt(data.Y_train)

    """
    I have no clue about the actual math details here, I just let tensorflow handle it.

    these lines do the "denormalization" of the predicted price
        and return the predicted price with the original scale (millions || thousands)
    """
    variance = output_normalizer.variance.numpy()
    mean = output_normalizer.mean.numpy()
    normalized_predicted_price = predicted_price * np.sqrt(variance) + mean
    return normalized_predicted_price


"""
This function calls for user input on a house listing, and
    then gives it to the model to predict the price of the house.
    It then returns the predicted price and the actual market price for the new user entered case.
"""
def predict_price():
    # get the model
    model = save_or_load.load_model("house_price_model.keras")

    #collect input from user about the new house listing
    new_house_features, actual_market_price = get_new_test_case.get()

    #use the model to predict the price of the new house listing
    predicted_price = model.predict(new_house_features)
    
    #remap the model predicted price back to the original scale (millions || thousands)
    denormalized_predicted_price = denormalize_price_output(predicted_price)
    
    #return the predicted price and the actual market price for the new user entered case
    return denormalized_predicted_price[0][0], actual_market_price

