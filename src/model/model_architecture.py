import tensorflow as tf

from pathlib import Path

from src.data import data
from src.model import use_model


"""
NORMALIZATION
    * changes the data to be scaled similarly
    * makes it easier for the model to work with
#normalize the input data, map it between 0 and 1
input_normalizer = tf.keras.layers.Normalization(axis=-1)
input_normalizer.adapt(data.X_train)

#normalize the output data, map it between 0 and 1
output_normalizer = tf.keras.layers.Normalization(axis=None)
output_normalizer.adapt(data.Y_train)
"""

"""
Model Architecture
"""
# Define and adapt normalization layers separately
input_normalizer = tf.keras.layers.Normalization(axis=-1)
input_normalizer.adapt(data.X_train)

# Note: Normalizing the output layer is uncommon in this manner, but preserving intent.
output_normalizer = tf.keras.layers.Normalization(axis=None)
output_normalizer.adapt(data.Y_train)

# define the model architecture
new_model = tf.keras.Sequential([
    input_normalizer,

    # 2 layers, each with 64 neurons and ReLU activation
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
   
    # Output layer, one neuron
    tf.keras.layers.Dense(1),
    # output_normalizer # Typically you don't normalize the output of the final prediction in the model itself like this
])

#'Compile' the model, use Adam optimizer and mean squared error loss function
new_model.compile(
    optimizer='Adam',
    loss='mean_squared_error'
)

use_model.train_model(new_model, data)

src_dir = Path(__file__).resolve().parents[1]
model_path = src_dir / "stored_model" / "house_price_model.keras"
new_model.save(str(model_path))


