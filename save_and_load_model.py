import tensorflow as tf
import numpy as np


def save_model(model, file_path):
    model.save(file_path)

def load_model(file_path):
    return tf.keras.models.load_model(file_path)



