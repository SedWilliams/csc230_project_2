# csc230\_project\_2

### MODULES
* [README](#readme)
* [data.py](data.py)
    * Defines the training dataset used by the model.
    * Functions: *(none)*
    * Exposes input data as `X_train` and target data as `y_train`.
* [main.py](main.py)
    * Runs the predictor and prints a buy/overpriced decision.
    * Functions: *(none)*
* [get\_new\_test\_case.py](get_new_test_case.py)
    * Prompts the user for a new house listing to evaluate.
    * Functions: `get()`
* [house\_price\_model.keras](house_price_model.keras)
    * Saved trained Keras model (loaded for inference).
    * These are the trained model parameters, no need to edit.
* [model\_architecture.py](model_architecture.py)
    * Builds, trains, and saves the house price model into \*.keras file.
    * Functions: *(none)*
* [save\_and\_load\_model.py](save_and_load_model.py)
    * Saves and loads Keras models from disk.
    * Functions: `save_model()`, `load_model()`
* [use\_model.py](use_model.py)
    * Trains the model and predicts/denormalizes prices for new inputs.
    * Functions: `train_model()`, `denormalize_price_output()`, `predict_price()`

