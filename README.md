# csc230\_project\_2

### MODULES
* [README](#readme)
* [src/data/data.py](src/data/data.py)
    * Defines the training dataset used by the model.
    * Functions: *(none)*
    * Exposes input data as `X_train` and target data as `y_train`.
* [src/main.py](src/main.py)
    * Runs the predictor and prints a buy/overpriced decision.
    * Functions: *(none)*
* [src/util/get\_new\_test\_case.py](src/util/get_new_test_case.py)
    * Prompts the user for a new house listing to evaluate.
    * Functions: `get()`
* [src/stored\_model/house\_price\_model.keras](src/stored_model/house_price_model.keras)
    * Saved trained Keras model (loaded for inference).
    * These are the trained model parameters, no need to edit.
* [src/model/model\_architecture.py](src/model/model_architecture.py)
    * Builds, trains, and saves the house price model into \*.keras file.
    * Functions: *(none)*
* [src/model/save\_and\_load\_model.py](src/model/save_and_load_model.py)
    * Saves and loads Keras models from disk.
    * Functions: `save_model()`, `load_model()`
* [src/model/use\_model.py](src/model/use_model.py)
    * Trains the model and predicts/denormalizes prices for new inputs.
    * Functions: `train_model()`, `denormalize_price_output()`, `predict_price()`


