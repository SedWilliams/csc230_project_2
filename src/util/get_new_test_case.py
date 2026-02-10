import numpy as np

def get():
    size = input("Enter the size of the house in square feet: ")
    bedrooms = input("Enter the number of bedrooms: ")
    parking = input("Does the house have good parking?: [1=\"Yes\", 0=\"No\"] ")
    new_house_features = np.array([[float(size), float(bedrooms), float(parking)]])

    actual_market_price = float(input("Enter the actual market price of the house: "))

    return new_house_features, actual_market_price

