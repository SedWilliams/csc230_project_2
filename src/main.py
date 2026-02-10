from src.model import use_model

#get the price prediction from the model
price_prediction, actual_market_price = use_model.predict_price()

#print the predicted price and the actual market price for the new user entered case
print(f"Predicted Price: ${int(price_prediction)}")
print(f"Listed at: ${int(actual_market_price)}")

threshold = price_prediction * 1.02 # returns value that's 2% above the model predicted price

# if market price is greater than 2% above the predicted price, it's overpriced. 
#       Otherwise, it's a good deal.
#       
#       Inform the user accordingly.
if actual_market_price <= threshold:
    print("DECISION: YES (Buy)")
else:
    print("DECISION: NO (Overpriced)")

