import pandas as pd
import numpy as np
from joblib import load
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the model and preprocessor
preprocessor = load('preprocessor.joblib')
model = load('real_estate_price_prediction_model.pkl')

# Load the unseen data CSV
unseen_df = pd.read_csv('unseen.csv')

# Preprocess the unseen data
# Assuming the unseen data does not include the 'Price' column as it's unseen
X_unseen_processed = preprocessor.transform(unseen_df)

# Make predictions on the unseen data
predicted_prices = model.predict(X_unseen_processed)

# Assuming we have actual prices for evaluation (you can remove this part if you don't have actual prices)
actual_prices = np.array([
    800000, 450000, 680000, 800000, 450000, 680000, 1500000, 750000,
    1200000, 2500000, 1800000, 900000, 3000000, 2200000, 500000
])  # This should be the actual prices array

# Evaluate the model predictions
mae = mean_absolute_error(actual_prices, predicted_prices)
mse = mean_squared_error(actual_prices, predicted_prices)
rmse = np.sqrt(mse)
r2 = r2_score(actual_prices, predicted_prices)

# Print the metrics
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R²) Score: {r2}")

# Print predicted prices
for i, price in enumerate(predicted_prices):
    print(f"Sample {i+1}: Predicted Price = {price}")

# Save the predicted prices alongside the actual for comparison
comparison_df = pd.DataFrame({
    'Actual Price': actual_prices,
    'Predicted Price': predicted_prices
})
comparison_df.to_csv('predicted_vs_actual.csv', index=False)
print("Predicted vs Actual prices saved to 'predicted_vs_actual.csv'")
