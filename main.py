import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
import numpy as np
import pickle
from joblib import dump, load

def file_exists(filepath):
    return os.path.exists(filepath)

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df['Space'] = pd.to_numeric(df['Space'].str.replace(',', ''), errors='coerce')
    df['Price'] = pd.to_numeric(df['Price'].str.replace(',', ''), errors='coerce')
    df.dropna(inplace=True)
    return df

def split_and_save_data(df):
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    test_df, val_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    train_df.to_csv('train.csv', index=False)
    test_df.to_csv('test.csv', index=False)
    val_df.to_csv('val.csv', index=False)

def train_model(X_train, y_train):
    preprocessor = ColumnTransformer(transformers=[
        ('num', MinMaxScaler(), ['Space']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Area', 'City', 'Neighborhood', 'Property type'])
    ])
    
    model = make_pipeline(preprocessor, XGBRegressor(n_estimators=500, learning_rate=0.05, objective='reg:squarederror', random_state=42, reg_lambda=1))
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X, y_actual):
    y_pred = model.predict(X)
    mae = mean_absolute_error(y_actual, y_pred)
    mse = mean_squared_error(y_actual, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_actual, y_pred)
    return mae, mse, rmse, r2

if __name__ == "__main__":
    if not (file_exists('train.csv') and file_exists('test.csv') and file_exists('val.csv')):
        df = load_and_preprocess_data('KSA Real Estate - Dataset.csv')
        split_and_save_data(df)
    
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    val_df = pd.read_csv('val.csv')
    
    X_train = train_df.drop(['Price'], axis=1)
    y_train = train_df['Price']
    X_test = test_df.drop(['Price'], axis=1)
    y_test = test_df['Price']
    X_val = val_df.drop(['Price'], axis=1)
    y_val = val_df['Price']
    
    model = train_model(X_train, y_train)
    
    # Save the trained model and its preprocessor
    with open('real_estate_price_prediction_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    dump(model.named_steps['columntransformer'], 'preprocessor.joblib') # Save the preprocessor
    
    print("Model and preprocessor saved.")
    
    # Evaluate on Training Set
    train_mae, train_mse, train_rmse, train_r2 = evaluate_model(model, X_train, y_train)
    print("Training Set Evaluation:")
    print(f"Mean Absolute Error (MAE): {train_mae}")
    print(f"Mean Squared Error (MSE): {train_mse}")
    print(f"Root Mean Squared Error (RMSE): {train_rmse}")
    print(f"R-squared (R²) Score: {train_r2}\n")

    # Evaluate on Test Set
    test_mae, test_mse, test_rmse, test_r2 = evaluate_model(model, X_test, y_test)
    print("Test Set Evaluation:")
    print(f"Mean Absolute Error (MAE): {test_mae}")
    print(f"Mean Squared Error (MSE): {test_mse}")
    print(f"Root Mean Squared Error (RMSE): {test_rmse}")
    print(f"R-squared (R²) Score: {test_r2}\n")
    
    # Evaluate on Validation Set
    val_mae, val_mse, val_rmse, val_r2 = evaluate_model(model, X_val, y_val)
    print("Validation Set Evaluation:")
    print(f"Mean Absolute Error (MAE): {val_mae}")
    print(f"Mean Squared Error (MSE): {val_mse}")
    print(f"Root Mean Squared Error (RMSE): {val_rmse}")
    print(f"R-squared (R²) Score: {val_r2}")

    unseen_df = pd.read_csv('unseen.csv')

    # Preprocess unseen data (Note: The target column 'Price' should not be in unseen_df)
    X_unseen = unseen_df.drop(columns=['Transaction reference number', 'Date', 'Property classification'], errors='ignore')  # Adjust based on your CSV

    # Ensure the preprocessor and model are loaded 
    model = load('real_estate_price_prediction_model.pkl')

    # Predict on the unseen data
    predicted_prices = model.predict(X_unseen)

    # Print or save the predictions
    print("Predicted Prices for Unseen Data:")
    for i, price in enumerate(predicted_prices):
        print(f"Sample {i+1}: Predicted Price = {price}")
    
    actual_prices = np.array([
    800000, 450000, 680000, 800000, 450000, 680000, 1500000, 750000,
    1200000, 2500000, 1800000, 900000, 3000000, 2200000, 500000])  # This should be the actual prices array

    # Calculate performance metrics
    mae = mean_absolute_error(actual_prices, predicted_prices)
    mse = mean_squared_error(actual_prices, predicted_prices)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual_prices, predicted_prices)

    # Print performance metrics
    print(f"\nMean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R-squared (R²) Score: {r2}")

    # Print predicted prices and actual prices
    for i, price in enumerate(predicted_prices):
        print(f"Sample {i+1}: Predicted Price = {price}, Actual Price = {actual_prices[i]}")
