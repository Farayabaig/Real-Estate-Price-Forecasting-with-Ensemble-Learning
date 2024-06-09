import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
from joblib import dump, load

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df['Space'] = pd.to_numeric(df['Space'].str.replace(',', ''), errors='coerce')
    df['Price'] = pd.to_numeric(df['Price'].str.replace(',', ''), errors='coerce')
    df.dropna(inplace=True)
    return df

def train_model(X_train, y_train):
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), ['Space']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['Area', 'City', 'Neighborhood', 'Property type']),
        ]
    )

    xgb_model = XGBRegressor(n_estimators=500, learning_rate=0.05, objective='reg:squarederror', random_state=42, reg_lambda=1)
    linear_model = LinearRegression()

    ensemble_model = VotingRegressor(
        estimators=[
            ('xgb', xgb_model),
            ('lr', linear_model)
        ],
        weights=[0.7, 0.3]
    )

    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', ensemble_model)])
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_model(model, X, y_actual):
    y_pred = model.predict(X)
    mae = mean_absolute_error(y_actual, y_pred)
    mse = mean_squared_error(y_actual, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_actual, y_pred)
    return mae, mse, rmse, r2

if __name__ == "__main__":
    df = load_and_preprocess_data('KSA Real Estate - Dataset.csv')
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    test_df, val_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    X_train, y_train = train_df.drop(['Price'], axis=1), train_df['Price']
    X_test, y_test = test_df.drop(['Price'], axis=1), test_df['Price']
    X_val, y_val = val_df.drop(['Price'], axis=1), val_df['Price']

    model = train_model(X_train, y_train)
    pickle.dump(model, open('real_estate_ensemble_model.pkl', 'wb'))
    dump(model.named_steps['preprocessor'], 'preprocessor.joblib')

    # Evaluation
    for name, X, y in [("Training", X_train, y_train), ("Test", X_test, y_test), ("Validation", X_val, y_val)]:
        mae, mse, rmse, r2 = evaluate_model(model, X, y)
        print(f"{name} Set Evaluation:")
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")
        print(f"R-squared (R²) Score: {r2}\n")

    # Unseen data prediction
    unseen_df = pd.read_csv('unseen.csv')
    X_unseen = unseen_df.drop(columns=['Transaction reference number', 'Date', 'Property classification'], errors='ignore')
    predicted_prices = model.predict(X_unseen)
    actual_prices = np.array([800000, 450000, 680000, 800000, 450000, 680000, 1500000, 750000, 1200000, 2500000, 1800000, 900000, 3000000, 2200000, 500000])

    # Unseen data evaluation
    mae, mse, rmse, r2 = mean_absolute_error(actual_prices, predicted_prices), mean_squared_error(actual_prices, predicted_prices), np.sqrt(mse), r2_score(actual_prices, predicted_prices)
    print("Unseen Data Evaluation:")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R-squared (R²) Score: {r2}")

    for i, price in enumerate(predicted_prices):
        print(f"Sample {i+1}: Predicted Price = {price}, Actual Price = {actual_prices[i]}")
