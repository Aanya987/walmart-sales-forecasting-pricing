"""
train_model.py
Trains and evaluates the Walmart sales forecasting model.
Run: python train_model.py
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_and_clean_data(path='Walmart.csv'):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df = df.sort_values(['Store', 'Date']).reset_index(drop=True)
    return df


def engineer_features(df):
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['IsThanksgiving'] = df['Date'].dt.strftime('%Y-%m-%d').isin(
        ['2010-11-26', '2011-11-25']
    ).astype(int)
    df['Sales_Lag1'] = df.groupby('Store')['Weekly_Sales'].shift(1)
    df['Sales_RollingMean4'] = (
        df.groupby('Store')['Weekly_Sales'].shift(1).rolling(4).mean()
    )
    return df.dropna().reset_index(drop=True)


def train_test_split_by_date(df, split_date='2012-04-01'):
    split_date = pd.Timestamp(split_date)
    train = df[df['Date'] < split_date]
    test = df[df['Date'] >= split_date]
    return train, test


def evaluate(y_true, y_pred, model_name=""):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"\n{model_name} Results:")
    print(f"MAE:  {mae:,.0f}")
    print(f"RMSE: {rmse:,.0f}")
    print(f"MAPE: {mape:.2f}%")
    return mae, rmse, mape


def main():
    FEATURES = ['Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI',
                'Unemployment', 'Month', 'WeekOfYear', 'IsThanksgiving',
                'Sales_Lag1', 'Sales_RollingMean4']
    TARGET = 'Weekly_Sales'

    print("Loading and cleaning data...")
    df = load_and_clean_data()

    print("Engineering features...")
    df = engineer_features(df)

    train, test = train_test_split_by_date(df)
    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    print("\nTraining Linear Regression (baseline)...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    evaluate(y_test, lr_model.predict(X_test), "Linear Regression")

    print("\nTraining Random Forest...")
    rf_model = RandomForestRegressor(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    preds_rf = rf_model.predict(X_test)
    evaluate(y_test, preds_rf, "Random Forest")

    # Feature importance
    importances = pd.Series(
        rf_model.feature_importances_, index=FEATURES
    ).sort_values(ascending=False)
    print("\nTop features:\n", importances.head(5))

    # Save results for dashboard
    results = test.copy()
    results['Predicted_Sales'] = preds_rf
    results.to_csv('results.csv', index=False)
    print("\nSaved results.csv for dashboard use.")


if __name__ == "__main__":
    main()
