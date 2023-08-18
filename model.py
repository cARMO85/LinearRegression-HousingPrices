import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import os
import warnings

# Suppress scikit-learn warnings
warnings.filterwarnings(action="ignore", module="sklearn")

# Load data
data_preprocessed = pd.read_csv('/Users/Paul/Desktop/data_preprocessed.csv')

def train_linear_regression_model(data):
    # Split data and train linear regression model
    X = data.drop('price', axis=1)
    y = data['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    return model

def save_model(model, filename="trained_model.pkl"):
    # Save model to file
    joblib.dump(model, filename)

def load_model(filename="trained_model.pkl"):
    # Load model from file
    if os.path.exists(filename):
        return joblib.load(filename)
    else:
        raise FileNotFoundError(f"{filename} not found!")

def predict_house_price_from_input(model, input_variables):
    # Predict house price with given model
    if len(input_variables) != len(model.coef_):
        raise ValueError("Mismatch in input variables.")
    return model.predict([input_variables])[0]

# Train and save model
trained_model = train_linear_regression_model(data_preprocessed)
save_model(trained_model)

# Sample prediction
sample_input = data_preprocessed.iloc[0].drop('price').values.tolist()
predicted_price = predict_house_price_from_input(trained_model, sample_input)
print(predicted_price)
