# data_processing.py

import pandas as pd

def load_data(filepath):
    # Load dataset from CSV
    return pd.read_csv("/Users/paul/Desktop/Housing.csv")

def preprocess_data(data):
    # Convert categorical columns to numerical using one-hot encoding
    categorical_columns = [
        'mainroad', 'guestroom', 'basement', 'hotwaterheating',
        'airconditioning', 'parking', 'prefarea', 'furnishingstatus'
    ]
    return pd.get_dummies(data, columns=categorical_columns, drop_first=True)

def get_mean_house_price(data):
    # Calculate and return mean house price (rounded to 2 decimal places)
    return round(data['price'].mean(), 2)

# Save preprocessed data to CSV
data_preprocessed.to_csv('/Users/Paul/Desktop/data_preprocessed.csv', index=False)

# Sample usage to test the module's functionality
data = load_data("Housing.csv")
data_preprocessed = preprocess_data(data)
print(f"Mean House Price: {get_mean_house_price(data_preprocessed)}")
