import unittest
import pandas as pd

# Import relevant functions
from model import train_linear_regression_model, predict_house_price_from_input, load_model

class TestModelFunctions(unittest.TestCase):

    def setUp(self):
        # Load preprocessed data and train model
        self.data = pd.read_csv('/Users/Paul/Desktop/data_preprocessed.csv')
        self.trained_model = train_linear_regression_model(self.data)
    
    def test_train_linear_regression_model(self):
        # Check if model is trained
        self.assertIsNotNone(self.trained_model)

    def test_predict_house_price_from_input(self):
        # Predict price using sample data and check if result is float
        sample_input = self.data.iloc[0].drop('price').values.tolist()
        price = predict_house_price_from_input(self.trained_model, sample_input)
        self.assertIsInstance(price, float)

    def test_load_model(self):
        # Check if model can be loaded from file
        model = load_model("trained_model.pkl")
        self.assertIsNotNone(model)

if __name__ == "__main__":
    unittest.main()
