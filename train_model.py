import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from joblib import dump

# Load the dataset
file_path = 'used_cars.csv'  
data = pd.read_csv(file_path)

# Clean and preprocess the data
data['price'] = data['price'].replace('[\$,]', '', regex=True).astype(float)
data['car_age'] = 2025 - data['model_year']  
data = data[['model', 'car_age', 'price']].dropna()

# Convert categorical feature 'model' to numerical labels
data['model'] = pd.factorize(data['model'])[0]

# Features and target
X = data[['model', 'car_age']]
y = data['price']

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model (using Linear Regression as requested)
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model
dump(model, 'car_price_predictor.pkl')
print("Model training completed and saved as 'car_price_predictor.joblib'.")
