import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from joblib import dump
file_path = 'used_cars.csv'  
data = pd.read_csv(file_path)
data['price'] = data['price'].replace('[\$,]', '', regex=True).astype(float)
data['car_age'] = 2025 - data['model_year']  
data = data[['model', 'car_age', 'price']].dropna()
data['model'] = pd.factorize(data['model'])[0]
X = data[['model', 'car_age']]
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = linear_model.predict(X_test)
dump(model, 'car_price_predictor.pkl')
print("Model training completed and saved as 'car_price_predictor.joblib'.")
