import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

csv_file_path = 'C:\Users\MEPI\OneDrive - Lebanese American University\Desktop\Computer science\car data.csv'  
missing_values = car_data.isnull().sum()
car_data_encoded = pd.get_dummies(car_data, columns=['Fuel_Type', 'Selling_type', 'Transmission'], drop_first=True)

# Create a new feature 'Car_Age'
car_data_encoded['Car_Age'] = 2024 - car_data_encoded['Year']  # Assuming the current year is 2024
car_data_encoded.drop(['Year'], axis=1, inplace=True)
car_data_encoded.drop(['Car_Name'], axis=1, inplace=True)

# Split the data into features (X) and target variable (y)
X = car_data_encoded.drop('Selling_Price', axis=1)
y = car_data_encoded['Selling_Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Building
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predicting the Test set results
y_pred = model.predict(X_test)

# Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output results
print(f'Missing Values:\n{missing_values}')
print(f'Mean Absolute Error: {mae}')
print(f'R² Score: {r2}')
