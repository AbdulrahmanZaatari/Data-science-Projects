import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
data = pd.read_csv(r'C:\Users\MEPI\Downloads\Advertising.csv')
print(data.head())
data.drop(['Unnamed: 0'], axis=1, inplace=True)
print(data.isnull().sum())
sns.pairplot(data)
plt.show()
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True)
plt.show()
X = data.drop(['Sales'], axis=1)
y = data['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
new_data = [[150, 20, 15]]
predicted_sales = model.predict(new_data)
print(f"Predicted Sales: {predicted_sales[0]}")


