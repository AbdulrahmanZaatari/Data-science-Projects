import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

iris = datasets.load_iris()
X = iris.data
y = iris.target

iris_df = pd.DataFrame (X, columns = iris.feature_names)
print(iris_df.head())
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=43)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

predictions = knn.predict(X_test)
#print(confusion_matrix(y_test, predictions))
#print(classification_report(y_test, predictions))

def get_user_input():
    print("\nEnter the measurements of the iris flower.")
    print("Type 'exit' to stop the program.")

    sepal_length = input("Enter sepal length (cm): ")
    if sepal_length.lower() == 'exit':
        return None
    sepal_width = input("Enter sepal width (cm): ")
    if sepal_width.lower() == 'exit':
        return None
    petal_length = input("Enter petal length (cm): ")
    if petal_length.lower() == 'exit':
        return None
    petal_width = input("Enter petal width (cm): ")
    if petal_width.lower() == 'exit':
        return None

    return [float(sepal_length), float(sepal_width), float(petal_length), float(petal_width)]

while True:
    try:
        user_input = get_user_input()
        if user_input is None:
            print("Exitting program.")
            break

        new_data = np.array([user_input])
        prediction = knn.predict(new_data)
        print("Predicted species:", iris.target_names[prediction[0]])

    except ValueError:
        print("Invalid input. Please enter valid numbers.")
    except Exception as e:
        print(f"An error occurred: {e}")
