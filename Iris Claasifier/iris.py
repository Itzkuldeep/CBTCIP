import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

iris_data = pd.read_csv("C:\Users\kulde\Coding\CBTCIP\Iris Claasifier\Iris.csv")

iris_data.head(10)

print("Target Labels", iris_data["Species"].unique())

fig = px.scatter(iris_data, x="SepalWidthCm", y="SepalLengthCm", color="Species")
fig.show()

x = iris_data.drop("Species", axis=1)
y = iris_data["Species"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=0)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)

x_new = np.array([[5, 2.9, 1, 0.2,1.4]])
prediction = knn.predict(x_new)
print("Prediction: {}".format(prediction))