import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Load the Iris dataset
iris = pd.read_csv("iris.csv")

# Explore the dataset
print("Dataset Information:")
print(iris.info())
print("\nSummary Statistics:")
print(iris.describe())
print("\nTarget Labels:", iris["Species"].unique())

# Visualize the data using Plotly Express
fig = px.scatter(iris, x="SepalWidthCm", y="SepalLengthCm", color="Species", title="Sepal Width vs. Sepal Length")
fig.show()

# Prepare the data for modeling
x = iris.drop("Species", axis=1)
y = iris["Species"]

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Create and train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)  # Adjust the number of neighbors as needed
knn.fit(x_train, y_train)

# Make predictions for the test set
y_pred = knn.predict(x_test)

# Evaluate the KNN classifier
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# Create a heatmap of the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=iris["Species"].unique(), yticklabels=iris["Species"].unique())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Make a prediction for a new data point
x_new = np.array([[4, 2.9, 1, 0.2]])
new_prediction = knn.predict(x_new)
print("\nPrediction for New Data Point:", new_prediction[0])
