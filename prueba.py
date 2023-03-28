import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LogisticRegression import LogisticRegression


df = pd.read_csv("framingham.csv")
variables = ["age","diabetes"]
variables2 = ["TenYearCHD"]
X = df[variables].values
y = df[variables2].values
y = y.ravel()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


lr = LogisticRegression(X_train, y_train, X_test, y_test, lr=0.001, n_iters=1000, cv=6)

accuracy = lr.fit()
print("Accuracy:", accuracy)

test_accuracy = lr.score()
print("Test accuracy:", test_accuracy)
