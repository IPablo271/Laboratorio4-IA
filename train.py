import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from LogisticRegression import LogisticRegression





#Varivales a utilizar para saber si un paciente sufrira de paro cardiaco
#Age
#Diabetes


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



clf = LogisticRegression(X_train,y_train,X_test,y_test,lr=0.01)
clf.fit()
y_pred = clf.predict()

print("Accurcy del modelo: "+str(clf.score()))

clf.cross_val()

plt.plot(y_test, label="Valores reales")

# crear una gráfica de línea para los valores predichos
plt.plot(y_pred, label="Valores predichos")




# agregar etiquetas y título
plt.xlabel("Índice")
plt.ylabel("Valores")
plt.title("Comparación entre valores reales y predichos")

# agregar una leyenda
plt.legend()

# mostrar la gráfica
plt.show()