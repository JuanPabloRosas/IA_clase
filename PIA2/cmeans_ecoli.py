from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 

colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']

df=pd.read_csv("C:\\Users\\pablo\\Desktop\\ecoli_data.csv", header=None)
print("*** CONJUNTO DE DATOS ***")
print(df.head())

#NOMBRES
my_file = open("C:\\Users\\pablo\\Desktop\\ecoli_names.txt", "r")
data = my_file.read()
data_into_list = data.replace('\n', ',').split(",")
my_file.close()


training_data = df.sample(frac=0.8, random_state=25)
testing_data = df.drop(training_data.index)

X = StandardScaler().fit_transform(training_data)    # estandariz los datos
pca = PCA(n_components=2) 
pca.fit(X)
X=pca.transform(X)

x1 = []
x2 = []
for i in range(len(X)):
   x1.append(X[i][0])
   x2.append(X[i][1])

Y = StandardScaler().fit_transform(testing_data)    # estandariz los datos
pca = PCA(n_components=2) 
pca.fit(Y)
Y=pca.transform(Y)
y1 = []
y2 = []
for i in range(len(Y)):
   y1.append(Y[i][0])
   y2.append(Y[i][1])


# Visualize the training data
fig0, ax0 = plt.subplots()
for label in range(3):
    ax0.plot(x1, x2, '.', color=colors[label])
ax0.set_title('Datos de entrenamiento: 80%')
plt.show()

# Visualize the test data
fig1, ax1 = plt.subplots()
for label in range(3):
    ax1.plot(y1, y2, '.', color=colors[label])
ax1.set_title('Datos de prueba: 20%')
plt.show()

#------------------------------------------------------------
alldata = np.vstack((x1, x2))
cntr, u_orig, _, _, _, _, _ = fuzz.cluster.cmeans(alldata, 3, 2, error=0.005, maxiter=1000)


# Show 3-cluster model
fig2, ax2 = plt.subplots()
ax2.set_title('Modelo entrenado')
for j in range(3):
    ax2.plot(alldata[0, u_orig.argmax(axis=0) == j],alldata[1, u_orig.argmax(axis=0) == j], 'o',label='grupo ' + str(j))
ax2.legend()
plt.show()

#------------------------------------------------------------

alldata2 = np.vstack((y1, y2))

u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict( alldata2, cntr, 2, error=0.005, maxiter=1000)

cluster_membership = np.argmax(u, axis=0)  

fig3, ax3 = plt.subplots()
ax3.set_title('Clasificación de los puntos de prueba')
for j in range(3):
    ax3.plot(alldata2[0, u.argmax(axis=0) == j],alldata2[1, u.argmax(axis=0) == j], 'o',label='grupo ' + str(j))
ax3.legend()

plt.show()