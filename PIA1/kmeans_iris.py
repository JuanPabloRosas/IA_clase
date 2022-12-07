import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
import seaborn as sns

#Agrupamiento
#datos
df=pd.read_csv("C:\\Users\\pablo\\Desktop\\iris_data.csv", header=None)
print("*** CONJUNTO DE DATOS ***")
print(df.head())

#NOMBRES
my_file = open("C:\\Users\\pablo\\Desktop\\iris_names.txt", "r")
data = my_file.read()
data_into_list = data.replace('\n', ',').split(",")
my_file.close()
#NOMBRES


X = StandardScaler().fit_transform(df)    # estandariz los datos
pca = PCA(n_components=2) 
pca.fit(X)
X_new=pca.transform(X)
print()
print("*** COMPONENTES PRINCIPALES ***")
print(X_new[:5])

clusters_num = 3
#Creacion del modelo
print()
print("*** AGRUPAMIENTO ***")
results = []
for j in range(0,5000):
   kmeans = KMeans(n_clusters=clusters_num, init="k-means++")
   clusters=kmeans.fit_predict(X_new)
   print(kmeans.cluster_centers_)

   labels = kmeans.labels_
   print(kmeans.labels_)

   e_setosa = 0
   e_versicolor = 0
   e_virginica = 0
   for i in range(len(labels)):
      if(data_into_list[i] == 'Iris-setosa'):
         if(labels[i] != 0):
            e_setosa = e_setosa + 1
      if(data_into_list[i] == 'Iris-versicolor'):
         if(labels[i] != 1):
            e_versicolor = e_versicolor + 1
      if(data_into_list[i] == 'Iris-virginica'):
         if(labels[i] != 2):
            e_virginica = e_virginica + 1


   print('# Errores setosa:', e_setosa)
   print('# Errores versicolor:', e_versicolor)
   print('# Errores virginica:', e_virginica)
   results.append([e_setosa, ' setosa'])
   results.append([e_versicolor, ' versicolor'])
   results.append([e_virginica, ' virginica'])


"""
#Cantidad de elementos por grupo
counters=[0 for i in range(clusters_num)]


for i in range(len(labels)):
    counters[labels[i]]+=1


print()
print("*** ELEMENTOS POR GRUPO ***")
print(counters)

#Grafica en 2D
pca_data = pd.DataFrame(X_new,columns=['Componente 1','Componente 2']) 
pca_data['Grupo'] = kmeans.labels_
print(pca_data)
sns.scatterplot(x="Componente 1",y="Componente 2",hue="Grupo",data=pca_data,legend="full")
plt.show()
"""
r = pd.DataFrame(results)
r.columns = ['y', 'type']
print(r)
sns.boxplot( data=r, x = 'type', y = 'y')
plt.show()