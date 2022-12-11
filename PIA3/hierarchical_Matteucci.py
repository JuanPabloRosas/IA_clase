# needed imports
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.vq import whiten
from scipy.cluster.hierarchy import linkage

data = np.array([[0  ,662,877,255,412,996], # BA 0
                 [662,0  ,295,468,268,400], # FI 1
                 [877,295,0  ,754,564,138], # MI 2
                 [255,468,754,0  ,219,869], # NA 3
                 [412,268,564,219,0  ,669], # RM 4
                 [996,400,138,869,669,0]])  # TO 5

# Datos normalizados
data = whiten(data)
# Matriz de distancias
matrix = linkage(data,method='single',metric='euclidean' )

# Dendograma
labels = ['BA','FI','MI','NA','RM','TO']
dn = dendrogram(matrix, labels=labels)
plt.title('Dendrograma')
plt.show()