"""
Fuente: https://www.aprendemachinelearning.com/k-means-en-python-paso-a-paso/
Como ejemplo utilizaremos de entradas un conjunto de datos que obtuve de un proyecto propio,
en el que se analizaban rasgos de la personalidad de usuarios de Twitter.
He filtrado a 140 “famosos” del mundo en diferentes areas: deporte, cantantes, actores, etc.
Basado en una metodología de psicología conocida como “Ocean: The Big Five” tendemos como características
 de entrada:
usuario (el nombre en Twitter)
“op” = Openness to experience  grado de apertura mental a nuevas experiencias, curiosidad, arte
“co” =Conscientiousness  grado de orden, prolijidad, organización
“ex” = Extraversion  grado de timidez, solitario o participación ante el grupo social
“ag” = Agreeableness  grado de empatía con los demás, temperamento
“ne” = Neuroticism,  grado de neuroticismo, nervioso, irritabilidad, seguridad en sí mismo.
Wordcount  Cantidad promedio de palabras usadas en sus tweets
Categoria  Actividad laboral del usuario (actor, cantante, etc.)
Utilizaremos el algoritmo K-means para que agrupe estos usuarios -no por su actividad laboral
- si no, por sus similitudes en la personalidad. Si bien tenemos 8 columnas de entrada, sólo utilizaremos
3 en este ejemplo, de modo que podamos ver en un gráfico tridimensional
-y sus proyecciones a 2D- los grupos resultantes. Pero para casos reales, podemos utilizar todas las
dimensiones que necesitemos. Una de las hipótesis que podríamos tener es: “Todos los cantantes tendrán
personalidad parecida” (y así con cada rubro laboral). Pues veremos si lo probamos, o por el contrario,
los grupos no están relacionados necesariamente con la actividad de estas Celebridades.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from mpl_toolkits.mplot3d import Axes3D

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

class Kmean:
    def __init__(self):
        # Inicializo atributos
        self.plt = plt
        self.dataframe = pd.read_csv(r"analisis.csv")
        plt.rcParams['figure.figsize'] = (16, 9)
        plt.style.use('ggplot')
        self.X = np.array(self.dataframe[["op", "ex", "ag"]])  # Selecciono solamente 3 columnas
        self.y = np.array(self.dataframe['categoria'])
        self.k, self.kmeans, self.centroids  = 0, None, None

    # Gráfica en 3D con colores representando las categorías.
    def plot_3D(self):
        fig = self.plt.figure()
        ax = Axes3D(fig)
        colores = ['blue', 'red', 'green', 'cyan', 'yellow', 'orange', 'black', 'pink', 'brown', 'purple']
        asignar = [colores[row] for row in self.y]
        ax.scatter(self.X[:, 0], self.X[:, 1], self.X[:, 2], c=asignar, s=60)
        ax.set_xlabel('Eje X')
        ax.set_ylabel('Eje Y')
        ax.set_zlabel('Eje Z')
        self.plt.show()  # Mostrar la gráfica

    def get_elbow(self):#Se selecion k = 5
        Nc = range(1, 20)  # Rango de número de clusters
        kmeans = [KMeans(n_clusters=i) for i in Nc]  # Crear lista de modelos KMeans
        score = [kmeans[i].fit(self.X).score(self.X) for i in range(len(kmeans))]  # Calcular la puntuación
        self.plt.figure()  # Crear nueva figura para la curva del codo
        self.plt.plot(Nc, score, marker='o')  # Graficar la puntuación
        self.plt.xlabel('Number of Clusters')
        self.plt.ylabel('Score')
        self.plt.title('Elbow Curve')
        self.plt.show()  # Mostrar la gráfica
    
    #Ahora veremos esto en una gráfica 3D con colores para los grupos y
    #veremos si se diferencian: (las estrellas marcan el centro de cada cluster)
    #Aqui podemos ver que el Algoritmo de K-Means con K=5 ha agrupado a los 140 usuarios Twitter
    #por su personalidad, teniendo en cuenta las 3 dimensiones que utilizamos:
    # Openess, Extraversion y Agreeablenes
    def plot_3D_KMeans(self):
        # Ejecutamos el algoritmo para 5 clusters y obtenemos las etiquetas y los centroids.
        self.k = 5
        self.kmeans = KMeans(n_clusters=self.k).fit(self.X)
        self.centroids = self.kmeans.cluster_centers_
        # Predicting the clusters
        labels = self.kmeans.predict(self.X)
        # Getting the cluster centers
        C = self.kmeans.cluster_centers_
        colores = ['red', 'green', 'blue', 'cyan', 'yellow']
        asignar = [colores[row] for row in labels]
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(self.X[:, 0], self.X[:, 1], self.X[:, 2], c=asignar, s=60)
        ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c=colores, s=1000)
        self.plt.show()  # Mostrar la gráfica al final

# Crear la instancia de la clase Kmean
kmean_model = Kmean()
kmean_model.plot_3D()  # Mostrar la gráfica 3D
kmean_model.get_elbow()  # Mostrar la curva del codo
kmean_model.plot_3D_KMeans()



