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

class Kmean:
    def __init__(self):
        # Inicializo los atributos de la clase
        # plt se usa para gráficos, el dataframe se carga desde un archivo CSV
        self.plt = plt
        self.dataframe = pd.read_csv(r"analisis.csv")  # Carga los datos del CSV
        plt.rcParams['figure.figsize'] = (16, 9)  # Configura el tamaño de las figuras de los gráficos
        plt.style.use('ggplot')  # Aplica el estilo ggplot a los gráficos
        # Selecciono las columnas 'op', 'ex' y 'ag' como las variables que se usarán para KMeans
        self.X = np.array(self.dataframe[["op", "ex", "ag"]])  
        # Selecciono la columna 'categoria' como la variable de etiquetas
        self.y = np.array(self.dataframe['categoria'])  
        # Inicializo las variables para KMeans: número de clusters (k), modelo kmeans y centroides
        self.k, self.kmeans, self.centroids = 0, None, None

    # Método para graficar los datos en 3D, donde los colores representan diferentes categorías
    def plot_3D(self):
        # Crear una nueva figura para el gráfico 3D
        fig = self.plt.figure()
        ax = Axes3D(fig)
        # Defino los colores a usar para las diferentes categorías
        colores = ['blue', 'red', 'green', 'cyan', 'yellow', 'orange', 'black', 'pink', 'brown', 'purple']
        # Asigno los colores a cada fila de datos según su categoría en self.y
        asignar = [colores[row] for row in self.y]
        # Grafico los datos en el espacio 3D
        ax.scatter(self.X[:, 0], self.X[:, 1], self.X[:, 2], c=asignar, s=60)
        # Etiquetas para los ejes
        ax.set_xlabel('Eje X')
        ax.set_ylabel('Eje Y')
        ax.set_zlabel('Eje Z')
        # Mostrar la gráfica
        self.plt.show()

    # Método para calcular y graficar la curva del codo para seleccionar el mejor número de clusters
    def get_elbow(self):
        # Defino un rango de número de clusters a probar, de 1 a 19
        Nc = range(1, 20)  
        # Creo una lista de modelos KMeans para cada número de clusters en Nc
        kmeans = [KMeans(n_clusters=i) for i in Nc]
        # Calculo la puntuación (inercia) de cada modelo y la almaceno en 'score'
        score = [kmeans[i].fit(self.X).score(self.X) for i in range(len(kmeans))]  
        # Crear una nueva figura para la curva del codo
        self.plt.figure()
        # Graficar el número de clusters contra la puntuación para obtener la curva del codo
        self.plt.plot(Nc, score, marker='o')
        # Etiquetas para los ejes y título del gráfico
        self.plt.xlabel('Number of Clusters')
        self.plt.ylabel('Score')
        self.plt.title('Elbow Curve')
        # Mostrar la gráfica
        self.plt.show()

    # Método para graficar los datos en 3D y los clusters generados por KMeans
    def plot_3D_KMeans(self):
        # Defino k = 5 clusters para el algoritmo KMeans
        self.k = 5
        # Ejecuto el algoritmo KMeans y obtengo las etiquetas y centroides
        self.kmeans = KMeans(n_clusters=self.k).fit(self.X)
        self.centroids = self.kmeans.cluster_centers_  # Obtengo los centroides de los clusters
        # Predigo a qué cluster pertenece cada punto de datos
        labels = self.kmeans.predict(self.X)
        # Obtengo los centroides de los clusters
        C = self.kmeans.cluster_centers_
        # Defino los colores para los diferentes clusters
        colores = ['red', 'green', 'blue', 'cyan', 'yellow']
        # Asigno colores a cada punto según el cluster al que pertenece
        asignar = [colores[row] for row in labels]
        # Crear una nueva figura para el gráfico 3D
        fig = plt.figure()
        ax = Axes3D(fig)
        # Grafico los puntos de datos con colores según su cluster
        ax.scatter(self.X[:, 0], self.X[:, 1], self.X[:, 2], c=asignar, s=60)
        # Grafico los centroides con una estrella (*) para diferenciarlos
        ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c=colores, s=1000)
        # Mostrar la gráfica
        self.plt.show()

# Crear la instancia de la clase Kmean
kmean_model = Kmean()
# Mostrar la gráfica 3D de los datos categorizados
kmean_model.plot_3D()
# Mostrar la curva del codo para determinar el mejor número de clusters
kmean_model.get_elbow()
# Mostrar la gráfica 3D de los clusters generados por KMeans con los centroides
kmean_model.plot_3D_KMeans()




