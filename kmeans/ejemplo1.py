from numpy import random, array, linalg, newaxis, argmin
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

"""
Fuente: https://www.geeksforgeeks.org/k-means-clustering-introduction/
Se nos proporciona un conjunto de datos de elementos, con ciertas características y valore
para estas características (como un vector). La tarea es clasificar esos elementos en grupos.
Para conseguirlo utilizaremos el algoritmo K-means, un algoritmo de aprendizaje no supervisado.
'K' en el nombre del algoritmo representa la cantidad de grupos/clústeres en los que queremos
clasificar nuestros elementos.
El algoritmo clasificará los elementos en k grupos o grupos de similitud. Para calcular esa similitud,
usaremos la distancia euclidiana como medida.
El algoritmo funciona de la siguiente manera:  
Primero, inicializamos aleatoriamente k puntos, llamados medias o centroides de grupo.
Categorizamos cada elemento según su media más cercana y actualizamos las coordenadas de la media,
que son los promedios de los elementos categorizados en ese grupo hasta el momento.
Repetimos el proceso para un número determinado de iteraciones y al final tenemos nuestros clústeres.
Los "puntos" mencionados anteriormente se denominan medias porque son los valores medios de los elementos
categorizados en ellos.
"""
class Kmean:
    def __init__(self):
        # Inicializo atributos
        self.k, self.clusters = 3, None  # Establecer el número de clusters (k) y inicializar clusters
        # Crear un conjunto de datos con 'k' centros
        self.X, self.y = make_blobs(n_samples=500, n_features=2, centers=self.k, random_state=23)
    
    def plot_makeblobs(self):
        # Función para graficar los blobs generados
        plt.figure(figsize=(8, 6))  # Tamaño de la figura
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, cmap='viridis', s=50)  # Graficar puntos
        plt.title('Distribución de blobs')  # Título del gráfico
        plt.xlabel('Característica 1')  # Etiqueta del eje X
        plt.ylabel('Característica 2')  # Etiqueta del eje Y
        plt.show()  # Mostrar el gráfico
    
    def init_random_centroids(self):
        # Inicializar los centroides de manera aleatoria
        random.seed(23)  # Semilla para reproducibilidad
        # Crear centroides aleatorios en el rango [-2, 2]
        self.clusters = {idx: {'center': 2 * (2 * random.random((self.X.shape[1],)) - 1), 'points': []} for idx in range(self.k)}
    
    def plot_random_initialize_center(self):
        # Graficar los puntos y los centroides aleatorios
        plt.scatter(self.X[:, 0], self.X[:, 1])  # Graficar todos los puntos
        plt.grid(True)  # Activar la cuadrícula
        # Obtener los centros de los clusters
        centers = array([self.clusters[i]['center'] for i in self.clusters])
        plt.scatter(centers[:, 0], centers[:, 1], marker='*', c='red')  # Graficar los centroides
        plt.show()  # Mostrar el gráfico
        
    def assign_clusters(self):
        # Asignar cada punto al cluster más cercano
        centers = array([self.clusters[i]['center'] for i in range(self.k)])  # Obtener los centros
        dist = linalg.norm(self.X[:, newaxis] - centers, axis=2)  # Calcular distancias entre puntos y centros
        closest_clusters = argmin(dist, axis=1)  # Encontrar el índice del cluster más cercano
        # Re-inicializar los clusters para guardar los puntos
        self.clusters = {i: {'center': self.clusters[i]['center'], 'points': []} for i in range(self.k)}
        # Asignar puntos a los clusters
        [self.clusters[cluster_idx]['points'].append(self.X[idx]) for idx, cluster_idx in enumerate(closest_clusters)]
            
    def update_clusters(self):
        # Actualizar la posición de los centroides
        for i in range(self.k):
            points = array(self.clusters[i]['points'])  # Obtener puntos del cluster
            # Calcular el nuevo centro como la media de los puntos, o mantener el viejo si no hay puntos
            self.clusters[i]['center'] = points.mean(axis=0) if len(points) > 0 else self.clusters[i]['center']
            self.clusters[i]['points'].clear()  # Limpiar los puntos del cluster para la próxima iteración
    
    def pred_cluster(self):
        # Predecir el cluster para cada punto
        distances = linalg.norm(self.X[:, newaxis] - array([cluster['center'] for cluster in self.clusters.values()]), axis=2)
        self.pred = argmin(distances, axis=1)  # Obtener el índice del cluster más cercano
    
    def plot_data_with_predicted_cluster_center(self):
        # Graficar los puntos de datos con sus centros de cluster previstos
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.pred, s=30, cmap='viridis')  # Graficar los puntos con color según su cluster
        centers = array([self.clusters[i]['center'] for i in range(self.k)])  # Obtener los centros de los clusters
        plt.scatter(centers[:, 0], centers[:, 1], marker='^', c='red', s=100)  # Graficar los centros
        plt.show()  # Mostrar el gráfico

########################################### Ejemplo #1 #########################################################

# Crear una instancia de la clase Kmean
kmean_model = Kmean()
# Inicializar los centroides aleatorios
kmean_model.init_random_centroids()
# Asignar, actualizar y predecir el centro del cluster
kmean_model.assign_clusters()
kmean_model.update_clusters()
kmean_model.pred_cluster()
# Trazar los puntos de datos con su centro de grupo previsto
kmean_model.plot_data_with_predicted_cluster_center()