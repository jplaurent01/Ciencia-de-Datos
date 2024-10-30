from os import path, scandir
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time

# La siguiente clase lee los datos de los sets de datos
class ObtenerDatos:
    #Constructos
    def __init__(self, channels, path):
    # Se debe ingresas como atributos la cantidad de canales del archivo (columnas) y  tambien la ruta donde se encuentra la carpeta con los distintos archivos.
        self.data_folder, self.channels = path, channels

    # Metodo para cargar archivos de datos y concatenarlos en un DataFrame
    def load_data_from_files(self):
        # Generador que almacena direcion en memoria del nombre del primer archivo del set de datos # N
        # Se utiliza un generador para no cargar todos los nombres de los archivos en memoria y así optimizar recuros.
        file_names = (f.name for f in scandir(self.data_folder)) # Almaceno nombres de archivos por carpeta de set de datos.
        # El sigueinte for recorre cada uno de los generadores con los nombres de archivos por set de datos
        # Lee el archivo en chunks y selecciona las columnas necesarias que van desde 0 hasta self.channels, los datos se encuentran separados por espacios (\t)
        # Abro cada uno de los archivos de la carpeta en bloques de 1000 para no sobre cargar memoria de la comutadora, voy de 1000 en 1000 hasta recolectar todos los datos del archivo.
        # Almaceno en all_data_frames los datos de cada uno de los archivos leidos, utilizo un generador para ahorrar memoria.
        all_data_frames = (pd.concat((chunk.iloc[:, :self.channels] for chunk in pd.read_csv(path.join(self.data_folder, file_name), sep='\t', header=None, chunksize=5000, engine="c")),ignore_index=True) for file_name in file_names)
        return pd.concat(all_data_frames, ignore_index=True) # Retorno un dataframe con la concatenacion de los daatos de todos los archivos de la carpeta set datos # N.

    #El siguiente metodo aplica un Análisis de Componente Principal en N componentes principaples del conjunto de datos obtenidos
    def reduce_dimensionality(self, data, N):
        # Escalado de datos
        scaler = StandardScaler() # Estandarice las características eliminando la media y escalando a la varianza unitaria.
        # Crea una instancia del escalador `StandardScaler`, que normaliza los datos para que cada característica tenga una media de 0 y una desviación estándar de 1.
        # Este paso es esencial para el PCA, ya que asegura que cada característica contribuya equitativamente a la variación total.
        data_scaled = scaler.fit_transform(data) # Ajusta el transformador a X e y con los parámetros opcionales fit_params y devuelve una versión transformada de X.
        # `fit_transform` ajusta el escalador a los datos de entrada `data`, calcula la media y la desviación estándar para cada característica,
        # y aplica la transformación de estandarización. El resultado `data_scaled` contiene la versión estandarizada de `data`.

        # Aplicación de PCA
        pca = PCA(n_components=N) 
         # Instancia el modelo PCA para reducir la dimensionalidad de los datos a `N` componentes principales.
        # `n_components=N` especifica el número de componentes principales a conservar, que capturan la mayor parte de la varianza de los datos originales.
        data_pca = pca.fit_transform(data_scaled)
        # `fit_transform` ajusta el modelo PCA a los datos escalados `data_scaled` y transforma los datos, proyectándolos en el espacio reducido de N dimensiones.
        # El resultado `data_pca` contiene los datos en el nuevo espacio de las componentes principales, representado por un array de `N` columnas.

        # Visualización de los resultados
        #plt.figure(figsize=(10, 7))# Configura una figura de 10x7 pulgadas para visualizar los resultados del PCA en un gráfico de dispersión de dos componentes principales.
        #plt.scatter(data_pca[:, 0], data_pca[:, 1], alpha=0.5)
        # Crea un gráfico de dispersión de los datos proyectados en el espacio de las dos primeras componentes principales.
        # `data_pca[:, 0]` y `data_pca[:, 1]` corresponden a los valores de las primeras dos componentes principales.
        # `alpha=0.5` establece transparencia en los puntos para hacer más visibles las zonas densamente pobladas.
        #plt.xlabel("Componente Principal 1") # Etiqueta del eje X
        #plt.ylabel("Componente Principal 2") # Etiqueta del eje Y
        #plt.title("Análisis de Componentes Principales (PCA) - Datos de Vibración") # Titulo
        #plt.show() # Muestra el gráfico de dispersión generado.
        # Variancia explicada por cada componente principal
        print("Varianza explicada por cada componente principal:", pca.explained_variance_ratio_)
        print(sum(pca.explained_variance_ratio_))
        # Muestra la proporción de varianza explicada por cada uno de los componentes principales.
        # `pca.explained_variance_ratio_` es un array que indica cuánto de la varianza total en los datos originales es capturada por cada componente.
        # Esto es útil para evaluar cuánta información se retiene en la reducción de dimensionalidad.

# Cargo los datos del Set #1
#set_1 = ObtenerDatos(8, './4.+Bearings/4. Bearings/IMS/1st_test/1st_test' )
#data_1 = set_1.load_data_from_files()
#set_1.reduce_dimensionality(data_1, 4)
# Cargo los datos del set #2 
set_2 = ObtenerDatos(4, './4.+Bearings/4. Bearings/IMS/2nd_test/2nd_test' ) # Se especifica la cantidad de columnas y ruta del set #2
data_2 = set_2.load_data_from_files()  # Se cargan los datos del set #2 en un data frame 
set_2.reduce_dimensionality(data_2, 3) # Se aplica PCA para los datos del set #2 y para 2 componentes principales
# Cargo los datos del set #3
#set_3 = ObtenerDatos(4, './4.+Bearings/4. Bearings/IMS/3rd_test/3rd_test' )
#data_3 = set_2.load_data_from_files()
#set_3.reduce_dimensionality(data_3, 2)

