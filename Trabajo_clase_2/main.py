from os import path, scandir
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
        #Datos de operacion normal
        normal_data_frames = (pd.concat((chunk.iloc[:, :self.channels] for chunk in pd.read_csv(path.join(self.data_folder, file_name), sep='\t', header=None, chunksize=1000, engine="c")),ignore_index=True) for file_name in file_names if file_name != "2004.02.19.06.22.39")
        #Operacion de falla
        falla_data_frame =  pd.concat((chunk.iloc[:, :self.channels] for chunk in pd.read_csv(path.join(self.data_folder, "2004.02.19.06.22.39"), sep='\t', header=None, chunksize=1000, engine="c")),ignore_index=True)
        return pd.concat(normal_data_frames, ignore_index=True), falla_data_frame # Dataframe para operacion normal y operacion con fallo

    #El siguiente metodo aplica un Análisis de Componente Principal en N componentes principaples del conjunto de datos obtenidos
    def reduce_dimensionality(self, data, N, info):
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
        
        # Variancia explicada por cada componente principal
        #print("Almacena las direcciones de máxima varianza (los componentes principales): ", pca.components_)
        #print("Varianza explicada por cada componente principal:", pca.explained_variance_ratio_)
        print(info)
        print("Aproximacion con datos originales (%): ", sum(pca.explained_variance_ratio_)*100)
        # Muestra la proporción de varianza explicada por cada uno de los componentes principales.
        # `pca.explained_variance_ratio_` es un array que indica cuánto de la varianza total en los datos originales es capturada por cada componente.
        # Esto es útil para evaluar cuánta información se retiene en la reducción de dimensionalidad.


# Cargo los datos del set #2 
set_2 = ObtenerDatos(4, './4.+Bearings/4. Bearings/IMS/2nd_test/2nd_test' ) # Se especifica la cantidad de columnas y ruta del set #2
data_2_normal, data_2_falla = set_2.load_data_from_files()  # Se cargan los datos del set #2 en un data frame 
set_2.reduce_dimensionality(data_2_normal, 2, "Datos operacion nromal") # Se aplica PCA para datos de operacion normal
set_2.reduce_dimensionality(data_2_falla, 2, "Datos operacion falla") # Se aplica PCA para datos con falla


