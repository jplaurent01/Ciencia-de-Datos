from os import path, scandir
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
    
class ObtenerDatos:
    def __init__(self, channels, path):
        self.data_folder, self.channels = path, channels

    # Función para cargar archivos de datos y concatenarlos en un DataFrame
    def load_data_from_files(self):
        all_data_frames = []
        # Captura la lista de archivos que terminan en .txt o .asc
        file_names = (f for f in scandir(self.data_folder))
        
        for file_name in file_names:
            file_path = path.join(self.data_folder, file_name.name)
            # Lee el archivo en chunks y selecciona las columnas necesarias
            file_data = pd.concat(
                (chunk.iloc[:, :self.channels] for chunk in pd.read_csv(file_path, sep='\t', header=None, chunksize=1000)),
                ignore_index=True
            )
            all_data_frames.append(file_data)
        
        return pd.concat(all_data_frames, ignore_index=True)



        #all_data = np.vstack(all_data)  # Devuelve todos los datos como una sola matriz de numpy
        #return all_data
        # Usando comprensión de listas para cargar datos
        #for file_name in file_names:
        #    file_path = os.path.join(self.data_folder, file_name)
            # Cargar los datos y seleccionar las columnas necesarias
        #    data_chunk = pd.read_csv(file_path, sep='\t', header=None).iloc[:, :self.channels].values
        #    yield data_chunk  # Usar yield para devolver un chunk de datos

# Cargar los datos del Set #1 como ejemplo
#set_1 = ObtenerDatos(8, './4.+Bearings/4. Bearings/IMS/1st_test/1st_test' )
#data_1 = set_1.load_data_from_files()
#print(data_1)
set_2 = ObtenerDatos(4, './4.+Bearings/4. Bearings/IMS/2nd_test/2nd_test' )
data_2 = set_2.load_data_from_files()
#set_3 = ObtenerDatos(4, './4.+Bearings/4. Bearings/IMS/2nd_test/2nd_test' )
#data_3 = set_2.load_data_from_files()
#print(data_3)

# Escalado de datos
scaler = StandardScaler() # Estandarice las características eliminando la media y escalando a la varianza unitaria.
data_scaled = scaler.fit_transform(data_2) # Ajusta el transformador a X e y con los parámetros opcionales fit_params y devuelve una versión transformada de X.

# Aplicación de PCA
pca = PCA(n_components=2)  # Reducir a 2 componentes principales para visualización
data_pca = pca.fit_transform(data_scaled)

# Visualización de los resultados
plt.figure(figsize=(10, 7))
plt.scatter(data_pca[:, 0], data_pca[:, 1], alpha=0.5)
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.title("Análisis de Componentes Principales (PCA) - Datos de Vibración")
plt.show()

# Variancia explicada por cada componente principal
print("Varianza explicada por cada componente principal:", pca.explained_variance_ratio_)

"""
class otener_datos():
    def __init__(self, channels, path):
        self.data_folder, self.channels = path, channels

    # Función para cargar archivos de datos
    def load_data_from_files(self):
        # Aquí se captura la lista de archivos que terminan en .txt o .asc (o el formato que necesites)
        file_names = (f for f in scandir(self.data_folder))
        for file_name in file_names:
            file_path = path.join(self.data_folder, file_name.name)
            all_data = (chunk.iloc[:, :self.channels].values for chunk in pd.read_csv(file_path, sep='\t', header=None, chunksize=1000))

        return all_data

"""

"""
import os
import pandas as pd
import glob
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

class ModeloOrdenReducido():
    
    def read_file(self, file_path, channels):
        # Lee cada archivo y añade la columna de Timestamp extraída del nombre del archivo
        df = pd.read_csv(file_path, delimiter='\t', header=None, names=[f'Ch{i+1}' for i in range(channels)])
        timestamp = os.path.basename(file_path).split('.')[0]
        df['Timestamp'] = timestamp
        return df
    
    def read_and_concatenate_files(self, data_folder, channels, file_pattern="*"):
        # Usa una lista de archivos limitada para reducir el tiempo de procesamiento
        
        file_paths = ['./4.+Bearings/4. Bearings/IMS/1st_test/1st_test\\2003.11.25.23.39.56']
        
        #'./4.+Bearings/4. Bearings/IMS/1st_test/1st_test\\2003.11.25.23.29.56',
        # Usa ThreadPoolExecutor para leer los archivos en paralelo
        with ThreadPoolExecutor() as executor:
            data_frames = list(executor.map(lambda file_path: self.read_file(file_path, channels), file_paths))
        
        # Concatena todos los DataFrames en uno solo
        concatenated_data = pd.concat(data_frames, ignore_index=True)
        return concatenated_data
    
    def plot_data_points(self, concatenated_data, channels):
        # Gráfico de la cantidad de datos en función de cada canal
        fig, axs = plt.subplots(channels, 1, figsize=(14, channels*3), sharex=True)
        
        # Iterar sobre cada canal y graficar los puntos
        for i in range(1, channels + 1):
            channel = f'Ch{i}'
            axs[i-1].plot(concatenated_data.index, concatenated_data[channel], label=channel, marker='o', markersize=2, linestyle='-', linewidth=0.5)
            axs[i-1].set_ylabel('Amplitude')
            axs[i-1].legend()
        
        axs[-1].set_xlabel('Data Point Index')
        plt.suptitle('Amplitude vs Data Point Index for All Channels')
        plt.show()

# Ejemplo de uso:
datos = ModeloOrdenReducido()
data_set_1 = datos.read_and_concatenate_files('./4.+Bearings/4. Bearings/IMS/1st_test/1st_test', channels=8)
datos.plot_data_points(data_set_1, channels=8)
"""