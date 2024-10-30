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
        # Obtiene la lista de archivos en el directorio especificado
        #file_paths = sorted(glob.glob(os.path.join(data_folder, file_pattern)))

        #Se utilizan los dos ultimos archivos por que el programa tarda mucho
        file_paths = ['./4.+Bearings/4. Bearings/IMS/1st_test/1st_test\\2003.11.25.23.29.56', './4.+Bearings/4. Bearings/IMS/1st_test/1st_test\\2003.11.25.23.39.56']
        
        # Usa ThreadPoolExecutor para leer los archivos en paralelo
        with ThreadPoolExecutor() as executor:
            data_frames = list(executor.map(lambda file_path: self.read_file(file_path, channels), file_paths))
        
        # Concatena todos los DataFrames en uno solo
        concatenated_data = pd.concat(data_frames, ignore_index=True)
        print(concatenated_data)
        return concatenated_data
    
    def plot_data(self, concatenated_data, channels):
        # Convertir la columna Timestamp a un formato de tiempo adecuado si es necesario
        concatenated_data['Timestamp'] = pd.to_datetime(concatenated_data['Timestamp'], errors='coerce')
        
        # Gráfico de un canal (por ejemplo, Ch1) para visualizar la tendencia
        plt.figure(figsize=(14, 7))
        plt.plot(concatenated_data['Timestamp'], concatenated_data['Ch1'], label='Ch1')
        plt.xlabel('Timestamp')
        plt.ylabel('Amplitude')
        plt.title('Channel 1 Vibration Over Time')
        plt.legend()
        plt.show()

        # Subgráficos para comparar múltiples canales
        fig, axs = plt.subplots(channels, 1, figsize=(14, channels*3), sharex=True)
        for i in range(1, channels + 1):
            channel = f'Ch{i}'
            axs[i-1].plot(concatenated_data['Timestamp'], concatenated_data[channel], label=channel)
            axs[i-1].set_ylabel('Amplitude')
            axs[i-1].legend()
        
        axs[-1].set_xlabel('Timestamp')
        plt.suptitle('Vibration Over Time for All Channels')
        plt.show()

# Ejemplo de uso:
datos = ModeloOrdenReducido()
# Reemplaza 'path_to_set_1' con el directorio real que contiene los archivos para el Set 1
data_set_1 = datos.read_and_concatenate_files('./4.+Bearings/4. Bearings/IMS/1st_test/1st_test', channels=8)
# data_set_2 = datos.read_and_concatenate_files('./4.+Bearings/4. Bearings/IMS/2nd_test/2nd_test', channels=4)
# data_set_3 = datos.read_and_concatenate_files('./4.+Bearings/4. Bearings/IMS/3rd_test/4th_test/txt', channels=4)

# Concatenar opcionalmente todos los conjuntos de datos en un solo DataFrame
# all_data = pd.concat([data_set_1, data_set_2, data_set_3], ignore_index=True)

# Mostrar un resumen de los datos concatenados
# print(all_data.info())
# print(all_data.head())
