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
        file_paths = ['./4.+Bearings/4. Bearings/IMS/1st_test/1st_test\\2003.11.25.23.29.56', 
                      './4.+Bearings/4. Bearings/IMS/1st_test/1st_test\\2003.11.25.23.39.56']
        
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
