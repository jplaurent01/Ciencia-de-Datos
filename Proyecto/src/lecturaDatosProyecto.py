import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

class datosProyecto: # Clase que se encarga de leer datos del proyecto
    def __init__(self, path):# Constructor clase, recibe como parametro ruta de datos
        # Ruta del archivo a leer y variable que almacena circuitos que causan 80% fallas
        self.path, self.circuitos_bajo_80 = path, None
        # Se utiliza un generador para no sobre cargar memopria del programa y se lee archi en trozos de 1000 lineas
        self.dataFrame =  pd.concat((chunk for chunk in pd.read_csv(self.path, sep=';', header=None, chunksize=1000, engine="c", skiprows=1)),ignore_index=True)
        # Se le asignan al data frame los siguientes nombres de las columnas
        self.dataFrame.columns = ("Interrupción",  "Elemento",  "Fecha Salida",  "Fecha Entrada", "Causa", "Minutos Fuera", "Sistema", "Hora de Salida", "Hora de Entrada", "Nivel", "Kva", "Clientes", "Circuito", "Nombre Circuito")
        

    def display_Pareto(self): # Obtencion del pareto
        # Cuento la cantidad de veces que se repite un circuito dentro de la columna Circuito (Frecuencia de fallas)
        conteo_fallas = self.dataFrame['Nombre Circuito'].value_counts().reset_index() # Creo un dataframe auxiliar para ello
        conteo_fallas.columns = ('Circuito', 'Conteo')  # El nuevo dataframe tiene las columnas de 'Circuito', 'Conteo'

        # Ordeno los valores por frecuencia de mayor a menor
        conteo_fallas = conteo_fallas.sort_values(by='Conteo', ascending=False)
        
        # Calculo la acumulación porcentual de las frecuencias
        conteo_fallas['Acumulado %'] = conteo_fallas['Conteo'].cumsum()/conteo_fallas['Conteo'].sum()*100

        # Determino los circuitos que causan 80 % de las fallas, los guardo en una lista
        self.circuitos_bajo_80 = conteo_fallas[conteo_fallas['Acumulado %'] <= 80]['Circuito'].tolist()

        # Definicion colores pareto y tamño de linea
        color1, color2, line_size = 'steelblue','red', 4

        # Creación del gráfico de barras
        fig, ax = plt.subplots(figsize=(10, 6))  # Definir el tamaño del gráfico
        ax.bar(conteo_fallas['Circuito'], conteo_fallas['Conteo'], color=color1)

        # Se agrega la línea de porcentaje acumulado a la gráfica
        ax2 = ax.twinx()
        ax2.plot(conteo_fallas['Circuito'], conteo_fallas['Acumulado %'], color=color2, marker="D", ms=line_size)
        ax2.yaxis.set_major_formatter(PercentFormatter())

        # Se especifican colores para los ejes
        ax.tick_params(axis='y', colors=color1)
        ax2.tick_params(axis='y', colors=color2)

        # Rotación de las etiquetas del eje X a 90 grados para hacerlas verticales
        ax.set_xticklabels(conteo_fallas['Circuito'], rotation=90, ha='center')  # 90 grados de rotación para las etiquetas

        # Título y etiquetas de los ejes
        ax.set_xlabel('Circuito')
        ax.set_ylabel('Conteo de Fallas', color=color1)
        ax2.set_ylabel('Acumulado (%)', color=color2)
        plt.title('Gráfico de Pareto de Fallas por Circuito')

        # Mostrar el gráfico
        plt.tight_layout()
        plt.show()
      
    def Circuitos_a_Analizar(self):
        print(self.circuitos_bajo_80)
        print("Cantidad total de circuitos:", len(self.circuitos_bajo_80))

        # Filtrar el DataFrame para que solo contenga los circuitos a analizar
        df_filtrado = self.dataFrame[self.dataFrame['Nombre Circuito'].isin(self.circuitos_bajo_80)]

        # Guardar el DataFrame filtrado en un nuevo archivo Excel
        df_filtrado.to_excel(r'..\data_output\Circuitos_a_Analizar_2.xlsx', index=False)

        print("El archivo ha sido reducido y guardado como 'Circuitos_a_Analizar.xlsx'.")

    
    
