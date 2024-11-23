import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
import numpy as np
from datetime import  timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit



class datosProyecto: # Clase que se encarga de leer datos del proyecto
    def __init__(self, path):# Constructor clase, recibe como parametro ruta de datos
        # Ruta del archivo a leer y variable que almacena circuitos que causan 80% fallas
        self.path, self.circuitos_bajo_80, self.fallas_bajo_80 = path, None, None
        # Se utiliza un generador para no sobre cargar memopria del programa y se lee archi en trozos de 1000 lineas
        self.dataFrame =  pd.concat((chunk for chunk in pd.read_csv(self.path, sep=';', header=None, chunksize=1000, engine="c", skiprows=1)),ignore_index=True)
        # Se le asignan al data frame los siguientes nombres de las columnas
        self.dataFrame.columns = ("Interrupción",  "Elemento",  "Fecha Salida",  "Fecha Entrada", "Causa", "Minutos Fuera", "Sistema", "Hora de Salida", "Hora de Entrada", "Nivel", "Kva", "Clientes", "Circuito", "Nombre Circuito")
        # Se convierte la columna 'Fecha Salida' a tipo fecha con el formato 'día/mes/año'
        self.dataFrame['Fecha Salida'] = pd.to_datetime(self.dataFrame['Fecha Salida'], format='%d/%m/%Y')
        #Convierto las causa a strings
        self.dataFrame["Causa"] = self.dataFrame["Causa"].astype(str)

    # Hacer conteo de las fallas y sacar fallas que representan el 80%
    def display_ParetoDeMasFallas(self):
        # Cuento la cantidad de veces que se repite un circuito dentro de la columna Circuito (Frecuencia de fallas)
        conteo_fallas = self.dataFrame['Causa'].value_counts().reset_index() # Creo un dataframe auxiliar para ello
        conteo_fallas.columns = ('Fallas', 'Conteo')  # El nuevo dataframe tiene las columnas de 'Fallas', 'Conteo'

        # Ordeno los valores por frecuencia de mayor a menor
        conteo_fallas = conteo_fallas.sort_values(by='Conteo', ascending=False)
        
        # Calculo la acumulación porcentual de las frecuencias
        conteo_fallas['Acumulado %'] = conteo_fallas['Conteo'].cumsum()/conteo_fallas['Conteo'].sum()*100

        # Determino los fallas que causan 80 % de las fallas, los guardo en una lista
        self.fallas_bajo_80 = set(conteo_fallas[conteo_fallas['Acumulado %'] <= 80]['Fallas'])

        # Definicion colores pareto y tamño de linea
        color1, color2, line_size = 'steelblue','red', 4

        # Creación del gráfico de barras
        fig, ax = plt.subplots(figsize=(10, 6))  # Definir el tamaño del gráfico
        ax.bar(conteo_fallas['Fallas'], conteo_fallas['Conteo'], color=color1)

        # Se agrega la línea de porcentaje acumulado a la gráfica
        ax2 = ax.twinx()
        ax2.plot(conteo_fallas['Fallas'], conteo_fallas['Acumulado %'], color=color2, marker="D", ms=line_size)
        ax2.yaxis.set_major_formatter(PercentFormatter())

        # Se especifican colores para los ejes
        ax.tick_params(axis='y', colors=color1)
        ax2.tick_params(axis='y', colors=color2)

        # Rotación de las etiquetas del eje X a 90 grados para hacerlas verticales
        ax.set_xticklabels(conteo_fallas['Fallas'], rotation=90, ha='center')  # 90 grados de rotación para las etiquetas

        # Título y etiquetas de los ejes
        ax.set_xlabel('Fallas')
        ax.set_ylabel('Conteo de Fallas', color=color1)
        ax2.set_ylabel('Acumulado (%)', color=color2)
        plt.title('Gráfico de Pareto de Fallas')

        # Mostrar el gráfico
        plt.tight_layout()
        plt.show()
        

    def fallas_a_Analizar(self):
        # Imprime causas que causan el 80 porciento de las fallas
        print(self.fallas_bajo_80)
        # Imprime cantidad de fallas 
        print("Cantidad total de fallas:", len(self.fallas_bajo_80))

        # Crear un diccionario donde el key es la causa y el value es la cantidad de apariciones
        conteo_fallas_80 = {causa: self.dataFrame[self.dataFrame['Causa'] == causa].shape[0] for causa in self.fallas_bajo_80}
        
        # Imprimir el diccionario de fallas que causan el 80% con su respectivo conteo
        print("Conteo de fallas para causas que representan el 80%:")
        print(conteo_fallas_80)

        self.dataFrame = self.dataFrame[self.dataFrame['Causa'].isin(iter(self.fallas_bajo_80))]
        print("Se realizo filtro del pareto dento del dataframe")

        # Guardar el DataFrame filtrado en un nuevo archivo Excel
        #self.dataFrame.to_excel(r'..\data_output\Fallas_a_Analizar.xlsx', index=False)

    def create_features(self, df):
        """
        Create time series features based on time series index.
        """
        df = df.copy()
        df['dayofweek'] = df.index.dayofweek
        df['quarter'] = df.index.quarter
        df['month'] = df.index.month
        df['year'] = df.index.year
        df['dayofyear'] = df.index.dayofyear
        df['dayofmonth'] = df.index.day
        df['weekofyear'] = df.index.isocalendar().week
        target_map = df['cantidad incidentes por dia'].to_dict()
        df['lag1'] = (df.index - pd.Timedelta('365 days')).map(target_map)
        df['lag2'] = (df.index - pd.Timedelta('730 days')).map(target_map)
        df['lag3'] = (df.index - pd.Timedelta('1095 days')).map(target_map)
        return df

    #adding lag featurs to dataset

    def add_lags(self, df):
        target_map = df['cantidad incidentes por dia'].to_dict()
        df['lag1'] = (df.index - pd.Timedelta('365 days')).map(target_map)
        df['lag2'] = (df.index - pd.Timedelta('730 days')).map(target_map)
        df['lag3'] = (df.index - pd.Timedelta('1095 days')).map(target_map)
        return df

    def predecirResultadosCircuito2(self, circuito_de_interes):
        color_pal = sns.color_palette()
        plt.style.use('fivethirtyeight')
        plt.rcParams["axes.formatter.limits"] = (-99, 99)

       # Filtrar el DataFrame de acuerdo al circuito de interés (Causa)
        df_filtrado = self.dataFrame[self.dataFrame["Causa"] == circuito_de_interes]
        
        # Agrupar las fechas por cantidad de incidentes por día
        df_fecha_count = df_filtrado.groupby('Fecha Salida').size().reset_index(name='cantidad incidentes por dia')
        
        # Establecer la columna 'Fecha Salida' como índice para el análisis
        df_fecha_count.set_index('Fecha Salida', inplace=True)
        
        # Dividir los datos en entrenamiento y prueba
        train = df_fecha_count[df_fecha_count.index < pd.to_datetime("01/01/2024", format='%d/%m/%Y')]
        test = df_fecha_count[df_fecha_count.index >= pd.to_datetime("01/01/2024", format='%d/%m/%Y')]
        
        # Asegurar que las fechas estén alineadas
        y_train = train['cantidad incidentes por dia']
        y_test = test['cantidad incidentes por dia']
        
        # Crear y entrenar el modelo SARIMAX
        SARIMAXmodel = SARIMAX(y_train, order=(1, 1, 2), seasonal_order=(3, 2, 3, 12))
        SARIMAXmodel_fit = SARIMAXmodel.fit()
        
        # Realizar la predicción para el período de prueba (test)
        y_pred_3 = SARIMAXmodel_fit.get_forecast(len(test))
        y_pred_3_df = y_pred_3.conf_int(alpha=0.05)  # Intervalos de confianza

        # Predecir valores y redondearlos a enteros
        y_pred_3_df["Predictions"] = SARIMAXmodel_fit.predict(start=y_pred_3_df.index[0], end=y_pred_3_df.index[-1])
        y_pred_3_df["Predictions"] = y_pred_3_df["Predictions"].round().astype(int)
        
        # Ajustar las predicciones al índice de test (fechas de prueba)
        y_pred_3_df.index = test.index
        y_pred_3_out = y_pred_3_df["Predictions"]
        
        # Graficar los resultados
        plt.figure(figsize=(10, 6))
        plt.plot(train.index, train['cantidad incidentes por dia'], color="black", label="Entrenamiento",  marker="D", ms=4)
        plt.plot(test.index, test['cantidad incidentes por dia'], color="red", label="Prueba",  marker="D", ms=4)
        plt.plot(y_pred_3_out, color='blue', label='Predicciones SARIMA',  marker="D", ms=4)
        plt.title('Predicción de Fallas para Circuito {}'.format(circuito_de_interes))
        plt.xlabel('Fecha')
        plt.ylabel('Cantidad de Incidentes por Día')
        plt.xticks(rotation=45)
        plt.legend()
        plt.show()

        # Calcular y mostrar el RMSE para el modelo SARIMAX
        rmse_sarimax = np.sqrt(mean_squared_error(test["cantidad incidentes por dia"].values, y_pred_3_out))
        print("RMSE de SARIMAX: ", rmse_sarimax)

    def predecirResultadosCircuito(self, circuito_de_interes):
        color_pal = sns.color_palette()
        plt.style.use('fivethirtyeight')
        plt.rcParams["axes.formatter.limits"] = (-99, 99)

        # Filtrar las filas donde 'Causa' es igual a '106'
        df_filtrado = self.dataFrame[self.dataFrame["Causa"] == circuito_de_interes]
        df_fecha_count = df_filtrado.groupby('Fecha Salida').size().reset_index(name='cantidad incidentes por dia')

         # Establecer la columna 'Fecha Salida' como índice para el gráfico
        df_fecha_count.set_index('Fecha Salida', inplace=True)

        #df_fecha_count.plot(style='.', figsize=(15, 5), color=color_pal[0], title='Fallas por fecha')
        #plt.show()

        #running machine learning model

        tss = TimeSeriesSplit(n_splits=2, test_size=100, gap=1)
        df_fecha_count = df_fecha_count.sort_index()

        fold = 0
        preds = []
        scores = []
        for train_idx, val_idx in tss.split(df_fecha_count):
            train = df_fecha_count.iloc[train_idx]
            test = df_fecha_count.iloc[val_idx]

            train = self.create_features(train)
            test = self.create_features(test)

            FEATURES = ['dayofyear', 'dayofweek', 'quarter', 'month','year',
                        'lag1','lag2','lag3']
            TARGET = 'cantidad incidentes por dia'

            X_train = train[FEATURES]
            y_train = train[TARGET]

            X_test = test[FEATURES]
            y_test = test[TARGET]

            reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',    
                                n_estimators=1000,
                                early_stopping_rounds=50,
                                objective='reg:squarederror',
                                max_depth=3,
                                learning_rate=0.01)
            reg.fit(X_train, y_train,
                    eval_set=[(X_train, y_train), (X_test, y_test)],
                    verbose=100)

            y_pred = reg.predict(X_test)
            preds.append(y_pred)
            score = np.sqrt(mean_squared_error(y_test, y_pred))
            scores.append(score)


        #printing scores for each fold

        print(f'Score across folds {np.mean(scores):0.4f}')
        print(f'Fold scores:{scores}')

        # retraining model to predict future values

        df_fecha_count = self.create_features(df_fecha_count)

        FEATURES = ['dayofyear', 'dayofweek', 'quarter', 'month', 'year',
                    'lag1','lag2','lag3']
        TARGET = 'cantidad incidentes por dia'

        X_all = df_fecha_count[FEATURES]
        y_all = df_fecha_count[TARGET]

        reg = xgb.XGBRegressor(base_score=0.5,
                            booster='gbtree',    
                            n_estimators=220,
                            objective='reg:squarederror',
                            max_depth=3,
                            learning_rate=0.01)
        reg.fit(X_all, y_all,
                eval_set=[(X_all, y_all)],
                verbose=100)

        # creating future dates for model to predict

        future = pd.date_range('2024-01-01','2024-12-31')
        future_df = pd.DataFrame(index=future)
        future_df['isFuture'] = True
        df_fecha_count['isFuture'] = False
        df_and_future = pd.concat([df_fecha_count, future_df])
        df_and_future = self.create_features(df_and_future)
        df_and_future = self.add_lags(df_and_future)

        future_w_features = df_and_future.query('isFuture').copy()

        #running the predict function

        future_w_features['pred'] = reg.predict(future_w_features[FEATURES])


        #plotting output from model

        future_w_features['pred'].plot(figsize=(10, 5),
                                    color=color_pal[3],
                                    ms=1,
                                    lw=1,
                                    title='2024 Unit Sales Predictions')
        plt.show()

        """
        #creating testing and training splits

        train = df_fecha_count.loc[df_fecha_count.index < '01-01-2024']
        test = df_fecha_count.loc[df_fecha_count.index >= '01-01-2024']

        #plotting the split
        #plotting the split

        fig, ax = plt.subplots(figsize=(15, 5))
        train.plot(ax=ax, label='Training Set', title='Data Train/Test Split')
        test.plot(ax=ax, label='Test Set')
        ax.axvline('01-01-2024', color='black', ls='--')
        ax.legend(['Training Set', 'Test Set'])
        plt.show()
        """



        """  
        # Graficar los datos
        sns.set()
        plt.ylabel('Fallas 106')
        plt.xlabel('Fecha')
        plt.xticks(rotation=45)
        
        # Usamos 'Fecha Salida' como eje x y 'Causa' como eje y
        plt.plot(df_filtrado.index, df_filtrado['Causa'], marker='o')
        
        # Mostrar la gráfica
        plt.show()

        # Dividir los datos en conjunto de entrenamiento y prueba basados en fecha
        train = df_fecha_count[df_fecha_count.index < pd.to_datetime("22/9/2022", format='%d/%m/%Y')]  # Entrenamiento hasta agosto 2024
        test = df_fecha_count[df_fecha_count.index >= pd.to_datetime("22/9/2022", format='%d/%m/%Y')]  # Testeo después de agosto 2024

        # Graficar
        sns.set()
        plt.figure(figsize=(10, 6))  # Opcional: Ajustar el tamaño de la figura
        plt.plot(train.index, train['cantidad incidentes por dia'], color="black", label="Entrenamiento")  # Entrenamiento
        plt.plot(test.index, test['cantidad incidentes por dia'], color="red", label="Prueba")  # Testeo

        # Etiquetas y título
        plt.ylabel('Fallas 106')
        plt.xlabel('Fecha')
        plt.xticks(rotation=45)
        plt.title("Train/Test split for falla 106")
        
        ################### Autoregressive Moving Average (ARMA) Model ################################
        y = train['cantidad incidentes por dia']
        # Definamos nuestro modelo. Para definir un modelo ARMA con la clase SARIMAX, pasamos los parámetros de
        # orden de (1, 0, 1). Alfa corresponde al nivel de significancia de nuestras predicciones. Normalmente,
        # elegimos un alfa = 0,05. Aquí, el algoritmo ARIMA calcula los límites superior e inferior alrededor 
        # de la predicción de modo que haya un cinco por ciento de posibilidades de que el valor real esté fuera
        # de los límites superior e inferior. Esto significa que hay un 95 por ciento de confianza en que el valor 
        # real estará entre los límites superior e inferior de nuestras predicciones.
        ARMAmodel = SARIMAX(y, order = (1, 0, 1))
        #Ajuste del modelo
        ARMAmodel = ARMAmodel.fit()
        # Generar predicion
        y_pred = ARMAmodel.get_forecast(len(test.index))
        y_pred_df = y_pred.conf_int(alpha = 0.05) 
        y_pred_df["Predictions"] = ARMAmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
        y_pred_df.index = test.index
        y_pred_out = y_pred_df["Predictions"]

        plt.plot(y_pred_out, color='green', label = 'Predictions')

        arma_rmse = np.sqrt(mean_squared_error(test["cantidad incidentes por dia"].values, y_pred_df["Predictions"]))
        print("RMSE: ",arma_rmse)

        # ARIMA tiene tres parámetros. El primer parámetro corresponde al retraso (valores pasados),
        # el segundo corresponde a la diferenciación (esto es lo que hace que los datos no estacionarios sean
        # estacionarios) y el último parámetro corresponde al ruido blanco (para modelar eventos de choque).

        ARIMAmodel = ARIMA(y, order = (2, 2, 2))
        ARIMAmodel = ARIMAmodel.fit()

        y_pred_2 = ARIMAmodel.get_forecast(len(test.index))
        y_pred_2_df = y_pred_2.conf_int(alpha = 0.05) 
        y_pred_2_df["Predictions"] = ARIMAmodel.predict(start = y_pred_2_df.index[0], end = y_pred_2_df.index[-1])
        y_pred_2_df.index = test.index
        y_pred_2_out = y_pred_2_df["Predictions"] 
        plt.plot(y_pred_2_out, color='Yellow', label = 'ARIMA Predictions')
        # Mostrar leyenda
        plt.legend()

        # Mostrar el gráfico
        plt.show() 
        print("Fechas del teste: " + ", ".join(test.index.strftime('%d/%m/%Y')))
        print("Fechas del SARIMAX: " + ", ".join(y_pred_out.index.strftime('%d/%m/%Y')))
        print("Fechas del ARIMA: " + ", ".join(y_pred_2_out.index.strftime('%d/%m/%Y')))
        """
        
        

    # Montecarlo estimar fechas con mayor probabilidad de ocurrencia de las fallas
    def MonteCarloPorFechaConMayorProbabilidadOcurrenciaDeFallas(self, num_simulations = 50, forecast_days=30):
        # Filtrar el DataFrame para que solo contenga los circuitos a analizar
        self.dataFrame = self.dataFrame[self.dataFrame['Causa'].isin(iter(self.fallas_bajo_80))]

        # Inicializa un DataFrame para almacenar los resultados de las simulaciones
        resultados_simulacion = []

        # Para cada causa, realizamos una simulación
        for causa, df_causa in self.dataFrame.groupby('Causa'):

            # Contamos las ocurrencias de cada causa para este circuito
            conteo_causas = df_causa['Causa'].value_counts()
            
            # Normalizamos para que sumen a 1 (distribución de probabilidades)
            probabilidad_causas = conteo_causas / conteo_causas.sum()
            
            # Simulamos las fallas usando Monte Carlo
            simulaciones = np.random.choice(probabilidad_causas.index, size=num_simulations, p=probabilidad_causas.values)

            # Obtener la última fecha de falla para esta causa
            fecha_ultima_falla = df_causa['Fecha Salida'].max()
            
            # Simular futuras fallas y predecir su fecha de aparición
            for simulacion in simulaciones:
                # Simula un incremento aleatorio de días para la próxima falla
                dias_hasta_falla = np.random.randint(1, forecast_days)  # Entre 1 y `forecast_days` días
                # Nueva fecha de falla es la fecha de la última falla más los días simulados
                fecha_predicha = fecha_ultima_falla + timedelta(days=dias_hasta_falla)

                # Almacenar los resultados en la lista
                resultados_simulacion.append({
                    'Causa': simulacion,
                    'Fecha Predicha': fecha_predicha.strftime('%d/%m/%Y')
                })

        # Convertir los resultados a un DataFrame
        df_resultados = pd.DataFrame(resultados_simulacion)
        # Agrupar por 'Causa' y eliminar las fechas predichas repetidas
        df_resultados_unicos = df_resultados.groupby('Causa')['Fecha Predicha'].unique().reset_index()
        # Arreglo la informacion para desplegarla verticalemente
        self.df_resultados_exploded = df_resultados_unicos.explode('Fecha Predicha').reset_index(drop=True)
        # Este bloque de codigo traduce el significado del codigo de la falla
        self.traducirCodigoCausa()
        print(self.df_resultados_exploded)
        # Contar la cantidad de veces que aparece cada causa en la columna 'Causa'
        conteo_causas = self.df_resultados_exploded['Causa Nombre'].value_counts().reset_index()

        # Renombrar las columnas para que sea más claro
        conteo_causas.columns = ('Causa Nombre', 'Cantidad de Ocurrencias')

        # Mostrar el resultado
        print(conteo_causas)

        # Guardar los resultados en un archivo Excel
        self.df_resultados_exploded.to_excel(r'..\data_output\Simulaciones_Futuras_Fallas.xlsx', index=False)

        # Imprimir el mensaje de éxito
        print(f"Simulaciones completas. Resultados guardados en 'Simulaciones_Futuras_Fallas.xlsx'")

    def monteCarloProbabilidadFallaCircuitoFecha2(self,n_simulaciones=10000):
        ### Por fecha
        # # Extraer año y mes de la columna "Fecha Salida"
        self.dataFrame['Año'] = pd.to_datetime(self.dataFrame['Fecha Salida']).dt.year
        self.dataFrame['Mes'] = pd.to_datetime(self.dataFrame['Fecha Salida']).dt.month

        # # Contar el número de fallas por mes para cada año
        fallas_mensuales = self.dataFrame.groupby(['Año', 'Mes']).size().reset_index(name='Conteo')

        # # Calcular las probabilidades de ocurrencia mensual por año
        fallas_mensuales['Probabilidad'] = fallas_mensuales.groupby('Año')['Conteo'].transform(lambda x: x / x.sum())

        ## Crear un DataFrame para almacenar los resultados simulados
        simulaciones = []

        ## Realizar simulaciones Monte Carlo para cada año
        for year in fallas_mensuales['Año'].unique():
        ## Filtrar los datos para el año actual
            datos_año = fallas_mensuales[fallas_mensuales['Año'] == year]
        ## Obtener los meses y las probabilidades respectivas
            meses = datos_año['Mes']
            probabilidades = datos_año['Probabilidad']
            
        ## Simular ocurrencias mensuales
            simulacion = np.random.choice(meses, size=n_simulaciones, p=probabilidades)
            
        ## Contar ocurrencias simuladas y calcular probabilidad simulada
            conteo_simulado = pd.Series(simulacion).value_counts(normalize=True).sort_index()
            
        ## Guardar resultados
            simulaciones.append(pd.DataFrame({
                 'Mes': conteo_simulado.index,
                 'Probabilidad Simulada': conteo_simulado.values,
                 'Año': year
             }))

        ## Combinar los resultados de las simulaciones
        resultados_simulacion = pd.concat(simulaciones, ignore_index=True)

        ## Mostrar resultados de simulación
        print("Resultados de simulación de Monte Carlo:")
        print(resultados_simulacion)

        ### Guardar los resultados en un archivo Excel
        archivo_salida2 = r"..\data_output\resultados_simulacion_montecarlo_fecha.xlsx"
        resultados_simulacion.to_excel(archivo_salida2, index=False)
        print(f"Resultados guardados en {archivo_salida2}")

    # Metodo que traduce el valor el codigo de la falla en su equivalente de texto
    def traducirCodigoCausa(self):
        # 1. Leer el archivo causas_unicas.txt y crear un diccionario de mapeo
        causas_dict = {}
        # Abre el archivo y procesa cada línea
        with open('../data_input/causas_unicas.txt', 'r') as file:
            for line in file:
                # Divide cada línea por el guion y limpia los espacios
                parts = line.strip().split('-', 1)
                causa_numero = parts[0].strip()  # Número de la causa
                causa_nombre = parts[1].strip()  # Nombre de la causa
                causas_dict[causa_numero] = causa_nombre  # Agrega al diccionario

        # 2. Crear nuevas columnas para almacenar el causa_numero y causa_nombre
        self.df_resultados_exploded['Causa Nombre'] = self.df_resultados_exploded['Causa'].map(causas_dict)

    # El metodo "monteCarloProbabilidadFallaCircuito" determina la probabilidad de ocurrencia de una falla por circuito
    def monteCarloProbabilidadFallaCircuito(self, num_simulations=10000):
        #Inicializar un diccionario para almacenar las probabilidades de fallas por circuito
        probabilidades_falla = {}
                
        # Para cada circuito en los datos, realizamos una simulación
        for circuito, df_circuito in self.dataFrame.groupby('Nombre Circuito'):
            
            # Contamos las ocurrencias de cada causa para este circuito
            conteo_causas = df_circuito['Causa'].value_counts()
            
            # Normalizamos para que sumen a 1 (distribución de probabilidades)
            probabilidad_causas = conteo_causas / conteo_causas.sum()
            
            # Simular las fallas usando una distribución de Monte Carlo (Verificar si se simulan 10 mil veces simulacion)
            simulaciones = np.random.choice(probabilidad_causas.index, size=num_simulations, p=probabilidad_causas.values)
            
            # Calcular la probabilidad de que una causa ocurra en las simulaciones, se añmacenan en un dicionario
            probabilidades_falla[circuito] = {causa: (simulaciones == causa).mean() for causa in probabilidad_causas.index}
            
            # Mostrar la probabilidad de cada causa para este circuito
            print(f"\nProbabilidades de falla para el circuito {circuito}:")
            for causa, probabilidad in probabilidades_falla[circuito].items():
               print(f"Causa: {causa} - Probabilidad: {probabilidad * 100:.4f} %")
         
        #df_probabilidades = pd.DataFrame.from_dict(probabilidades_falla, orient='index')

        # Guardar el DataFrame de resultados en un archivo Excel
        #df_probabilidades.to_excel(r'..\data_output\Simulacion_probabilidad_de_Fallas.xlsx', index=False)

    # El metodo "monteCarloProbabilidadFallaCircuitoFecha" determina la ocurrencia de futuras fallas
    def monteCarloProbabilidadFallaCircuitoFecha(self,num_simulations=10000, forecast_days=30):
        # Inicializar un dataframe para almacenar los resultados de las simulaciones
        resultados_simulacion = []

        # Para cada circuito, realizamos una simulación
        for circuito, df_circuito in self.dataFrame.groupby('Nombre Circuito'):

            # Contamos las ocurrencias de cada causa para este circuito
            conteo_causas = df_circuito['Causa'].value_counts()
            
            # Normalizamos para que sumen a 1 (distribución de probabilidades)
            probabilidad_causas = conteo_causas / conteo_causas.sum()
            
            # Simulamos las fallas usando Monte Carlo
            simulaciones = np.random.choice(probabilidad_causas.index, size=num_simulations, p=probabilidad_causas.values)

            # Obtener el promedio de la fecha de la última falla
            fecha_ultima_falla = df_circuito['Fecha Salida'].max()
            
            # Simular futuras fallas y predecir su fecha de aparición
            for simulacion in simulaciones:
                # Se simula un incremento aleatorio de días para la próxima falla
                dias_hasta_falla = np.random.randint(1, forecast_days)  # Entre 1 y `forecast_days` días
                # Nueva fecha de falla es la fecha de la última falla más los días simulados
                fecha_predicha = fecha_ultima_falla + timedelta(days=dias_hasta_falla)

                # Almacenar los resultados en la lista, dentro de la lista hay un dicionario
                resultados_simulacion.append({
                    'Circuito': circuito,
                    'Causa': simulacion,
                    'Fecha Predicha': fecha_predicha.strftime('%d/%m/%Y'),
                    'Fecha Predicha (timestamp)': fecha_predicha
                })

        # Convertir los resultados a un DataFrame
        df_resultados = pd.DataFrame(resultados_simulacion)

        df_resultados.to_excel(r'..\data_output\Simulaciones_Futuras_Fallas.xlsx', index=False)


    def monteCarlo(self, n_simulaciones = 50000):

        # Cargar los datos del archivo Excel
        #archivo_excel = "frecuencia_causas.xlsx" 
        #datos = pd.read_excel(archivo_excel)

        #data = pd.read_excel("Circuitos_a_Analizar.xlsx", sheet_name='Sheet1')


        #if not {"causa", "frecuencia"}.issubset(datos.columns):
        #    raise ValueError("El archivo Excel debe contener las columnas 'causa' y 'frecuencia'.")

        # Probabilidad de cada causa basada en su frecuencia
        self.dataFrame["probabilidad"] = self.dataFrame ["frecuencia"] / self.dataFrame ["frecuencia"].sum()

        # Número de simulaciones
        #n_simulaciones = 50000

        # Realizar las simulaciones de Monte Carlo
        resultados_simulacion = np.random.choice(self.dataFrame["causa"], size=n_simulaciones, p=self.dataFrame["probabilidad"])

        # Contar la ocurrencia de cada causa en las simulaciones (Nuevo data frame)
        conteo_simulaciones = pd.DataFrame(resultados_simulacion, columns=["causa"]).value_counts().reset_index(name="conteo")
        conteo_simulaciones["probabilidad_simulada"] = conteo_simulaciones["conteo"] / n_simulaciones

        # Mostrar los resultados
        print("Resultados de simulación de Monte Carlo:")
        print(conteo_simulaciones)

        # Guardar los resultados en un archivo Excel
        archivo_salida = "..\data_output\resultados_simulacion_montecarlo.xlsx"
        conteo_simulaciones.to_excel(archivo_salida, index=False)
        print(f"Resultados guardados en {archivo_salida}")


        ### Por fecha
        # # Extraer año y mes de la columna "Fecha Salida"
        # data['Año'] = pd.to_datetime(data['Fecha Salida']).dt.year
        # data['Mes'] = pd.to_datetime(data['Fecha Salida']).dt.month

        # # Contar el número de fallas por mes para cada año
        # fallas_mensuales = data.groupby(['Año', 'Mes']).size().reset_index(name='Conteo')


        # # Calcular las probabilidades de ocurrencia mensual por año
        # fallas_mensuales['Probabilidad'] = fallas_mensuales.groupby('Año')['Conteo'].transform(lambda x: x / x.sum())

        # # Número de simulaciones para Monte Carlo
        # n_simulaciones = 10000

        # # Crear un DataFrame para almacenar los resultados simulados
        # simulaciones = []

        # # Realizar simulaciones Monte Carlo para cada año
        # for year in fallas_mensuales['Año'].unique():
        #     # Filtrar los datos para el año actual
        #     datos_año = fallas_mensuales[fallas_mensuales['Año'] == year]
            
        #     # Obtener los meses y las probabilidades respectivas
        #     meses = datos_año['Mes']
        #     probabilidades = datos_año['Probabilidad']
            
        #     # Simular ocurrencias mensuales
        #     simulacion = np.random.choice(meses, size=n_simulaciones, p=probabilidades)
            
        #     # Contar ocurrencias simuladas y calcular probabilidad simulada
        #     conteo_simulado = pd.Series(simulacion).value_counts(normalize=True).sort_index()
            
        #     # Guardar resultados
        #     simulaciones.append(pd.DataFrame({
        #         'Mes': conteo_simulado.index,
        #         'Probabilidad Simulada': conteo_simulado.values,
        #         'Año': year
        #     }))

        # # Combinar los resultados de las simulaciones
        # resultados_simulacion = pd.concat(simulaciones, ignore_index=True)

        # # Mostrar resultados de simulación
        # print("Resultados de simulación de Monte Carlo:")
        # print(resultados_simulacion)

        # # # Guardar los resultados en un archivo Excel
        # archivo_salida2 = "resultados_simulacion_montecarlo_fecha.xlsx"
        # resultados_simulacion.to_excel(archivo_salida2, index=False)
        # print(f"Resultados guardados en {archivo_salida2}")

