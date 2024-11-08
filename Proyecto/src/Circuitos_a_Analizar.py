import pandas as pd

# Lista de circuitos a analizar
circuitos_a_analizar = [
    23868, 23902, 23885, 47685, 20417, 22134, 11968, 1785, 51068, 47736,
    13668, 12053, 11934, 18751, 23851, 13634, 47753, 30651, 46002, 22168,
    30685, 17034, 6868, 13651, 6851, 47651, 27217, 45968, 8568, 8517,
    1717, 18734, 13617, 6885, 17017, 5117, 12002, 46053, 8585, 11951,
    1751, 25602, 8534, 11917, 18717, 47719, 6834, 1819, 44285, 1802,
    12019, 6902, 42517, 25585, 15453, 3434, 25534, 42534, 30668,
    3485, 25517, 15419, 11985
]

# Cargar el archivo Excel original
df = pd.read_excel('C:/Users/alchernandez/Documents/Universidad/CienciaDeDatos/Proyecto/DatosProyecto.xlsx')

# Filtrar el DataFrame para que solo contenga los circuitos a analizar
df_filtrado = df[df['Circuito'].isin(circuitos_a_analizar)]

# Guardar el DataFrame filtrado en un nuevo archivo Excel
df_filtrado.to_excel('Circuitos_a_Analizar.xlsx', index=False)

print("El archivo ha sido reducido y guardado como 'Circuitos_a_Analizar.xlsx'.")

