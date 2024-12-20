from lecturaDatosProyecto import *
from pathlib import Path

if __name__ == "__main__":

    # Obtener el directorio actual
    current_dir = Path(__file__).resolve().parent   
    # Concatenar la ruta con 'data' y 'DatosProyecto.xls' de manera eficiente
    data_file = current_dir.parent / 'data_input' / 'DatosProyecto.csv'

    # Usar la ruta concatenada para instanciar el objeto
    proyectoCienciaDatos = datosProyecto(data_file)
    proyectoCienciaDatos.display_ParetoDeMasFallas()
    proyectoCienciaDatos.fallas_a_Analizar()
    proyectoCienciaDatos.predecirResultadosCircuito2("106")
    #proyectoCienciaDatos.MonteCarloPorFechaConMayorProbabilidadOcurrenciaDeFallas()
    #proyectoCienciaDatos.monteCarloProbabilidadFallaCircuitoFecha2()
    # El metodo "monteCarloProbabilidadFallaCircuito" determina la probabilidad de ocurrencia de una falla por circuito
    #proyectoCienciaDatos.monteCarloProbabilidadFallaCircuito()
    # El metodo "monteCarloProbabilidadFallaCircuitoFecha" determina la ocurrencia de futuras fallas
    #proyectoCienciaDatos.monteCarloProbabilidadFallaCircuitoFecha()
    #proyectoCienciaDatos.monteCarlo()
