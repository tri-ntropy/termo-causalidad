import numpy as np     # Permite el empleo de los arreglos optimizados de Numpy para realizar cálculos numéricos
from numba import jit  # Incrementa la velocidad de la simulación

# Primera derivada
@jit(nopython = True)
def ddx(variable: np.ndarray, diferencial: float, full = False) -> np.ndarray:
    # Inicializacion en cero del vector derivada dado el tamaño de la variable
    derivada = np.zeros(variable.size)
    # Vectores de izquierda y derecha de la variable
    izquierda, derecha =  variable[: -2], variable[2 :]
    # Calculo de la primera derivada en todos los puntos del dominio de la variable, menos los extremos
    derivada[1 : -1] = (derecha - izquierda) / (2.0 * diferencial)
    if full:  # Si se requiere, se calculan los valores de la derivada en los extremos
        derivada[0] = (variable[1] - variable[0]) / diferencial
        derivada[-1] = (variable[-1] - variable[-2]) / diferencial
    return derivada

# Segunda derivada
@jit(nopython = True)
def d2dx2(variable: np.ndarray, diferencial: float, full = False) -> np.ndarray:
    # Inicializacion en cero del vector derivada dado el tamaño de la variable
    derivada = np.zeros(variable.size)
    # Vectores de izquierda, centro y derecha de la variable de entrada
    izquierda, centro, derecha = variable[: -2], variable[1 : -1], variable[2 :]
    # Calculo de la segunda derivada en todos los puntos del dominio de la variable, menos los extremos
    derivada[1 : -1] = (derecha - 2.0 * centro + izquierda) / (diferencial * diferencial)
    if full:  # Si se requiere, se calculan los valores de la derivada en los extremos
        derivada[0] = (variable[2] - 2.0 * variable[1] + variable[0]) / (diferencial * diferencial)
        derivada[-1] = (variable[-1] - 2.0 * variable[-2] + variable[-3]) / (diferencial * diferencial)
    return derivada

# Funcion que calcula los k del metodo de runge-kutta
@jit(nopython = True)
def calc_k(variable: np.ndarray, alfa: np.ndarray, dx: float) -> np.ndarray:
    # Calculo del gradiente y laplaciano de la temperatura
    d2Tdx2 = d2dx2(variable, dx, full = True)
    # Resultado
    return alfa * d2Tdx2

# Proceso de integracion de runge-kutta
@jit(nopython = True)
def rungekutta(variable: np.ndarray, alfa: np.ndarray, dx: float, dt: float) -> np.ndarray:
    # Calculo de k1
    y1 = np.copy(variable)
    k1 = calc_k(y1, alfa, dx)
    # Calculo de k2
    y2 = y1 + 0.5 * dt * k1
    k2 = calc_k(y2, alfa, dx)
    # Calculo de k3
    y3 = y1 + 0.5 * dt * k2
    k3 = calc_k(y3, alfa, dx)
    # Calculo de k4
    y4 = y1 + dt * k3
    k4 = calc_k(y4, alfa, dx)
    # Resultado
    return variable + (dt / 6.0) * (k1 + 2.0 * (k2 + k3) + k4)

# Calculo del flujo de calor
@jit(nopython = True)
def calc_q(variable: np.ndarray) -> np.ndarray:
    return - variable