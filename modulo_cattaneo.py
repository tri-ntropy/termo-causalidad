import numpy as np     #Permite el empleo de los arreglos optimizados de Numpy para realizar cálculos numéricos
from numba import jit  #Incrementa la velocidad de la simulación

#Primera derivada
@jit(nopython = True)
def ddx(variable: np.ndarray, diferencial: float, full = False) -> np.ndarray:
    #Inicializacion en cero del vector derivada dado el tamaño de la variable
    derivada = np.zeros(variable.size)
    #Vectores de izquierda y derecha de la variable
    izquierda, derecha =  variable[: -2], variable[2 :]
    #Calculo de la primera derivada en todos los puntos del dominio de la variable, menos los extremos
    derivada[1 : -1] = (derecha - izquierda) / (2.0 * diferencial)
    if full: #Si se requiere, se calculan los valores de la derivada en los extremos
        derivada[0] = (variable[1] - variable[0]) / diferencial
        derivada[-1] = (variable[-1] - variable[-2]) / diferencial
    return derivada

#Segunda derivada
@jit(nopython = True)
def d2dx2(variable: np.ndarray, diferencial: float, full = False) -> np.ndarray:
    #Inicializacion en cero del vector derivada dado el tamaño de la variable
    derivada = np.zeros(variable.size)
    #Vectores de izquierda, centro y derecha de la variable de entrada
    izquierda, centro, derecha = variable[: -2], variable[1 : -1], variable[2 :]
    #Calculo de la segunda derivada en todos los puntos del dominio de la variable, menos los extremos
    derivada[1 : -1] = (derecha - 2.0 * centro + izquierda) / (diferencial * diferencial)
    if full: #Si se requiere, se calculan los valores de la derivada en los extremos
        derivada[0] = (variable[2] - 2.0 * variable[1] + variable[0]) / (diferencial * diferencial)
        derivada[-1] = (variable[-1] - 2.0 * variable[-2] + variable[-3]) / (diferencial * diferencial)
    return derivada

@jit(nopython = True)
def calc_ku(u: np.ndarray, d2Tdx2: np.ndarray, alfa: np.ndarray, tau: np.ndarray) -> np.ndarray:
    return (alfa * d2Tdx2 - u) / tau

@jit(nopython = True)
def rungekutta_u(u: np.ndarray, T: np.ndarray, alfa: np.ndarray, tau: float, dx: float, dt: float) -> np.ndarray:
    d2Tdx2 = d2dx2(T, dx, full = False)
    #Calculo de k1
    y1 = np.copy(u)
    k1 = calc_ku(y1, d2Tdx2, alfa, tau)
    #Calculo de k2
    y2 = y1 + 0.5 * dt * k1
    k2 = calc_ku(y2, d2Tdx2, alfa, tau)
    #Calculo de k3
    y3 = y1 + 0.5 * dt * k2
    k3 = calc_ku(y3, d2Tdx2, alfa, tau)
    #Calculo de k4
    y4 = y1 + dt * k3
    k4 = calc_ku(y4, d2Tdx2, alfa, tau)
    #Resultado
    return u + (dt / 6.0) * (k1 + 2.0 * (k2 + k3) + k4)

@jit(nopython = True)
def calc_kq(q: np.ndarray, dTdx: np.ndarray, tau: np.ndarray) -> np.ndarray:
    return - (dTdx + q) / tau

@jit(nopython = True)
def rungekutta_q(q: np.ndarray, T: np.ndarray, tau: float, dx: float, dt: float) -> np.ndarray:
    dTdx = ddx(T, dx, full = True)
    #Calculo de k1
    y1 = np.copy(q)
    k1 = calc_kq(y1, dTdx, tau)
    #Calculo de k2
    y2 = y1 + 0.5 * dt * k1
    k2 = calc_kq(y2, dTdx, tau)
    #Calculo de k3
    y3 = y1 + 0.5 * dt * k2
    k3 = calc_kq(y3, dTdx, tau)
    #Calculo de k4
    y4 = y1 + dt * k3
    k4 = calc_kq(y4, dTdx, tau)
    #Resultado
    return q + (dt / 6.0) * (k1 + 2.0 * (k2 + k3) + k4)

@jit(nopython = True)
def calc_T(temperatura: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    return temperatura + dt * u