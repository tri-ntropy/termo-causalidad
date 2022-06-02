from scipy import stats
import numpy as np

def simbolizar(X, m = 3):
    """
    Convierte una serie numérica de valores a su versión simbólica basándose en
    ventanas de m valores consecutivos.
    
    Parámetros
    ----------
    X : Serie a simbolizar
    m : Longitud de la ventana
    
    Regresa
    ----------
    Arreglo de X simbólico
    """
    
    if type(X) != np.ndarray:
        X = np.array(X)

    if m >= X.size:
        raise ValueError("La serie debe ser más grande que la ventana")
    
    dummy = []
    for i in range(m):
        l = np.roll(X, -i)
        dummy.append(l[: -(m - 1)])
    
    dummy = np.array(dummy)
    
    simX = []
    
    for ventana in dummy.T:
        ranking = stats.rankdata(ventana, method = "dense")
        simbolo = np.array2string(ranking, separator = "")
        simbolo = simbolo[1 : -1]
        simX.append(simbolo)
        
    return np.array(simX)

def informacion_mutua(simX, simY):
    """
    Computa el valor IM(X, Y) entre las series simbólicas X y Y
    
    Parámetros
    ----------
    simX : Serie simbólica X
    simY : Serie simbólica Y
    
    Regresa
    ----------
    Valor de la información mútua simbólica
    """

    if simX.size != simY.size:
        raise ValueError('Los arreglos deben tener la misma longitud')
    
    n_simbolos = len(np.unique(np.concatenate((simX, simY))).tolist())
        
    jp = probabilidades_conjuntas(simX, simY)
    pX = probabilidades(simX)
    pY = probabilidades(simY)
    
    IM = 0

    for yi in list(pY.keys()):
        for xi in list(pX.keys()):
            a = pX[xi]
            b = pY[yi]
            
            try:
                c = jp[yi][xi]
                IM += c * np.log(c /(a * b)) / np.log(n_simbolos)
            
            except KeyError:
                continue
            
            except:
                print("Error inesperado")
                raise
        
    return IM

def entropia_transferencia(simX, simY):
    """
    Computa el valor T(Y->X) de las series simbólicas de Y a X
    
    Parámetros
    ----------
    simX : Serie simbólica X
    simY : Serie simbólica Y
    
    Regresa
    ----------
    Valor de la entropía de transferencia simbólica
    """

    if simX.size != simY.size:
        raise ValueError('Los arreglos deben tener la misma longitud')
        
    cp = probabilidades_transicion(simX)
    cp2 = probabilidades_transicion_dobles(simX, simY)
    jp = probabilidades_counjuntas_consecutivas(simX, simY)
    
    ETS = 0
    
    for yi in list(jp.keys()):
        for xi in list(jp[yi].keys()):
            for xii in list(jp[yi][xi].keys()):

                try:
                    a = cp[xi][xii]
                    b = cp2[yi][xi][xii]
                    c = jp[yi][xi][xii]
                    ETS += c * np.log2(b / a)
                    
                except KeyError:
                    continue
                
                except:
                    print("Error inesperado")
                    raise
    del cp
    del cp2
    del jp
    
    return ETS

def serie_ets(simX, simY, pasos):
    """
    Genera las dos series de entropía de transferencia simbólica
    T(X->Y) y T(Y->X) sobre un determinado número de pasos
    
    Parámetros
    ----------
    simX : Serie simbólica X
    simY : Serie simbólica Y
    pasos : Número de pasos en los cuales se calcula la entropía de
            transferencia simbólica
    
    Regresa
    ----------
    Las arreglos de curvas de entropía de transferencia simbólica
    """
    # Inicialización
    ets_xy = np.empty(pasos + 1)
    ets_yx = np.empty(pasos + 1)
    # Cálculo de valores
    for i in range(-1, pasos):
        ets_xy[i + 1] = entropia_transferencia(simX, np.roll(simY, -i))
        ets_yx[i + 1] = entropia_transferencia(simY, np.roll(simX, -i))
            
    return ets_xy, ets_yx

def causa(x, y, ct, tv = 3):
    """
    Genera las curvas de entropia de transferencia
    para una realizacion del sistema
    
    Parámetros
    ----------
    x : Serie de tiempo de la variable X
    y : Serie de tiempo de la variable Y
    ct : Valores de corrimiento de pasos en los cuales
         se calcula los valores de T(X->Y)
    
    Regresa
    ----------
    Los arreglos de T(X->Y), T(Y->X)
    """
    # Simbolización
    sim_x = simbolizar(x, tv)
    sim_y = simbolizar(y, tv)
    # Curvas de ETS
    entropia_xy, entropia_yx = serie_ets(sim_x, sim_y, pasos = ct)
    
    return entropia_xy, entropia_yx 

def probabilidades(simX):
    """
    Computa las probabilidades de los elementos del alfabeto
    de símbolos de la serie X. 
    
    Parámetros
    ----------
    simX : Serie simbólica X
    
    Regresa
    ----------
    Probabilidades del diccionario de símbolos
    """
    
    # Inicialización
    p = {}
    n = simX.size

    for xi in simX:
        if xi in p:
            p[xi] += 1.0 / n
        else:
            p[xi] = 1.0 / n
    
    return p

def probabilidades_conjuntas(simX, simY):
    """
    Computa las probabilidades conjuntas P(yi, xi)
    
    Parámetros
    ----------
    simX : Serie simbólica X
    simY : Serie simbólica Y
    
    Regresa
    ----------
    Matriz de probabilidades conjuntas
    """

    if simX.size != simY.size:
        raise ValueError('Los arreglos deben tener la misma longitud')
    
    # Inicialización
    jp = {}
    n = simX.size

    for yi, xi in zip(simY, simX):
        if yi in jp:
            if xi in jp[yi]:
                jp[yi][xi] += 1.0 / n
            else:
                jp[yi][xi] = 1.0 / n
        else:
            jp[yi] = {}
            jp[yi][xi] = 1.0 / n
    
    return jp

def probabilidades_condicionales(simX, simY):
    """
    Computa las probabilidades condicionales P(xi | yi)
    
    Parámetros
    ----------
    simX : Serie simbólica X
    simY : Serie simbólica Y
    
    Regresa
    ----------
    Matriz de las probabilidades condicionales
    """
    
    if simX.size != simY.size:
        raise ValueError('Los arreglos deben tener la misma longitud')
    
    # Inicialización
    cp = {}
    n = {}

    for xi, yi in zip(simX, simY):
        if yi in cp:
            n[yi] += 1
            if xi in cp[yi]:
                cp[yi][xi] += 1.0
            else:
                cp[yi][xi] = 1.0
        else:
            cp[yi] = {}
            cp[yi][xi] = 1.0
            n[yi] = 1

    for yi in list(cp.keys()):
        for xi in list(cp[yi].keys()):
            cp[yi][xi] /= n[yi]
    
    return cp

def probabilidades_transicion(simX):
    """
    Computa las probabilidades de transición P(xii | xi)
    
    Parámetros
    ----------
    simX : Serie simbólica X
    
    Regresa
    ----------
    Matriz de probabilidades de transición
    """

    cp = probabilidades_condicionales(simX[1 :], simX[: -1])
    
    return cp

def probabilidades_condicionales_dobles(simX, simY, simZ):
    """
    Computa las probabilidades condicionales P(xi | yi, zi).
    
    Parámetros
    ----------
    simX : Serie simbólica X.
    simY : Serie simbólica Y.
    simZ : Serie simbólica Z.
    
    Regresa
    ----------
    Matriz de probabilidades condicionales
    """

    if (simX.size != simY.size) or (simY.size != simZ.size):
        raise ValueError('Los arreglos deben tener la misma longitud')
    
    # Inicialización
    cp = {}
    n = {}

    for x, y, z in zip(simX, simY, simZ):
        if y in cp:
            if z in cp[y]:
                n[y][z] += 1.0
                if x in cp[y][z]:
                    cp[y][z][x] += 1.0
                else:
                    cp[y][z][x] = 1.0
            else:
                cp[y][z] = {}
                cp[y][z][x] = 1.0
                n[y][z] = 1.0
        else:
            cp[y] = {}
            n[y] = {}
            
            cp[y][z] = {}
            n[y][z] = 1.0
            
            cp[y][z][x] = 1.0

        
    for y in list(cp.keys()):
        for z in list(cp[y].keys()):
            for x in list(cp[y][z].keys()):
                cp[y][z][x] /= n[y][z]
    
    return cp

def probabilidades_transicion_dobles(simX, simY):
    """
    Computa las probabilidades de transición dobles P(xii | xi, yi)
    
    Parámetros
    ----------
    simX : Serie simbólica X
    simY : Serie simbólica Y
    
    Regresa
    ----------
    Matriz de probabilidades de transición dobles
    """

    if simX.size != simY.size:
        raise ValueError('Los arreglos deben tener la misma longitud')

    cp = probabilidades_condicionales_dobles(simX[1 :], simY[: -1], simX[: -1])
    
    return cp

def probabilidades_conjuntas_triples(simX, simY, simZ):
    """
    Computa las probabilidades conjuntas P(xi, yi, zi).
    
    Parámetros
    ----------
    simX : Serie simbólica X
    simY : Serie simbólica Y
    simZ : Serie simbólica Z
    
    Regresa
    ----------
    Matriz de probabilidades conjuntas triples
    """
    
    if (simX.size != simY.size) or (simY.size != simZ.size):
        raise ValueError('Los arreglos deben tener la misma longitud')

    
    # Inicialización
    jp = {}
    n = len(simX)

    for x, y, z in zip(simX,simY,simZ):
        if y in jp:
            if z in jp[y]:
                if x in jp[y][z]:
                    jp[y][z][x] += 1.0 / n
                else:
                    jp[y][z][x] = 1.0 / n
            else:                
                jp[y][z] = {}
                jp[y][z][x] = 1.0 / n
        else:
            jp[y] = {}
            jp[y][z] = {}
            jp[y][z][x] = 1.0 / n
    
    return jp

def probabilidades_counjuntas_consecutivas(simX, simY):
    """
    Computa las probabilidades conjuntas P(xii, xi, yi)
    
    Parámetros
    ----------
    simX : Serie simbólica X
    simY : Serie simbólica Y

    Regresa
    ----------
    Matriz de probabilidades conjuntas consecutivas
    """
    
    if len(simX) != len(simY):
        raise ValueError('All arrays must have same length')

    jp = probabilidades_conjuntas_triples(simX[1 :], simY[: -1], simX[: -1])
    
    return jp