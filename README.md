# Termo-causalidad

Repositorio de notebooks de Jupyter para el cálculo de Entropía de Transferencia Simbólica sobre las variables de los modelos de conducción de calor en 1D con propiedades intensivas constantes; es decir, materiales cuya densidad, capacidad calorífica, conductividad térmica, difusividad térmica, y longitud libre media de los fonones son constantes. Para el estudio se emplea la herramienta de la Entropía de Transferencia Simbólica.

## Para empezar

Es recomendable, además de contar con el software necesario, conocimiento sobre transferencia de calor, simulación numérica (diferencias finitas y Runge-Kutta de cuarto orden) y un poco de estadística descriptiva.

### Prerequisitos de software

* Alguna distribución reciente de Python, se sugiere conda (o anaconda)
* Numpy, Scipy, Numba, matplotlib, Seaborn

## Uso

Basta correr cada notebook en el orden que viene numerado.

* El mallado espacio-temporal se modifica en el notebook #1, donde se determina un ensamble de realizaciones de la condición de frontera, además de otros aspectos básicos. 
* Los notebook #2 son para realizar pruebas y corroborar que la simulación numérica de cada modelo converge.
* Los experimentos de los modelos de conducción de calor y los experimentos para validar la metología se encuentran los notebook #3.
* El notebook #4 sirve para obtener la calibración numérica
* Finalmente, el cálculo de la Entropia de Transferencia Simbólica, por modelo de conducción y experimento de validación se da en los notebook #5. Es importante indicar sobre que modelo, experimento y variables se calcula la Entropia de Transferencia Simbólica.

## Información adicional

Se mejorará la documentación en un futuro.
