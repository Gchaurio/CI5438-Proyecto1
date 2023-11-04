import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error
import os

class Model(object):
    
    def __init__(self, data: pd.DataFrame, dep: str, ind: list):
        self.data = data
        self.dep = dep
        self.ind = ind
        self.w = None
        self.errores = []
        self.test_y = None
        self.test_x = None
        self.real_values = None
        self.to_evaluate = None
        self.results = None
        self.convergencia = 0

    def get_training_test(self, data: pd.DataFrame):

        x_train, x_test, y_train, y_test = train_test_split(data[self.ind], data[self.dep], test_size=0.2, random_state=42)

        return x_train, x_test, y_train, y_test

    def train(self, learning_rate=0.01, iters=1000, epsilon=0.01):
        '''
        Aplicacion del gradiente descendiente para el entrenamiento del modelo.
        '''
        if self.w: # El modelo ya fue entrenado, se retornan los pesos calculados en el entrenamiento
            return self.w
        
        # Division del conjunto de datos: Conjunto entrnamiento - conjunto de pruebas
        X, self.test_x, Y, self.test_y = self.get_training_test(self.data)

        # Columna del valor independiente
        X["t_ind"] = 1.0

        # Arreglo inicial de pesos
        W = np.array([0.0] * len(list(X.columns.values)))
        
        # Algoritmo de descenso gradiente
        for i in range(iters):
            
            # Hipotesis
            hw = np.dot(X,W)
            
            # Error de la iteracion
            E = Y - hw

            # Guardamos el promedio de error de estA iteracion
            err_it = np.mean(abs(E))

            # Guardamos el promedio de error de esta iteracion en el arreglo de errores
            self.errores.append(err_it)
            
            # Se modifican los pesos segun los calculos de la iteracion
            W = W + (learning_rate * (np.dot(X.T, E) * (2/len(X))))

            # Numero de iteracion en la cual se consigue la convergencia
            self.convergencia = i

            # Condicion de convergencia -Error menor que epsilon o error igual en las ultimas dos iteraciones
            if err_it <= epsilon or (len(self.errores) > 2 and self.errores[-1] == self.errores[-2]):
                print("Convergencia Alcanzada")
                print(f'Iteracion de Convergencia: {i}')
                break

        # Almacenamos el valor final de los pesos
        self.w = W

        # Probamos los pesos obtenidos con el conjunto de pruebas
        self.test()

    def predict(self, data: pd.DataFrame):
        '''
        Funcion utilizada para predecir
        '''

        # Se hace el producto del arreglo de pesos calculado por los datos ingresados
        predict = np.dot(data,self.w)

        return predict

    def evaluate(self, data):
        '''
        Funcion utilizada para probar un conjunto de datos con el modelo
        '''

        # Valores reales del conjunto de datos
        real_values = data[self.dep]

        # Valores de las variables a ser evaluadas
        to_evaluate = data[self.ind]

        # Termino independiente
        to_evaluate["t_ind"] = 1.0

        # Prediccion de valores con el modelo
        result = self.predict(to_evaluate)

        # Analisis de error de los datos evaluados
        errors_rel = real_values - result
        error_rel_absolute = errors_rel.abs()
        error_rel_mean = error_rel_absolute.mean()

        # Print
        return print(f'error relativo = {error_rel_mean / real_values.mean()}')
    
    def test(self):
        '''
        Funcion que evalua los datos del conjunto de pruebas
        luego de entrenar el modelo
        '''
        
        # Se combinan los datos de prueba en un solo DataFrame
        data = self.test_x
        data["Price"] = self.test_y
        
        # Evaluacion
        self.evaluate(data)

    def plot_error(self, name="g1.png"):
        '''
        Grafico del error de cada iteracion
        '''

        k = len(self.errores)
        plt.xlabel("Iteraciones")
        plt.ylabel("Error")
        plt.plot(self.convergencia, self.errores[self.convergencia], c='red', marker='o')
        plt.plot(range(int(k)), self.errores[:int(k)])
        plt.savefig(os.path.join("graficos", name))
        plt.show()