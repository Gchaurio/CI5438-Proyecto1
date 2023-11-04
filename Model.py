import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error

class Model(object):
    
    def __init__(self, data: pd.DataFrame, dep: str, ind: list):
        self.data = data
        self.dep = dep
        self.ind = ind
        self.w = None
        self.errores = []
        self.allw = []
        self.test_y = None
        self.test_x = None
        self.real_values = None
        self.to_evaluate = None
        self.results = None

    def get_training_test(self, data: pd.DataFrame):

        x_train, x_test, y_train, y_test = train_test_split(data[self.ind], data[self.dep], test_size=0.2, random_state=42)

        return x_train, x_test, y_train, y_test

    def train(self, learning_rate=0.01, iters=1000, epsilon=0.001):
        '''
        Aplicacion del gradiente descendiente para el entrenamiento
        del modelo.
        '''
        if self.w: # El modelo ya fue entrenado, se retornan los pesos calculados en el entrenamiento
            return self.w
        
        # 
        X, X_test, Y, Y_test = self.get_training_test(self.data)

        self.test_y = Y_test
        self.test_x = X_test
        
        X["t_ind"] = 1.0

        W = np.array([0.5] * len(list(X.columns.values)))

        self.test_w = W
        
        for i in range(iters):
            hw = np.dot(X,W)
            E = Y - hw

            self.errores.append(np.mean(abs(E)))
            W = W + (learning_rate * (np.dot(X.T, E) * (2/len(X))))
            
            self.allw.append(W)

            err_it = np.mean(abs(E))

            if err_it <= epsilon:
                print("Convergencia Alcanzada")
                print(f'Iteracion de Convergencia: {i}')
                break

        self.w = W

    def predict(self, data: pd.DataFrame):

        predict =  np.matmul(data,self.w)

        return predict

    def evaluate(self, data):

        real_values = data[self.dep]

        to_evaluate = data[self.ind]

        to_evaluate["t_ind"] = 1.0

        result = self.predict(to_evaluate)

        self.real_values = real_values

        self.to_evaluate = to_evaluate

        self.results = result
        
        errors_rel = real_values - result
        error_rel_absolute = errors_rel.abs()
        error_rel_mean = error_rel_absolute.mean()

        # Print
        return print(f'error relativo = {error_rel_mean / real_values.mean()}')

    