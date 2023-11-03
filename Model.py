import pandas as pd
import numpy as np

class Model(object):
    
    def __init__(self, data: pd.DataFrame, dep: str, ind: list):
        self.df = data
        self.dep = dep
        self.ind = ind
        self.w = None
        self.errores = []
        self.allw = []

    def train(self, learningRate=0.01, iters=1000, epsilon=0.01):
        '''
        Aplicacion del gradiente descendiente para el entrenamiento
        del modelo.
        '''
        if self.w:
            return self.w
        
        Y = self.df.loc[:,self.dep]
        X = self.df.loc[:,self.ind]
        X.insert(0, "termino_indep", [1.0]*len(X))
        lr = learningRate

        W = [1.0 for i in X]
        
        for i in range(iters):
            hw = np.matmul(X,W)
            E = Y - hw

            self.errores.append(np.mean(abs(E)))
            W = W + (lr * (np.matmul(X.T, E) * (2/len(X))))
            
            self.allw.append(W)

            err_it = np.max(abs(E))

            if err_it <= epsilon:
                print("EPSILON")
                break

        self.w = W