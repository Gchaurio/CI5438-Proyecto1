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

    def train(self, learningRate=0.001, iters=1000):
        '''
        Aplicacion del gradiente descendiente para el entrenamiento
        del modelo.
        '''
        if self.w:
            return self.w
        
        Y = self.df.loc[:,self.dep]
        X = self.df.loc[:,self.ind]
        X.insert(0, "termino_indep", [1.0]*len(X))

        W = [1.0 for i in X]
        
        for i in range(iters):
            hw = np.dot(W, X.T)
            err = Y - hw
            self.errores.append(np.mean(abs(err)))
            W = W + (learningRate * np.dot(X.T, err) * (2/len(X)))
            self.w = W

            self.allw.append(W)