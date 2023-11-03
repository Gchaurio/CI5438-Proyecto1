import pandas as pd
import numpy as np

class Model(object):
    
    def __init__(self, data: pd.DataFrame, dep: str, ind: list):
        self.df = data.filter(ind+[dep])
        self.dep = dep
        self.ind = ind
        self.w = None

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

        self.w = pd.Series([1.0]*(len(self.dep)+1))
        
        for i in range(iters):
            hw = np.dot(self.w, X.T)
            err = Y - hw

            self.w = self.w + (learningRate * err * (2/len(X)))