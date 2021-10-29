import numpy as np


class Frankefunction:
    def __init__(self, x, y, n_complex=5):
        self.x,self.y = np.meshgrid(x,y)
        self.n_complex = n_complex

    def calculate(self):
        term1 = 0.75*np.exp(-(0.25*(9*self.x-2)**2) - 0.25*((9*self.y-2)**2))
        term2 = 0.75*np.exp(-((9*self.x+1)**2)/49.0 - 0.1*(9*self.y+1))
        term3 = 0.5*np.exp(-(9*self.x-7)**2/4.0 - 0.25*((9*self.y-3)**2))
        term4 = -0.2*np.exp(-(9*self.x-4)**2 - (9*self.y-7)**2)
        return term1 + term2 + term3 + term4

    def designMatrix(self):
        if len(self.x.shape) > 1:
            x = np.ravel(self.x)
            y = np.ravel(self.y)

        N = len(x)
        l = int((self.n_complex+1)*(self.n_complex+2)/2) 
        X = np.ones((N,l))

        for i in range(1,self.n_complex+1):
            q = int((i)*(i+1)/2)
            for k in range(i+1):
                X[:,q+k] = (x**(i-k))*(y**k)
        return X

    def dataset(self,mu_N=0 ,sigma_N=0.1):
        #x,y = np.meshgrid(self.x,self.y)
        self.data_output = self.calculate() +mu_N +sigma_N*np.random.randn(len(self.x),len(self.y)) #input data
        self.data_output_target = np.ravel(self.calculate())

