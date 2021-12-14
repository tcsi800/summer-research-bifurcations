import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Match,Tuple, Optional
import numpy.random as rnd

class Interval():
    left: float
    right: float 

    def __init__(self, l: float, r: float) -> None:
        self.left = l
        self.right = r

    def xin(self, x: float) -> bool:
         return x<self.right and x>self.left

f1 = lambda x,y: x*y+x**3
# f1 = lambda x,y: y - (1/3) * x**3 + x


# dx = f(x,y) * dt + σ * dWt
# dy = ε * dt





def euler_maruyama(f: Callable[[float, float], float], t1: float, t2: float, x0: float, y0: float, N: int, K, eps: float = 0, sig: float = 1, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = rnd.default_rng(seed)
    W = rng.standard_normal((N-1, K))
    t = np.linspace(t1, t2, N)
    dt = (t2-t1)/N
    y = np.zeros((N, K))
    x = np.zeros((N, K))
    x[0,:] = x0 * np.ones(K)
    y[0,:] = y0 * np.ones(K)
    a = 1.05
    dx = lambda i: f(x[i,:], y[i,:]) * dt + sig * np.sqrt(dt) * W[i,:]
    # dy = lambda i: eps*(a*np.ones(K)-x[i-1,:]) * dt
    

    for i in range(1, N):
        y[i,:] = y[i-1,:] + eps*dt * np.ones(K)
        x[i,:] = x[i-1,:] + dx(i-1) 
        


    return t, x, y

if __name__=="__main__":
    
    K = 100
    N = 1000
    t, x, y = euler_maruyama(f1, 0,  50, 0, -1, N, K, eps = 0.05, sig=0.1, seed=100)
    # for k in range(K):
    #     z = False
    #     for i in range(N):
    #         if x[i,k]<y[i,k] or z:
    #             z = True
    #             x[i,k] = y[i,k]

    f, ax  = plt.subplots()
    # ax1 = ax.twinx()
    v = np.std(x, axis = 1)
    # ax.plot(t,v)
    [ax.plot(t,x[:,i]) for i in range(K)] 
    
    # ax1.plot(x,y)
    ax.set_ylim(-1,1.3)
    plt.show()
