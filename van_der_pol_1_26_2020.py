'''
Van der Pol Oscillator Simulation & Analysis
'''
import Simulation_Analysis_Toolset as sat
import numpy as np
import matplotlib.pyplot as plt
import cmath



class VanderPolSim(sat.DynamicalSystemSim):
    '''Simulation object - feed parameters and returns simulation data structure'''
    def __init__(self, T = 50, dt = .01, mu = 3, x0 = .5, y0 = .5):
    
        X0 = np.asarray([x0, y0])
        super().__init__(X0, T, dt)
        self.mu = mu   # damping coefficient 
        

    def deriv(self, t, X):
        '''
        Derivative function for use in integrator. X is 2-vector of state
        van der pol eqs:
        dx/dt = y
        dy/dt = mu * (1-x**2) * y  - x
         '''
        x = X[0]
        y = X[1]
        mu = self.mu
        
        def y_dot(x, y, mu):
            return mu * (1-x*x) * y - x
    
        def x_dot(y):
            return y
    
        num = np.asarray([x_dot(y), y_dot(x, y, mu)])
        return num

    
    def copy_sim(self):
        return VanderPolSim(
            T = self.T,
            dt = self.dt,
            mu = self.mu,
            x0 = self.X0[0],
            y0 = self.X0[1]
            )

class VanderPolAnalyzer(sat.PlanarLimitCycleAnalyzer):
    
    def lc_eigen_decomp(self, sim_data, limit_cycle):
        
        def l_pos(x, y, mu):
            ''' eigenvalue 0 of linear approximation'''
            num = mu * (1 - x**2) + cmath.sqrt(mu**2 * (1 - x**2)**2 + 4 * (2*mu*x*y - 1) )
            return np.real(num / 2)
    
        def l_neg(x, y, mu):
            ''' eigenvalue 0 of linear approximation'''
            num = mu * (1 - x**2) - cmath.sqrt(mu**2 * (1 - x**2)**2 + 4 * (2*mu*x*y - 1) )
            return np.real(num / 2)
        
        def evec_pos(x, y, mu):
            ''' get the eigenvector associated with positive eigval '''
            vec = np.asarray([1, l_pos(x, y, mu)])
            return vec / np.linalg.norm(vec)
        
        def evec_neg(x, y, mu):
            ''' get the other eigenvector '''
            vec = np.asarray([
                l_neg(x, y , mu) - mu * (1 - x**2),
                2 * mu * x * y - 1
                ])
            return vec / np.linalg.norm(vec)
        
        xs = limit_cycle['X'][0,:]
        ys = limit_cycle['X'][1,:]
        mu = sim_data['mu']
        N = len(xs)
        
        l_negs = np.zeros((N,1))
        l_poss = np.zeros((N,1))
        evec_negs = np.zeros((2,N))
        evec_poss = np.zeros((2,N))

        for i in np.arange(N):
            l_negs[i] = l_neg(xs[i], ys[i], mu)
            l_poss[i] = l_pos(xs[i], ys[i], mu)
            
            evec_negs[:,i] = evec_neg(xs[i], ys[i], mu)
            evec_poss[:,i] = evec_pos(xs[i], ys[i], mu)
            
    
        
        eig_dec = {
            "l_neg" : l_negs,
            "l_pos" : l_poss,
            "evec_neg" : evec_negs,
            "evec_pos" : evec_poss,
            }  
        
        return eig_dec

sim = VanderPolSim()
data = sim.run_sim()

sim_pert  = sim.copy_sim()
sim_pert.X0[0] = 1
sim_pert.X0[1] = 2

data_pert = sim_pert.run_sim()

vpa = VanderPolAnalyzer()

limit_cycle = vpa.get_limit_cycle(data)
ed = vpa.lc_eigen_decomp(data, limit_cycle)

idxs = np.linspace(0,len(limit_cycle['t'])-1, num = 20, dtype = int)



plt.show()




