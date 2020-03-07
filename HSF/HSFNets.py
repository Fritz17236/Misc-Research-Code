'''
Created on Feb 19, 2020

@author: cbfritz

Class definitions for Spiking Neural Networks Simulations
'''

import Simulation_Analysis_Toolset as sat
import numpy as np
import matplotlib.pyplot as plt
from abc import abstractmethod
from matplotlib import cm 
from _nsis import err
'''
Class Definitions
'''
    

class SpikingNeuralNet(sat.DynamicalSystemSim):
    ''' 
    A SpikingNeuralNet object is used to simulate a spiking neural network that
    implements a given linear dynamical system. This is an abstract class used to
    define general properties common to all spiking neural nets.
    '''
    
    def __init__(self, T, dt, N, D, lds, lam, t0 = 0):
         
    # neural net needs a linear dynamical system to implement (x' = Ax + Bu)
        self.A = lds.A  # Dynamics Matrix
        self.B = lds.B  # Input Matrix
        self.u = lds.u # Input Signal (function handle with at least t argument)
        self.lds = lds # save lds for later if needed
        self.N = N # number of neurons in network
        self.lam = lam #leak term
        self.D = D # Decoder matrix
        
        self.O = {}
        for i in np.arange(self.N):
            self.O[str(i)] = []
        
        self.V = np.zeros((N,1))
        
        super().__init__(None, T, dt, t0)
    
    
    def deriv(self, t, X):
        sat.DynamicalSystemSim.deriv(self, t, X)
        pass
    
    @abstractmethod  
    def run_sim(self):
        pass
    
    
    @abstractmethod
    def V_dot(self, X):
        pass


    @abstractmethod
    def r_dot(self, r):
        '''
        Compute the post synaptic current
        '''
        pass
    

    @abstractmethod
    def spike(self, V, O_t):
        ''' 
        update neuron voltage V with spikes occuring at O_t,
        O_t is a N-vector of 1s (spike at time t) and zeros (no spike)
        '''
        assert( len(O_t) <= self.N)


class GapJunctionDeneveNet(SpikingNeuralNet):
    ''' Spiking Net According to Classic Deneve Paper w/ Erics Voltage Modification''' 
    def __init__(self, T, dt, N, D, lds, lam, t0 = 0):
        super().__init__(T = T, dt = dt, N = N, D = D, lds = lds, lam = lam, t0 = t0)
        
        #compute voltage eq matrices 
        self.Mv = D.T @ lds.A @ np.linalg.pinv(D.T)
        self.Mr = D.T @ (lds.A + lam * np.eye(A.shape[0])) @ D
        self.Mo = - D.T @ D
        self.Mc = D.T @ lds.B
        
        self.t = [t0]
        
        self.vth = np.asarray(
            [1/2 *  D[:,i].T @ D[:,i] for i in np.arange(N)],
            dtype = np.double
            )
        
        self.r  = (np.asarray([np.linalg.pinv(D) @ lds.X0]).T)
        
        
    def V_dot(self):
        u = self.lds.u(self.t[-1])
        assert((self.Mc@u).shape== self.V[:,-1:].shape), "Scaled input has incorrect shape %s" \
        " when state is %s"%((self.Mc@u).shape, self.V[:,-1:].shape)
        
        return self.Mv @ (self.V[:,-1:]) + self.Mr @ (self.r[:,-1:]) + self.Mc @ u 
    


    def r_dot(self):
        '''
        Compute the post synaptic current
        '''
        return -self.lam * self.r[:,-1:]
       
            
    def spike(self, idx): 
        self.V[:,-1] += self.Mo[:,idx]
        self.r[idx,-1] += 1
        self.O[str(idx)].append(self.t[-1])
        
        
    def run_sim(self):
        dt = self.dt
        vth = self.vth
        while self.t[-1] < self.T:
            print('Simulation Time: %f' %self.t[-1])
            spiked = self.V[:,-1] > vth
            
            for idx in np.nonzero(spiked)[0]:
                self.spike(idx)
                break
            self.r = np.append(self.r, self.r[:,-1:] +  dt * self.r_dot(), axis=1 )
            self.V = np.append(self.V, self.V[:,-1:] +  dt * self.V_dot(), axis=1 )
            self.t.append(self.t[-1] + self.dt)
        
        data = {}

        data['r'] = np.asarray(self.r)
        data['V'] = np.asarray(self.V)
        data['t'] = np.asarray(self.t)
        assert(data['r'].shape == data['V'].shape), "r has shape %s but V has shape %s"%(data['r'].shape, data['V'].shape)
        assert(data['V'].shape[1] == len(data['t'])), "V has shape %s but t has shape %s"%(data['V'].shape, data['t'].shape)
        
        data['x_hat'] = self.D @ data['r'] 
        true_data = self.lds.run_sim()
        data['x_true'] = true_data['X']
        data['t_true'] = true_data['t']
        print('Simulation Complete.')
        return data


class SpikeDropDeneveNet(GapJunctionDeneveNet):
    def __init__(self, T, dt, N, D, lds, lam, p, t0 = 0):
        super().__init__(T, dt, N, D, lds, lam, t0)
        self.p = p
        
    def spike(self,idx):
        draw = np.random.binomial(1, self.p, size = (self.N,))
        self.V[:,-1] += np.diag(draw) @ self.Mo[:,idx]
        self.r[idx,-1] += 1
        self.O[str(idx)].append(self.t[-1])
     
'''
Helper functions
'''

 
def exp_err(p, D, l):
    
    vec = p*(1-p) * np.square(D.T @ D[:,l])
    sig_v = np.diag(vec)
    
    d_dag = np.linalg.pinv(D)
    sig_e =  d_dag.T @ sig_v @ d_dag
    return (1- p)**2 + np.trace(sig_e), (1-p)**2, np.trace(sig_e)
    #return 1 + p * (a - 2) + p**2 * (1 - a)

def dissonance(D, V):
    ''' given an N-vector V and decoder matrix D, compute the dissonance'''
    dt_dag = np.linalg.pinv(D.T)
    n_pts = V.shape[1]
    N = V.shape[0]
    diss = np.zeros((n_pts,1))
    
    for i in np.arange(n_pts):
        diss[i,0] = np.linalg.norm( ( D.T @ dt_dag - np.eye(N))@ (V[:,i]) )
        
    return diss / np.sqrt(N)
    
    
def gen_decoder(d, N, mode='random'):
        '''
        Generate a d x N decoder mapping neural activities to 
        state space (X). Only 'random' mode implemented, which
        draws decoders from 0 mean gaussian with std=1
        '''
        if mode == 'random':
            return np.random.normal(loc=0, scale=1, size=(d, N) )
        elif mode == '2d cosine':
            assert(d == 2), "2D Cosine Mode is only implemented for d = 2"
            thetas = np.linspace(0, 2 * np.pi, num=N)
            D = np.zeros((d,N))
            D[0,:] = np.cos(thetas)
            D[1,:] = np.sin(thetas)
            
            return D
      
      
run_sim = False

       
A =  np.zeros((2,2))
A[0,1] = 1
A[1,0] = -1
A =  A
B = np.eye(2)
u0 = np.zeros((A.shape[0]))
x0 = np.asarray([0, 1])


T = 10 
sim_dt = 1e-3
lds_dt = .001
p = .9
#lds = sat.LinearDynamicalSystem(x0, A, u0, B, u = lambda t: 1*np.ones((B.shape[1],1)), T = T, dt = lds_dt)
lds = sat.LinearDynamicalSystem(x0, A, u0, B, u = None, T = T, dt = lds_dt)


plt.close('all')
Ns = np.asarray([10, 20, 50, 100, 500, 1000, 5000])

N = 50
lam =  1
mode = '2d cosine'

cmap = cm.get_cmap('viridis',N)(np.arange(N))

exp_errs = []
biasq = []
vars = []

for N in Ns:
    D = gen_decoder(len(x0), N, mode)
    
    err, bsq, var = exp_err(.9, D, 1)
    exp_errs.append(err)
    biasq.append(bsq)
    vars.append(var)
    
plt.figure()
plt.plot(Ns, exp_errs, label='Expected Error')
plt.plot(Ns, biasq, label= 'Bias Squared')
plt.plot(Ns, vars, label='Variance (trace of cov matrix)')
plt.xlabel('Number of Neurons')
plt.ylabel('Expected Error Norm')
plt.title("Variance inversely scales with N while Bias constant")
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()



ps = np.linspace(0,1, num = 100)
plt.figure()

exp_errs = []
biasq = []
vars = []
for p in ps:
    e, bsq, var = exp_err(p, D, 1 ) 
    exp_errs.append(e)
    biasq.append(bsq)
    vars.append(var)

plt.plot(ps,exp_errs,label='expected error')
plt.plot(ps,biasq,label='bias squared')
plt.plot(ps,vars,label='variance')

plt.legend()


#plt.figure()
plt.show()
net = GapJunctionDeneveNet(T=T, dt=sim_dt, N=N, D=D, lds=lds, lam=lam, t0 = 0)
#net = SpikeDropDeneveNet(T=T, dt=sim_dt, N=N, D=D, lds=lds, lam=lam,p = p, t0 = 0)




if run_sim:
    data = net.run_sim() 
    
    sim_diss = dissonance(D, data['V'])
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(data['t_true'], data['x_true'][0,:],label='x')
    ax1.plot(data['t'],  data['x_hat'][0,:],label=r'$\hat{x}$')
    ax2.plot(data['t'], sim_diss,label='Dissonance')
    plt.title("xhat vs xtrue") 
    plt.legend()
    
     
    plt.figure()
    plt.plot(data['t'], np.mean(data['V'],axis=0),linewidth=5)
    plt.title('mean(V(t))')
      
    plt.figure()
    for i in np.arange(N):
        #if np.max(data['r'][i,:]) > .25:
        plt.plot(data['t'],data['r'][i,:],label='i = %i'%i)
    plt.title('r(t)')
    
    
    plt.show()