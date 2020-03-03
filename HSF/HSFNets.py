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
from scipy.integrate._ivp.ivp import solve_ivp
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
        
        ## Spike times, where simulation stops & updates state
        # used for interpolating r(t)
        self.t0 = t0
        
        
        self.O = {}
        for i in np.arange(self.N):
            self.O[str(i)] = np.empty((1,))
        
        V0 = np.zeros((N,))
        
        super().__init__(V0, T, dt, t0)
    
    
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
        
        self.vth = np.asarray(
            [1/2 *  D[:,i].T @ D[:,i] for i in np.arange(N)],
            dtype = np.double
            )
         
        #self.X0 = np.abs(np.random.normal(1,1,(N,)))
        self.r0s = np.abs(np.asarray([np.linalg.pinv(D) @ lds.X0]).T)
        
  
    
    def spike_occurred_i(self, t, V, i):
        '''
        return the difference between neuron i's voltage & threshold
        threshold here is norm so is alwyas nonnegative 
        '''
        
        return V[i] - self.vth[i]
        
    
    def spike_occurred(self):
        
        func_list = [] 
        for i in np.arange(N):
            func = lambda t, V, j = i: self.spike_occurred_i(t, V, j)
            func.terminal = True
            func.direction = 1
            func_list.append(func)
        
        return func_list
            
        
    def deriv(self, t, V):
        ''' 
        \dot{V} = Mv V + Mr r + Mo o + Mc c
        Mo o  (fast update) is implemented by spike function
        c comes from attached linear dynamical system
        v is state variable
        r is convolution of impulse response psi with spike raster 
        '''
        return  self.Mv @ V + self.Mr @ self.r(t) + self.Mc @ self.lds.u(t) 
        
        
    def r(self,ts):
        '''
        Compute Post synaptic current/filtered spike train of N neurons at time t.
        Assumes an initial rate at time 0 r0 is an attribute of self.         
        '''
              
        # if ts is int, just return r(t0)
        
        # else return r(t) as N x len(ts) matrix  
        def r_helper(self, t):
            try:
                assert(t - self.t0s[-1] >= 0), "Time since last spike t-t0 should be positive but was %f"%(t-self.t0s[-1])
                t0 = self.t0s[-1]
                r0 = self.r0s[:,-1]
            except AssertionError:
                t0 = self.t0s[-2]
                r0 = self.r0s[:,-2]
            
            return np.exp(-self.lam * (t-t0)) * r0    
        
        
        try:
            rs = np.zeros((self.N, len(ts)))
            
            for idx, t in enumerate(ts):
                rs[:,idx] = r_helper(self, t)
        
        except TypeError:
            rs = r_helper(self,ts)

        return rs

            
    def spike(self, V, t_spike): 
        
        O_t = self.get_spiked_neurons(V, single_spike = True)
        O_t = np.abs(V - self.vth) >= 1e-15
        diffs = V - self.vth
        
        SpikingNeuralNet.spike(self, V, O_t, )
        assert(np.sum(O_t) > 0), "Simulation detected spike but no neurons had V >= thresh.  Max was %f"%np.max(diffs)
        self.r0s = np.hstack((self.r0s, np.expand_dims(self.r(t_spike-self.t0s[-1]) + O_t, axis = 1)))
        spike_idx = np.nonzero(O_t)[0]       
        V += self.Mo@O_t
        #O_t[spike_idx] = 0

 #       assert(np.max(V - self.vth) < 0), "Voltage above threshold after spiking"
            
        
            
        self.t0s = np.append(self.t0s, t_spike)

    
  
        
    def run_sim(self):
        data = super().run_sim()        
        data['x_hat'] = self.D @ data['r'] 
        true_data = self.lds.run_sim()
        data['x_true'] = true_data['X']
        data['t_true'] = true_data['t']
        print('Simulation Complete.')
        return data


    def get_spiked_neurons(self, X0, single_spike):
        '''
        Return O(t) which has entries 1 if neuron j should spike at given state
        '''
    
        O_t = np.isclose(X0, self.vth) +  np.abs(X0) - self.vth >= 0
        
        
        if single_spike:
            max_spike = np.argmax(X0 - self.vth)
            O_t = np.zeros_like(O_t)
            O_t[max_spike] = 1 
        
            
        return O_t
'''
Helper functions
'''
    
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
             
A =  np.zeros((2,2))
A[0,1] = 1
A[1,0] = -1
A = .1 * A
B = np.eye(2)
u0 = np.zeros(A.shape[0])
x0 = np.asarray([0, 1])


T = 5
sim_dt = 1e-5
lds_dt = .001
lds = sat.LinearDynamicalSystem(x0, A, u0, B, u = lambda t: 1*np.ones((B.shape[1],)), T = T, dt = lds_dt)

N = 50
lam =  10
mode = '2d cosine'

cmap = cm.get_cmap('viridis',N)(np.arange(N))


D = gen_decoder(len(x0), N, mode)

net = GapJunctionDeneveNet(T=T, dt=sim_dt, N=N, D=D, lds=lds, lam=lam, t0 = 0)

data = net.run_sim() 
meane = np.mean(np.linalg.pinv(D.T)@data['V'],axis=0)    


# print(np.min(np.diff(data['t'])))
# print(np.max(np.diff(data['t'])))
plt.figure()
plt.plot(data['t_true'],data['x_true'][0,:])
plt.plot(data['t'], data['x_hat'][0,:])
plt.title("xhat vs xtrue") 
#



 
plt.figure()
plt.plot(data['t'],meane,linewidth=5)
plt.title('mean(V(t))')
#plt.axhline(y = net.vth[i],c=cmap[i],ls = '--',linewidth=5)
#plt.plot(data['x_hat'][0,:], data['x_hat'][1,:])
# plt.plot(data['x_true'][0,:], data['x_true'][1,:])
# # 
# 
# 
# 
#  
plt.figure()
for i in np.arange(N):
    #if np.max(data['r'][i,:]) > .25:
    plt.plot(data['t'],data['r'][i,:],label='i = %i'%i)
plt.title('r(t)')
#plt.legend()


plt.show()