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
import copy
'''
Class Definitions
'''

##bug: pingponging all spikes go simultaneously and lead to exponential growth/error 
    # try: when a spike occurrs, make a queue and apply the first spike, while there are spikes in queue, spike
        # maybe mad eit worse
        
    # only one psike then skip 
        # kind of works but wierd sidewise travelling of voltage curves 
    # use smaller dt 10e-4 instead of 10e-3
        # meh, no improvement when lower max step 
    # make max step 10e-3 with dt small
        # takes too long for too small dt
        
    #try: make time added after spike very small (eg 10e-6)
        # with max step off is horrible
        # with max step 10e-3 tspan teval error, address:
            # changed teval but horrible 
        

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
        self.t0s= [0]
        
        
        self.O = {}
        for i in np.arange(self.N):
            self.O[str(i)] = np.empty((1,))
        
        X0 = np.zeros((N,))
        
        super().__init__(X0, T, dt, t0)
    
    
    def run_sim(self):
        ts = None
        t_events = None
        X = None
        X0 = self.X0
        t0 = self.t0
        
        while t0 < self.T - self.dt:
            print("Simulation Time: %f"%t0)
            t_eval = np.linspace(t0, self.T, num =int((self.T-t0)/self.dt))
            data = solve_ivp(
                self.deriv, # Derivative function
                (t0, self.T), # Total time interval
                X0, # Initial State  
                t_eval = t_eval,
                method='LSODA', 
                dense_output=True, 
                events=self.spike_occurred(),
             #   rtol = 10e-10,
             #   atol = 10e-15,
                max_step = 10e-3,
                )
   
            if X is None: 
                X = data.y[:,:-2]
                ts = data.t[:-2]
                t_events = data.t_events
            
            else:

                X = np.hstack((X, data.y[:,:-2]))
                ts = np.append(ts, data.t[:-2])
                t_events = np.append(t_events, data.t_events)   
         
            
            if data.status == 1: # if spike threshold crossed, spike
                #find spike in t_events list
                t_spike = [l[0] for l in data.t_events if len(l) is not 0][0]
                X0 = data.sol(t_spike).copy()
                t0 = self.spike(X0, t_spike, t0)
                
                  
                
            else:
                
                t0 = self.T

        sim_data = {}
        for param in self.__dict__.keys():
            sim_data[str(param)] = self.__dict__[param]
        
        sim_data['V'] = X
        sim_data['t'] = ts
        sim_data['t_events'] = np.asarray(t_events)
        return sim_data
    
    @abstractmethod
    def deriv(self, t, X):
        pass


    @abstractmethod
    def r(self, ts):
        '''
        Compute the post synaptic current
        '''
        pass
    
    @abstractmethod
    def spike_occurred(self, t, y):
        '''
        Based on State & Time decide whether neuron(s) spikes. 
        Used by root finding algorithm to determine when neuron voltage 
        v- thresh > 0, i.e when a neuron crosses threshold voltage.
        ret_idxs is set to true means that the index of any spiking neuron will
        be returned.
        '''

    @abstractmethod
    def get_spiked_neurons(self, X0):
        '''
        Return O(t) which has entries 1 if neuron j should spike at given state
        '''
        pass
     
    @abstractmethod
    def spike(self, V, O_t, t):
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
            [(1 / 2) * D[:,i].T @ D[:,i] for i in np.arange(N)]
            )
         
        self.r0s = np.asarray([np.linalg.pinv(D) @ lds.X0]).T
        
  
    
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
            #func.direction = 1
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
        
        return  self.Mv @ V + self.Mr @ self.r(t) + self.Mc @ self.lds.u(t)# + np.random.normal((self.N,))
        
        
    def r(self,ts):
        '''
        Compute Post synaptic current/filtered spike train of N neurons at time t.
        Assumes an initial rate at time 0 r0 is an attribute of self.         
        '''
              
              
              
        # if ts is int, just return r(t0
        
        # else return r(t) as N x len(ts) matrix  
        def r_helper(self, t):
            return np.exp(-self.lam * t) * self.r0s[:,-1]    
        
        
        try:
            rs = np.zeros((self.N, len(ts)))
            
            for idx, t in enumerate(ts):
                rs[:,idx] = r_helper(self, t)
        
        except TypeError:
            rs = r_helper(self,ts)

        return rs

##TODO
#debug: check r0 over time. make it an array
         
            
    def spike(self, V, t_spike, t0): 
             
        O_t = self.get_spiked_neurons(V)    
        if np.sum(O_t) > 0: 
            SpikingNeuralNet.spike(self, V, O_t, t_spike)
            V += self.Mo @ O_t
            self.r0s = np.hstack((self.r0s, np.expand_dims(self.r(t_spike-t0) + O_t, axis = 1)))
            t0 += (10e-4)
            O_t = self.get_spiked_neurons(V)
            
            self.t0s.append(t_spike)

        return t0
        
        
    def run_sim(self):
        data = SpikingNeuralNet.run_sim(self)
        
        #compute rs by interpolation r0/t0s at desired time points
        data['r'] = np.zeros((self.N, len(data['t'])))
        for i in np.arange(self.N):
            data['r'][i,:] = np.interp(data['t'],data['t0s'],data['r0s'][i,:])
            
        data['x_hat'] = self.D @ data['r'] 
        true_data = self.lds.run_sim()
        data['x_true'] = true_data['X']
        print('Simulation Complete.')
        return data


    def get_spiked_neurons(self, X0):
        '''
        Return O(t) which has entries 1 if neuron j should spike at given state
        '''
    
        O_t = [1 if np.isclose(X0[i], self.vth[i]) else 0 for i in np.arange(N)]
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
B = np.zeros(A.shape)
u0 = np.zeros(A.shape[0])
x0 = np.asarray([1, 0])
T = 5
dt = .0001
lds = sat.LinearDynamicalSystem(x0, A, u0, B, u = None, T = T, dt = dt)

N = 50
lam =  dt
mode = '2d cosine'

cmap = cm.get_cmap('viridis',N)(np.arange(N))


D = gen_decoder(len(x0), N,mode)

net = GapJunctionDeneveNet(T=T, dt=dt, N=N, D=D, lds=lds, lam=lam, t0 = 0)

data = net.run_sim() 
    


# print(np.min(np.diff(data['t'])))
# print(np.max(np.diff(data['t'])))
plt.figure()
plt.plot(data['x_true'][0,:])
plt.plot(data['x_hat'][0,:]) 
#

idxs = np.arange(0,N, 10) 
plt.figure()
for i in idxs:
    plt.plot(data['t'],data['V'][i,:],c=cmap[i],linewidth=5)
    plt.axhline(y = net.vth[i],c=cmap[i],ls = '--',linewidth=5)
#plt.plot(data['x_hat'][0,:], data['x_hat'][1,:])
# plt.plot(data['x_true'][0,:], data['x_true'][1,:])
# # 
# 
# 
# 
#  
plt.figure()
for i in np.arange(N):
    if np.max(data['r'][i,:]) > .25:
        plt.plot(data['t'],data['r'][i,:],label='i = %i'%i)
#plt.legend()


plt.show()