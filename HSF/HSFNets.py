'''
Created on Feb 19, 2020

@author: cbfritz

Class definitions for Spiking Neural Networks Simulations
'''

import Simulation_Analysis_Toolset as sat
import numpy as np
import matplotlib.pyplot as plt
from abc import abstractmethod
from scipy.integrate._ivp.ivp import solve_ivp



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
        
        self.O = [np.asarray([]) for i in np.arange(self.N)] #spike raster is list of empty lists
        super().__init__(lds.X0, T, dt, t0)
    
    
    @abstractmethod
    def deriv(self, t, X):
        pass
    @abstractmethod
    def spike(self, indices):
        ''' 
        Given an array of indices representing neurons, update
        each neuron at indices[i] according to the spike rule
        '''
        assert( len(indices) <= self.N)
    # r(t) leaky firing rates computed from o(t)  
    
    # run -- how? 


def gen_decoder(d, N, mode='random'):
        '''
        Generate a d x N decoder mapping neural activities to 
        state space (X). Only 'random' mode implemented, which
        draws decoders from 0 mean gaussian with std=1
        '''
        return np.random.normal(loc=0, scale=1, size=(d, N) )

A =  np.zeros((2,2))
A[0,1] = 1
A[1,0] = -1
B = np.zeros(A.shape)
u0 = np.zeros(A.shape[0])
x0 = np.asarray([1, 0])
T = 10
dt = .001
lds = sat.LinearDynamicalSystem(x0, A, u0, B, u = None, T = T, dt = dt)

N = 1000
lam = 1
D = gen_decoder(len(x0), N)

#net = SpikingNeuralNet(T, dt, N, D, lds, lam)

def run_sim(self, event=None):
    ts = np.empty((1,))
    t_events = np.empty((1,))
    X = np.empty_like(self.X0)
    X0 = self.X0
    t0 = self.t0
    while True:
        data = solve_ivp(
            self.deriv, # Derivative function
            (t0, self.T), # Total time interval
            X0, # Initial State
            t_eval=np.arange(t0, self.T, self.dt),  
            method='LSODA', 
            dense_output=True, 
            events=event
            )
        X = np.append(X, data.y, axis = 1 )
        ts = np.append(ts, data.t )
        #t_events = np.append(data.t_events,axis=1)
        
        
        if event and data.status == 1: # Event was hit
            # New start time for integration
            t0 = data.t[-1]
            # Reset initial state
            X0 = data.y[:, -1].copy()
            X0[0] += 1
        else:
            break
        
    

    sim_data = {}
#     for param in self.__dict__.keys():
#         sim_data[str(param)] = self.__dict__[param]
    
    sim_data['X'] = X
    sim_data['t'] = ts
    sim_data['t_events'] = np.asarray(t_events)
    return sim_data
     


def event(t, y):
    ''' blah'''
    
    thresh = .5
    return y[0] - thresh

# event based state updating
# if event occurs (x > thresh), record event, and move x += delta
event.direction = 1
event.terminal = True
lds.run_sim = run_sim
data = lds.run_sim(lds, event)


print(data.keys())
print(data['X'][0].shape)
#plt.plot(data['t'],data['X'][0,:])
#plt.axvline(x = data['t_events'][0][0])
plt.show()