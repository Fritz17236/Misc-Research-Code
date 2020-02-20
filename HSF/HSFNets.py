'''
Created on Feb 19, 2020

@author: cbfritz

Class definitions for Spiking Neural Networks Simulations
'''

import Simulation_Analysis_Toolset as sat
import numpy as np
import matplotlib.pyplot as plt
from abc import abstractmethod



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

def event(t, y):
    ''' blah'''
    
    thresh = .5
    return y[0] - thresh

# event based state updating
# if event occurs (x > thresh), record event, and move x += delta
event.direction = 1
data = lds.run_sim(event)
plt.plot(data['t'],data['X'][0,:])
plt.axvline(x = data['t_events'][0][0])
plt.show()