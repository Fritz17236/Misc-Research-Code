'''
Created on Feb 19, 2020

@author: cbfritz

Class definitions for Spiking Neural Networks Simulations
'''

import Simulation_Analysis_Toolset as sat
import numpy as np
import matplotlib.pyplot as plt
from abc import abstractmethod
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
    
    
    def dissonance(self, V):
        ''' 
        Given an N-t vector V and decoder matrix D,
        compute the dissonance.
        '''
        n_pts = V.shape[1]
        N = V.shape[0]
        D = self.D
        
        dtd = np.linalg.pinv(D.T @ D)
        I = np.eye((N))
        
        diss = np.zeros((N,n_pts))
        for i in np.arange(n_pts):
            diss[:,i] = (dtd - I) @ V[:,i]
            
        return diss
    
    
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
        self.Mr = D.T @ (lds.A + lam * np.eye(lds.A.shape[0])) @ D
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
        data['O'] = self.O
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
        self.readout_mask = np.ones((N,))
        
    def spike(self,idx):
        draw = np.random.binomial(1, self.p, size = (self.N,))
        self.V[:,-1] += np.diag(draw) @ self.Mo[:,idx]
        self.r[idx,-1] += 1
        self.readout_mask = draw
        self.O[str(idx)].append(self.t[-1])
     
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