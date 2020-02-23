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

'''
Class Definitions
'''

class SpikingNeuralNet(sat.DynamicalSystemSim):
    ''' 
    A SpikingNeuralNet object is used to simulate a spiking neural network that
    implements a given linear dynamical system. This is an abstract class used to
    define general properties common to all spiking neural nets.
    '''
    
    def __init__(self, T, dt, N, D, lds, lam, V0 = .1, t0 = 0):
         
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
            self.O[str(i)] = np.empty((1,))
        
        V0s = np.random.normal(loc=0,scale=V0, size = (N,))
        r0s = np.zeros_like(V0s)
        
        #1st half of state is Vs, 2nd Half is rs (post synaptic currents)
        X0 = np.append(V0s, r0s)
        
        super().__init__(X0, T, dt, t0)
    
    
    def run_sim(self):
        ts = None
        t_events = None
        X = None
        X0 = self.X0
        t0 = self.t0
        while t0 < T-self.dt:
            print("Simulation Time: %f"%t0)
            data = solve_ivp(
                self.deriv, # Derivative function
                (t0, self.T), # Total time interval
                X0, # Initial State
                t_eval=np.linspace(t0, self.T, int(self.T/self.dt)),  
                method='LSODA', 
                dense_output=False, 
                events=self.spike_occurred,
                rel_tol = 10e-10,
                abs_tol = 10e-15,
                max_step = 10e-15
                )
            if X is None:
                X = data.y
                ts = data.t
                t_events = data.t_events
            else:
                X = np.append(X, data.y, axis = 1)
                ts = np.append(ts, data.t )
                t_events = np.append(t_events, data.t_events,axis=1)
            
            
            if data.status == 1: # if spike threshold crossed, spike
                t0 = data.t[-1] + dt
                X0 = data.y[:, -1].copy()
                O_t = self.spike_occurred(data.t[-1], X0, ret_idx=True)
                self.spike(X0, O_t, data.t[-1])
            else:
                break
    
        sim_data = {}
        for param in self.__dict__.keys():
            sim_data[str(param)] = self.__dict__[param]
        
        sim_data['V'] = X[0:self.N,:]
        sim_data['r'] = X[self.N:,:]
        sim_data['t'] = ts
        sim_data['t_events'] = np.asarray(t_events)
        return sim_data
    
    @abstractmethod
    def deriv(self, t, X):
        pass
    
    @abstractmethod
    def spike_occurred(self, t, y, ret_idxs = False):
        '''
        Based on State & Time decide whether neuron(s) spikes. 
        Used by root finding algorithm to determine when neuron voltage 
        v- thresh > 0, i.e when a neuron crosses threshold voltage.
        ret_idxs is set to true means that the index of any spiking neuron will
        be returned.
        '''

     
    @abstractmethod
    def spike(self, V, O_t, t):
        ''' 
        update neuron voltage V with spikes occuring at O_t,
        O_t is a N-vector of 1s (spike at time t) and zeros (no spike)
        '''
        assert( len(O_t) <= self.N)


class GapJunctionDeneveNet(SpikingNeuralNet):
    ''' Spiking Net According to Classic Deneve Paper w/ Erics Voltage Modification'''
    
    def __init__(self, T, dt, N, D, lds, lam, V0 = 0, t0 = 0):
        super().__init__(T = T, dt = dt, N = N, D = D, lds = lds, lam = lam, V0 = V0, t0 = t0)
        
        #compute voltage eq matrices 
        self.Mv = D.T @ lds.A @ np.linalg.pinv(D.T)
        self.Mr = D.T @ (lds.A + lam * np.eye(A.shape[0])) @ D
        self.Mo = - D.T @ D
        self.Mc = D.T @ lds.B
        
        self.vth = np.asarray(
            [(1 / 2) * D[:,i].T @ D[:,i] for i in np.arange(N)]
            )
         
        
        
    def spike_occurred(self, t, X, ret_idx = False):
        '''
        Find the neuron closest to crossing threshold from below.
        '''
        V = X[0:self.N]
        diffs = V - self.vth
        
        
        if ret_idx:
            #get each neuron at min val 
            O_t = np.zeros((self.N,))
            O_t[V - self.vth > 0] = 1
            return O_t
        else:
            if np.max(diffs) > 0:
                return 0
            else:
                return np.max(diffs)
        
    spike_occurred.terminal = True # Set both attributes for use in event-based spike detection
    spike_occurred.direction = 1
    
    def deriv(self, t, X):
        ''' 
        \dot{V} = Mv V + Mr r + Mo o + Mc c
        Mo o  (fast update) is implemented by spike function
        c comes from attached linear dynamical system
        v is state variable
        r is convolution of impulse response psi with spike raster 
        '''
        N = self.N
        
        V = X[0:N]
        r = X[N:]
        
        dV = self.Mv @ V + self.Mr @ r + self.Mc @ self.lds.u(t)
        dr = -self.lam * r
        
        return np.append(dV, dr)
    
    
    def spike(self, X, O_t, t, single_spike_only=True):
        
        if single_spike_only:
            spike_idx = np.random.choice(np.argwhere(O_t==1)[0])
            O_t = np.zeros_like(O_t)
            O_t[spike_idx] = 1        
        
        V = X[0:self.N]
        r = X[self.N:]
        SpikingNeuralNet.spike(self, V, O_t, t)
        V += self.Mo @ O_t
        r += O_t
        
        X = np.vstack((V,r))
        spikes = np.argwhere(O_t == 1)
        for sp in spikes:
            self.O[str(sp[0])] = np.append(self.O[str(sp[0])], t) 
        
    def run_sim(self):
        data = SpikingNeuralNet.run_sim(self)
        data['x_hat'] = self.D @ data['r']
        true_data = self.lds.run_sim()
        data['x_true'] = true_data['X']
        return data

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
dt = .01
lds = sat.LinearDynamicalSystem(x0, A, u0, B, u = None, T = T, dt = dt)

N = 100
lam = 100000
mode = '2d cosine'
#mode = 'gaussian'
D = gen_decoder(len(x0), N)

net = GapJunctionDeneveNet(T=T, dt=dt, N=N, D=D, lds=lds, lam=lam, V0 = .1, t0 = 0)

data = net.run_sim() 
    

#plt.figure()
#plt.plot(data['t'], data['V'][0,:])
#plt.plot(data['t'], data['x_true'][0,:])


# plt.figure()
#plt.plot(data['x_hat'][0,:], data['x_hat'][1,:])
#plt.plot(data['x_true'][0,:], data['x_true'][1,:])

plt.figure()
for i in np.arange(N):
    if np.max(data['r'][i,:]) > 0:
        plt.plot(data['t'],data['r'][i,:],label='%i'%i)
plt.legend()


plt.show()