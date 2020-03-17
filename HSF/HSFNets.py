'''
Created on Feb 19, 2020

@author: cbfritz

Class definitions for Spiking Neural Networks Simulations
'''
import Simulation_Analysis_Toolset as sat 
import numpy as np
from abc import abstractmethod
import ctypes as ct
import matplotlib.pyplot as plt #@UnusedImport squelch
from numba import jit
'''
Class Definitions
'''
   
#load c solver
c_lib = ct.CDLL('c_src/c_solver.so')
    
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
        self.suppress_console_output = False
        
        self.O = {}
        for i in np.arange(self.N):
            self.O[str(i)] = []
        
        self.num_pts = int(np.floor(((T - t0)/dt))) 
        
        self.V = np.zeros((N,self.num_pts))
        self.r = np.zeros_like(self.V)
        self.t = np.zeros((self.num_pts,))
        super().__init__(None, T, dt, t0)
        
        self.inject_noise = (False, None) # to add voltage noise, set this to (true, func(net)) where func is a handle 
                                          # and net is this network object 

    
    def deriv(self, t, X):
        sat.DynamicalSystemSim.deriv(self, t, X)
        pass
    
    
    def decoherence(self, V):
        ''' 
        Given an N-t vector V and decoder matrix D,
        compute the decoherence.
        '''
    
        D = self.D
        
        dtd = np.linalg.pinv(D.T @ D)
        I = np.eye((V.shape[0]))
        
        dec = np.zeros(V.shape)
        for i in np.arange(dec.shape[1]):
            dec[:,i] = (dtd - I) @ V[:,i]
            
        return dec

    
    def pack_data(self, final=True):
        ''' Pack simulation data into dict object '''
        data = super().pack_data()
        
        data['r'] = np.asarray(self.r)
        data['V'] = np.asarray(self.V)
        data['t'] = np.asarray(self.t)
        data['O'] = self.O
        assert(data['r'].shape == data['V'].shape), "r has shape %s but V has shape %s"%(data['r'].shape, data['V'].shape)
        assert(data['V'].shape[1] == len(data['t'])), "V has shape %s but t has shape %s"%(data['V'].shape, data['t'].shape)
        
        data['x_hat'] = self.D @ data['r'] 
        data['x_true'] = data['lds_data']['X']
        data['t_true'] = data['lds_data']['t']
        data['dec'] = self.decoherence(data['V'])
        
        #compute error trajectory, e = dtdag @ V
        if final:
            num_pts = len(data['t_true'])
            errs = np.zeros((data['A'].shape[0],num_pts))
            dt_dag = np.linalg.pinv(data['D'].T)
            for i in np.arange(num_pts):
                errs[:,i] = dt_dag @ data['V'][:,i]
                
            data['error'] = errs
        return data
    
    @abstractmethod  
    def run_sim(self):
        ''' Return the packed simulation data'''
        return self.pack_data()
    
    
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
    
    def __init__(self, T, dt, N, D, lds, lam, t0 = 0, thresh = 'full'):
        super().__init__(T = T, dt = dt, N = N, D = D, lds = lds, lam = lam, t0 = t0)
        
        #compute voltage eq matrices 
        self.Mv = D.T @ lds.A @ np.linalg.pinv(D.T)
        self.Mr = D.T @ (lds.A + lam * np.eye(lds.A.shape[0])) @ D
        self.Mo = - D.T @ D
        self.Mc = D.T @ lds.B
        
        
        
        
        self.vth = np.asarray(
            [ D[:,i].T @ D[:,i] for i in np.arange(N)],
            dtype = np.double
            )
        
        if thresh is 'full':
            self.vth = self.vth * .5
            
        
        self.r[:,0:]  = (np.asarray([np.linalg.pinv(D) @ lds.X0]).T)
        
        
    def V_dot(self):
        '''implemented in jit in run_sim'''
        u = self.lds.u(self.t[-1])
        if (u.ndim == 1):
            u = np.expand_dims(u, axis = 1)
        assert((self.Mc@u).shape== self.V[:,-1:].shape), "Scaled input has incorrect shape %s" \
        " when state is %s"%((self.Mc@u).shape, self.V[:,-1:].shape)
        
        if self.inject_noise[0]:
            noise = self.inject_noise[1](self) / self.dt
        else:
            noise = np.zeros((self.N,1))
        
        return self.Mv @ (self.V[:,-1:]) + self.Mr @ (self.r[:,-1:]) + self.Mc @ u + noise 
    
    
    def r_dot(self):
        '''
        implemented in jit in run_sim
        '''
        pass
      
                
    def spike(self, idx):
        '''implemented using jit in run_sim'''
        pass
       
        
    def run_sim(self): 
        '''
        process data and call c_solver library to quickly run sims
        '''
        @jit(nopython=True)     
        def fast_sim(dt, vth, num_pts, t, V, r, Mv, Mo, Mr, Mc,  lam):
            '''run the sim quickly using numba just-in-time compilation'''
            #run sim
            for count in np.arange(num_pts):
                #get largest voltage above threshold
                diff = V[:,count] - vth
                max_v = np.max(diff)
                if max_v >= 0:
                    idx = np.argmax(diff) 
                    V[:,count] += Mo[:,idx]
                    r[idx,count] += 1
                    #O[str(idx)].append(t[-1])
                    
                r_dot = -lam * r[:,count]   
                v_dot = Mv @ (V[:,count]) + Mr @ (r[:,count])    
                
                r[:,count+1] = r[:,count] +  dt * r_dot
                V[:,count+1] = V[:,count] +  dt * v_dot
                t[count] = dt
                count += 1
                
                
        
        self.lds_data = self.lds.run_sim()
        U = lds_data['U']
        vth = self.vth
        num_pts = self.num_pts
        dt = self.dt
        t = np.asarray(self.t)
        V = self.V
        r = self.r
        Mv = self.Mv
        Mo = self.Mo
        Mr = self.Mr
        Mc = self.Mc
        lam  = self.lam
        
    
        fast_sim(dt, vth, num_pts, t, V, r, Mv, Mo, Mr, Mc, lam)
        print(V.shape)

        #self.O = O

        if not self.suppress_console_output:
            print('Simulation Complete.')
        
        return super().run_sim()


class SpikeDropDeneveNet(GapJunctionDeneveNet):
    
    def __init__(self, T, dt, N, D, lds, lam, p, t0 = 0, thresh = 'not full'):
        super().__init__(T, dt, N, D, lds, lam, t0, thresh)
        self.p = p
        self.seed = 0
           
    def spike(self,idx):
        np.random.seed(self.seed)
        draw = np.random.binomial(1, self.p, size = (self.N,))
        self.V[:,-1] += np.diag(draw) @ self.Mo[:,idx]
        self.r[idx,-1] += 1
        self.O[str(idx)].append(self.t[-1])
        self.seed += 1
     
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
            D = np.zeros((d,N), dtype = np.float64)
            D[0,:] = np.cos(thetas)
            D[1,:] = np.sin(thetas)
            
            return D
    
    
    
      
A =  np.zeros((2,2))
A[0,1] = 1
A[1,0] = -1
A = 10 * A
N = 128
lam =  5
mode = '2d cosine'
p = 1

    
D = gen_decoder(A.shape[0], N, mode=mode)
B = np.eye(2)
u0 = .001*D[:,0]
x0 = np.asarray([0, 1])
T = 10
dt = 1e-5

lds = sat.LinearDynamicalSystem(x0, A, u0, B, u = lambda t: u0 , T = T, dt = dt)


net = GapJunctionDeneveNet(T=T, dt=dt, N=N, D=D, lds=lds, lam=lam, t0=0, thresh = 'not full')


data = net.run_sim()

plt.plot(data['x_hat'][0,:],label='xhat')
plt.plot(data['x_true'][0,:],label='xtru')
plt.legend()

plt.figure()
plt.plot(np.max(data['V']))
plt.show()
