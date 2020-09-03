'''
Created on Feb 19, 2020

@author: cbfritz

Class definitions for Spiking Neural Networks Simulations
'''
import Simulation_Analysis_Toolset as sat
 
import numpy as np
from abc import abstractmethod
import matplotlib.pyplot as plt #@UnusedImport squelch
import numba
from numba import jit
from scipy.optimize import nnls
from scipy.linalg import expm
from utils import pad_to_N_diag_matrix
import warnings


'''
Helper Functions
'''

@jit(nopython=True)
def fast_sim(dt, vth, num_pts, t, V, r, O, U,  Mo, Mc, A_exp, spike_trans_prob, spike_nums, floor_voltage=False, one_spike_per_step=True):
    '''run the sim quickly using numba just-in-time compilation'''
    N = V.shape[0]
    max_spikes = len(O[0,:])
    
    
    for count in np.arange(num_pts-1):

        state = np.hstack((V[:,count], r[:,count]))
        state = A_exp @ state

        r[:,count+1] = state[N:]
        V[:,count+1] = state[0:N] +  dt * Mc @ U[:,count]
        
        if floor_voltage:
            for i in np.arange(N):
                if V[i,count+1] < -vth[i]:
                    V[i,count+1] = -vth[i]
            
        t[count+1] =  t[count] + dt 
        count += 1
        
        # spike any neurons
        diffs = V[:,count] - vth
        if one_spike_per_step:
            if np.any(diffs > 0):
                idx = np.argmax(diffs > 0)
                
                
                V[:,count] +=   Mo[:,idx]
                
                if spike_trans_prob == 1:
                    r[idx,count] +=  1
                else:
                    sample = np.random.uniform(0, 1)
                    if spike_trans_prob >= sample:
                        r[idx,count] += 1
                    
                spike_num = spike_nums[idx]
                if spike_num >= max_spikes:
                    print("Maximum Number of Spikes Exceeded in Simulation  for neuron ",
                           idx,
                           " having ", spike_num, " spikes with spike limit ", max_spikes)
                    assert(False)
                    
                O[idx,spike_num] = t[count]
                spike_nums[idx] += 1
        else:
            for idx in range(V.shape[0]):
                if diffs[idx] > 0:
                    V[:,count] +=   Mo[:,idx]
                    
                    if spike_trans_prob == 1:
                        r[idx,count] +=  1
                    else:
                        sample = np.random.uniform(0, 1)
                        if spike_trans_prob >= sample:
                            r[idx,count] += 1
                        
                    spike_num = spike_nums[idx]
                    if spike_num >= max_spikes:
                        print("Maximum Number of Spikes Exceeded in Simulation  for neuron ",
                               idx,
                               " having ", spike_num, " spikes with spike limit ", max_spikes)
                        assert(False)
                        
                    O[idx,spike_num] = t[count]
                    spike_nums[idx] += 1
                
             
warnings.simplefilter('ignore', category = numba.errors.NumbaPerformanceWarning) # Execute after declaring fast_sim to prevent Numba Warning about contiguous arrays     


''' 
Class Definitions
'''
class SpikingNeuralNet(sat.DynamicalSystemSim):  
    ''' 
    A SpikingNeuralNet object is used to simulate a spiking neural network that
    implements a given linear dynamical system. This is an abstract class used to
    define general properties common to all spiking neural nets.
    '''
    
    FLOAT_TYPE = np.float64
    
    def __init__(self, T, dt, N, D, lds, t0 = 0):
         
    # neural net needs a linear dynamical system to implement (x' = Ax + Bu)
        self.A = lds.A  # Dynamics Matrix
        self.B = lds.B  # Input Matrix
        self.u = lds.u # Input Signal (function handle with at least t argument)
        self.lds = lds # save lds for later if needed        
        self.N = N # number of neurons in network
        self.D = ( D ).astype(SpikingNeuralNet.FLOAT_TYPE) # Decoder matrix
        
        self.num_pts = int(np.floor(((T - t0)/dt))) 
        self.V = np.zeros((N,self.num_pts), dtype=SpikingNeuralNet.FLOAT_TYPE)
        self.r = np.zeros_like(self.V, dtype=SpikingNeuralNet.FLOAT_TYPE)
        self.t = np.zeros((self.num_pts,), dtype=SpikingNeuralNet.FLOAT_TYPE)
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
        
        dec = np.zeros(V.shape, dtype=SpikingNeuralNet.FLOAT_TYPE)
        for i in np.arange(dec.shape[1]):
            dec[:,i] = (dtd - I) @ V[:,i]
            
        return dec
      
    def set_initial_rs(self, D, X0):
        '''
        Use nonnegative least squares to set initial PSC/r values
        '''
        r0 = nnls(D, X0)[0]
        self.r[:,0]  = np.asarray(r0)
    
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
            errs = np.zeros((data['A'].shape[0],num_pts), dtype=SpikingNeuralNet.FLOAT_TYPE)
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
    
    def __init__(self, T, dt, N, D, lds, t0 = 0, spike_trans_prob = 1):
        super().__init__(T = T, dt = dt, N = N, D = D, lds = lds, t0 = t0)
        
        assert(spike_trans_prob <= 1 and spike_trans_prob >= 0), "Spike transmission probability = {0} is not between 0 and 1".format(spike_trans_prob)
        self.spike_trans_prob=spike_trans_prob
        
        #compute voltage eq matrices 
        self.Mv = ( D.T @ lds.A @ np.linalg.pinv(D.T) ).astype(SpikingNeuralNet.FLOAT_TYPE)
        self.Mr = ( D.T @ (lds.A + np.eye(lds.A.shape[0])) @ D ).astype(SpikingNeuralNet.FLOAT_TYPE)
        self.Mo = ( - D.T @ D ).astype(SpikingNeuralNet.FLOAT_TYPE) 
        self.Mc = ( D.T @ lds.B ).astype(SpikingNeuralNet.FLOAT_TYPE)
        self.max_spikes = int(1e6)       
        self.O = np.zeros((N, self.max_spikes),dtype = np.float32) #pre allocate space for spike rasters - needed for use in Numba implementation
        self.spike_nums = np.zeros((self.N,), dtype = np.int)
        self.vth = np.asarray(
            [ D[:,i].T @ D[:,i] for i in np.arange(N)],
            dtype = SpikingNeuralNet.FLOAT_TYPE
            ) / 2

          
        self.set_initial_rs(D, lds.X0)
     
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

        self.lds_data = self.lds.run_sim()
        U = self.lds_data['U']
        vth = self.vth
        num_pts = self.num_pts
        dt = self.dt
        t = np.asarray(self.t)
        V = self.V
        r = self.r
        O = self.O
        Mv = self.Mv
        Mo = self.Mo
        Mr = self.Mr
        Mc = self.Mc
        spike_trans_prob  = self.spike_trans_prob
        spike_nums = self.spike_nums
        
        top = np.hstack((Mv, Mr))
        bot = np.hstack((np.zeros((self.N, self.N), dtype=SpikingNeuralNet.FLOAT_TYPE), -np.eye(self.N)))
        A_exp = np.vstack((top, bot)) #matrix exponential for linear part of eq
        A_exp = ( expm(A_exp* dt) ).astype(SpikingNeuralNet.FLOAT_TYPE)
        
        
        print('Starting Simulation.')
        fast_sim(dt, vth, num_pts, t, V, r, O, U,  Mo,  Mc,  A_exp, spike_trans_prob, spike_nums)        
        print('Simulation Complete.')
        
        return super().run_sim()

class ClassicDeneveNet(GapJunctionDeneveNet):
    ''' 
    Classic Deneve is a special case of Gap Junction coupling.
    The coupling matrix Mv is replaces with the identity matrix times a leak parameter lam_v
    ''' 
    def __init__(self, T, dt, N, D, lds, lam_v, t0 = 0):
        super().__init__(T, dt, N, D, lds, t0)
        self.Mv = -np.eye(N) * lam_v
    
class SelfCoupledNet(GapJunctionDeneveNet):
    '''
    Extends Gap junction net with the following changes:
        - Mv is changed to derived diagonal matrix
        - D matrices are rotated to basis of A
        - A is diagonalized 
        - Thresholds are scales by tau_s 
    '''
    def __init__(self, T, dt, N, D, lds, t0 = 0, spike_trans_prob=1):
        super().__init__(T, dt, N, D, lds, t0, spike_trans_prob)
        
        if np.linalg.matrix_rank(lds.A) < np.min(lds.A.shape):
            print("Dynamics Matrix A is  not full rank, has dims (%i, %i) but rank %i"%(lds.A.shape[0], lds.A.shape[1], np.linalg.matrix_rank(lds.A)))
            assert(False)
        else:
            lamA_vec, uA = np.linalg.eig(lds.A)
            lamA_vec = np.concatenate((lamA_vec, lamA_vec))
            uA = np.concatenate((uA, -1 * uA), axis = 1)
            uA[uA == 0] = 0

        dim = np.linalg.matrix_rank(D)
        assert(N >= 2 * dim), "Must have at least 2 * dim neurons but rank(D) = %i and N = %i"%(dim,N)
              
        lamA = pad_to_N_diag_matrix(lamA_vec, N)     
        self.lamA = lamA
        self.uA = uA
        
        assert(np.all(np.imag(np.real_if_close(lamA)) == np.zeros(lamA.shape))), ('A must have real eigenvalues', lamA)
        
        _, sD, vDT  = np.linalg.svd(D)
        sD = np.concatenate((sD, sD))
        sD = pad_to_N_diag_matrix(sD, N)
        
        self.Mv =  lamA  
        self.Mr = (np.eye(N) + lamA) 
  
#         sD_inv = np.zeros(sD.shape)
#         for i in range(2*dim):
#             sD_inv[i, i]  = sD[i, i]**-1      
#         self.sD = sD
#         self.sD_inv = sD_inv
#         uA = np.pad(uA, ((0,0),(0, N - 2 * dim)) ,mode='constant')
#         print(sD_inv.shape, uA.shape)
#         
#         Delta = sD_inv @ uA
#         
#         
#         self.set_initial_rs(Delta, lds.X0)
#         self.D = Delta
#         
#         self.vth =  (1 / 2) * np.ones((self.N,))
#         
#         lamA_inv = np.zeros(lamA.shape)
#         for i in range(2 * dim):
#             lamA_inv[i,i] = lamA[i,i]**-1
#         
#         Beta = sD @ uA.T @ self.lds.B
#         self.Mc = Beta        
#         self.vDT = vDT
#         self.Mo = - sD @ sD @ np.eye(N)
        
        sD_inv = np.zeros(sD.shape)
        for i in range(2*dim):
            sD_inv[i, i]  = sD[i, i]**-1      
        self.sD = sD
        self.sD_inv = sD_inv
        uA = np.pad(uA, ((0,0),(0, N - 2 * dim)) ,mode='constant')
        Delta = uA @ sD_inv
        
        self.set_initial_rs(Delta, lds.X0)
        self.D = Delta
        
        self.vth =  (1 / 2) * np.ones((self.N,))
        
        lamA_inv = np.zeros(lamA.shape)
        for i in range(2 * dim):
            lamA_inv[i,i] = lamA[i,i]**-1
        
        Beta = lamA @ sD @ lamA_inv @ uA.T @ self.lds.B
        self.Mc = Beta        
        self.vDT = vDT
        self.Mo = - np.eye(N)
        
        
        
    def run_sim(self): 
        '''
        process data and call c_solver library to quickly run sims
        '''
        self.lds_data = self.lds.run_sim()
        U = self.lds_data['U']
        vth = self.vth
        num_pts = self.num_pts
        t = np.asarray(self.t)
        V = self.V
        r = self.r
        O = self.O
        spike_nums = self.spike_nums
        Mv = self.Mv
        Mo = self.Mo
        Mr = self.Mr
        Mc = self.Mc
        dt = self.dt
        spike_trans_prob  = self.spike_trans_prob

        bot = np.hstack((np.zeros((self.N, self.N)), -  np.eye(self.N)))
        top = np.hstack((Mv, Mr))
        A_exp = np.vstack((top, bot)) #matrix exponential for linear part of eq
        A_exp = ( expm(A_exp*dt) ).astype(SpikingNeuralNet.FLOAT_TYPE)
        
        print('Starting Simulation.')
        fast_sim(dt, vth, num_pts, t, V, r, O, U,  Mo,  Mc,  A_exp, spike_trans_prob, spike_nums, floor_voltage=True, one_spike_per_step=False)
        print('Simulation Complete.')
        return SpikingNeuralNet.run_sim(self)


'''
Helper functions
'''
    
def gen_decoder(d, N, mode='random'): 
        '''
        Generate a d x N decoder mapping neural activities to 
        state space (X). Only 'random' mode implemented, which
        draws decoders from 0 mean gaussian with std=1
        '''
        assert(mode in ['random', '2d cosine']), 'The provided decoder mode \"{0}\" is not one of \"random\" or \"2d cosine\".'.format(mode) 
    
        if mode == 'random':
            D =  np.random.normal(loc=0, scale=1, size=(d, N) )
            
            
        elif mode == '2d cosine':
            assert(d == 2), "2D Cosine Mode is only implemented for d = 2"
            thetas = np.linspace(0, 2 * np.pi, num=N)
            D = np.zeros((d,N), dtype = np.float64)
            D[0,:] = np.cos(thetas)
            D[1,:] = np.sin(thetas)
        
        for i in range(D.shape[1]):
                D[:,i] /= np.linalg.norm(D[:,i])
        return D
    