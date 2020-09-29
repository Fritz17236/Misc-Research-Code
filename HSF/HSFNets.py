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
from utils import has_real_eigvals, cart_to_polar
import warnings
from scipy.signal import hilbert


'''
Helper Functions
'''

@jit(nopython=True)
def fast_sim(dt, vth, num_pts, t, V, r, O, U, Mr,   Mo, Mc, A_exp, spike_trans_prob, spike_nums, floor_voltage=False, rotating_basis=False, ang_vels=None):
    '''run the sim quickly using numba just-in-time compilation'''
    N = V.shape[0]
    max_spikes = len(O[0,:])
    Mo0 = Mo.copy()
    Mc0 = Mc.copy()
    for count in np.arange(num_pts-1):
        if rotating_basis: #if angular velocity is defined (bases are rotating) apply e^(iwt) to the static complex matrices Mo, Mc
            for j in np.arange(ang_vels.shape[0]):
                Mo[j,:] = Mo0[j,:] * np.exp(0*1j*ang_vels[j]*t[count])
                Mc[j,:] = Mc0[j,:] * np.exp(0*1j*ang_vels[j]*t[count])
        else:
            Mo = Mo0
            Mc = Mc0 
        
        state = np.hstack((V[:,count], r[:,count]))
        state = A_exp @ state

        r[:,count+1] = state[N:]
        V[:,count+1] = state[0:N] +  dt * Mc @ U[:,count]  
        
        if floor_voltage:
            for i in np.arange(N):
                if np.abs(V[i,count+1]) < -vth[i]:
                    V[i,count+1] = -vth[i] 
            

        
        # spikeneurons
        diffs = np.abs(V[:,count+1]) - vth
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
            
 
                
        t[count+1] =  t[count] + dt 
        count += 1       
             
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
        self.X0 = self.lds.X0        
        self.N = N # number of neurons in network
        self.D = ( D ).astype(SpikingNeuralNet.FLOAT_TYPE) # Decoder matrix
        self.num_pts = int(np.floor(((T - t0)/dt))) 
        self.V = np.zeros((N,self.num_pts), dtype=SpikingNeuralNet.FLOAT_TYPE)
        self.r = np.zeros_like(self.V, dtype=SpikingNeuralNet.FLOAT_TYPE)
        self.t = np.zeros((self.num_pts,), dtype=SpikingNeuralNet.FLOAT_TYPE)
        super().__init__(self.X0, T, dt, t0)
        
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
    
    def set_initial_rs(self, D, X0, complex=False):
        '''
        Use nonnegative least squares to set initial PSC/r values
        '''
        if complex:
            
            Dreal = np.zeros((2*D.shape[0], 2* D.shape[1] ))
            Dreal[0:D.shape[0],0:D.shape[1]] = np.real(D)
            Dreal[D.shape[0]:,D.shape[1]:] = np.imag(D)
         
            assert(X0.ndim==1), "X0 should be a vector but had {0} dims".format(X0.ndim)
            
            
            Xreal = np.zeros((2*len(X0),))
            Xreal[0:len(X0)] = np.real(X0)
            Xreal[len(X0):] = np.imag(X0)
         
            assert(self.r.shape[0]==D.shape[1]), "r0 with length {0} does not match the cols of D with shape {1}".format(self.r.shape[0], D.shape[1])
            
            rreal = nnls(Dreal, Xreal)[0]
            rreal = np.linalg.pinv(Dreal) @ Xreal
         
            rcomplex = np.zeros((len(self.r[:,0]),),dtype = np.complex128)
            rcomplex = rreal[0:len(rreal)//2] +  1j * rreal[len(rreal)//2:]
                        
       
    
            
            self.r[:,0] = rcomplex
            
        else:
            self.r[:,0] = nnls(D, X0)[0]
    @abstractmethod
    def pack_data(self, final=True):
        ''' Pack simulation data into dict object '''
        data = super().pack_data()
     
        assert(data['r'].shape == data['V'].shape), "r has shape %s but V has shape %s"%(data['r'].shape, data['V'].shape)
        assert(data['V'].shape[1] == len(data['t'])), "V has shape %s but t has shape %s"%(data['V'].shape, data['t'].shape)
        

        return data
    
    @abstractmethod
    def get_net_estimate(self):
        ''' Get the network estimate '''
        pass
 
    @abstractmethod
    def get_true_state(self):
        ''' Get the state space data from the true dynamical system'''
        pass
    
    @abstractmethod
    def get_time(self):
        ''' return simulation time'''
        
    @abstractmethod  
    def run_sim(self):
        ''' Return the packed simulation data'''
        return self.pack_data()
     

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
            
    def get_net_estimate(self):
        return self.D @ self.r
    
    def get_time(self):
        return self.t
    
    def get_true_state(self):
        return self.lds_data['X']
    
    def pack_data(self):
        self.data = super().pack_data()
        self.data['x_hat'] = self.get_net_estimate()
        self.data['x_true'] = self.get_true_state()
        self.data['t_true'] = self.get_time()
        
        return self.data
        
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
        bot = np.hstack((np.zeros((self.N, self.N), dtype=SpikingNeuralNet.FLOAT_TYPE), -(np.eye(self.N)*(1 + 1j))))
        A_exp = np.vstack((top, bot)) #matrix exponential for linear part of eq
        A_exp = ( expm(A_exp* dt) ).astype(SpikingNeuralNet.FLOAT_TYPE)
        
        
        print('Starting Simulation.')
        fast_sim(dt, vth, num_pts, t, V, r, O, U, Mr,  Mo,  Mc,  A_exp, spike_trans_prob, spike_nums)        
        print('Simulation Complete.')
        
        return self.pack_data()


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
        
        dim = np.linalg.matrix_rank(D)
        
        assert(np.linalg.matrix_rank(lds.A) == np.min(lds.A.shape)), "Dynamics Matrix A is  not full rank, has dims (%i, %i) but rank %i"%(lds.A.shape[0], lds.A.shape[1], np.linalg.matrix_rank(lds.A))
        assert(N >= 2 * dim), "Must have at least 2 * dim neurons but rank(D) = %i and N = %i"%(dim,N)
        assert(has_real_eigvals(lds.A)), "A must have real eigenvalues"
        
        self.__set_vth() 
        self.__set_Mv__()
        self.__set_Beta__()
        self.__set_Mr__()       
        self.__set_Mo__()
        self.__rotate_decoder()
        self.set_initial_rs(self.D, self.lds.X0)
       
    def __set_Mv__(self):
        ''' Set the Diagonal Voltage Leak Matrix using d x d Matrix A and Neurons N
        Assigns Mc =  N x N matrix having the top 2 d diagonals from the eigvals of A (repeated once)
        '''
        Mv =  np.zeros((self.N,self.N))
        lamA_vec, _ = np.linalg.eig(self.lds.A)
        lamA_vec = np.concatenate((lamA_vec, lamA_vec))
        for i, lam in enumerate(lamA_vec):
            Mv[i, i] = lam
            
        self.Mv = Mv
        
    def __set_Beta__(self):
        '''
        Set the input matrix to the new basis. 
        Beta = U.T B U,    
        Assigns Beta =  N x 2d matrix, 
        '''
        dim = (self.D).shape[0]
        _, sD, _ = np.linalg.svd(self.D)
        _, uA = np.linalg.eig(self.lds.A)
        
        #sD = np.hstack((sD, sD))
        #uA = np.hstack((uA, -uA))
        
        self.Mc = np.zeros((self.N, dim))
        
        self.Beta= uA.T @ self.lds.B @ uA
        self.Mc[0 : dim, :] = np.diag(sD) @ self.Beta
        self.Mc[dim : 2*dim, :] = -np.diag(sD) @ self.Beta

        
    def __set_Mr__(self):
        ''' Set the Post-synaptic matrix via SVD of D
        Assigns self.Mr = S @ (L + I) @ S.T, where D = USV.T and A =ULU.T, as an NxN matrix
        Also Sets D = uA @ S
        '''
        lamA_vec, uA = np.linalg.eig(self.lds.A)
        
        lamA_vec = np.hstack((lamA_vec, lamA_vec))
        uA = np.hstack((uA, -uA))
        
        L = np.diag(lamA_vec)
        
        _, S, _ = np.linalg.svd(self.D)
        S = np.hstack((S, S))
        
        dim = self.D.shape[0]
        
        self.Mr = np.zeros((self.N, self.N))
        self.Mr[0 : 2*dim, 0 : 2*dim] = np.diag(S) @ (np.eye(2*dim) + L) @ np.diag(S)
        
    def __set_Mo__(self):
        ''' 
        Set the Voltage Fast Reset Matrix
        
        Assigns self.Mo = S.T @ S where D = USV.T, as an NxN matrix
        '''
        
        _, S, _ = np.linalg.svd(self.D)
        dim = self.D.shape[0]
        S = np.diag(np.hstack((S, S)))
        
        self.Mo = np.zeros((self.N, self.N))
        self.Mo[0 : 2*dim, 0 : 2*dim] = -np.square(S)
        
    def __set_vth(self):
        ''' Set the threshold voltages for the rotated neurons'''
        _, sD, _ =  np.linalg.svd(self.D)
        dim = self.D.shape[0]
        sD = np.hstack((sD, sD, np.zeros((self.N - 2*dim,))))
        self.vth = np.square(sD) / 2
        
    def __rotate_input(self, U):
        '''Given a d x T time series of input, rotate to new basis uA'''
        _, uA = np.linalg.eig(self.lds.A)
        return  uA.T @ U
    
    def __rotate_decoder(self):
        ''' Move the assigned decoder to rotated basis. overwrites self.D!'''
        _, uA = np.linalg.eig(self.lds.A)
        _, S, _ = np.linalg.svd(self.D)
        
        #uA = np.hstack((uA, -uA))
        
        S = np.diag(S)
        
        dim = self.D.shape[0]
        
        self.D = np.zeros((dim, self.N))
        self.D[:,0: 2*dim] = np.hstack((uA @ S, -uA @ S))
        
        
        
        
        
    def run_sim(self): 
        '''
        process data and call c_solver library to quickly run sims
        '''
        self.lds_data = self.lds.run_sim()
        U = self.__rotate_input(self.lds_data['U'])
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

        bot = np.hstack((np.zeros((self.N, self.N)), -np.eye(self.N)))
        top = np.hstack((Mv, Mr))
        A_exp = np.vstack((top, bot)) #matrix exponential for linear part of eq
        A_exp = ( expm(A_exp*dt) ).astype(SpikingNeuralNet.FLOAT_TYPE)
        
        print('Starting Simulation.')
        fast_sim(dt, vth, num_pts, t, V, r, O, U, Mr,  Mo,  Mc,  A_exp, spike_trans_prob, spike_nums, floor_voltage=True)
        print('Simulation Complete.')
        
        
        
        return self.pack_data()


class ComplexSelfCoupledNet(GapJunctionDeneveNet):
    '''
    Extends Gap junction net with the following changes:
        - Mv is changed to derived diagonal matrix
        - D matrices are rotated to basis of A
        - A is diagonalized 
        - Thresholds are scales by tau_s 
    '''
    def __init__(self, T, dt, N, D, lds, t0 = 0, spike_trans_prob=1):
        super().__init__(T, dt, N, D, lds, t0, spike_trans_prob)
        
        dim = np.linalg.matrix_rank(D)
        
        assert(np.linalg.matrix_rank(lds.A) == np.min(lds.A.shape)), "Dynamics Matrix A is  not full rank, has dims (%i, %i) but rank %i"%(lds.A.shape[0], lds.A.shape[1], np.linalg.matrix_rank(lds.A))
        assert(N >= 2 * dim), "Must have at least 2 * dim neurons but rank(D) = %i and N = %i"%(dim,N)
        self.__set_bases__()
        
        self.__complexify_bases__()
        
        self.__complexify_initial_conditions__()
        self.__set_vth__() 
        self.__set_Mv__()
        self.__set_Beta__()
        self.__set_Mr__()       
        self.__set_Mo__()
        self.__rotate_decoder__()
        self.set_initial_rs(self.D, self.X0, complex=True)
   
        
    def __complexify_initial_conditions__(self):
        '''
        Find complex extensions for 
        V0, r0, X0 and set them as initial conditions
        '''
        
        #compute imaginary components        
        Vi = hilbert(self.V[:,0])
        ri = hilbert(self.r[:,0])
        xi = hilbert(self.X0)
        
        #complexify data structs
        self.V = self.V.astype(np.complex128)
        self.r = self.r.astype(np.complex128)
        self.X0 = self.X0.astype(np.complex128)
        
        
        # complexify initial conditions using 
        self.V[:,0] = self.V[:,0] + 1j * Vi
        self.r[:,0] = self.r[:,0] + 1j * ri
        self.X0 = self.X0 + 1j * xi
        
        
    
    def __set_bases__(self):
        '''Set the bases U(W) and V'''
                
        self.dim = self.D.shape[0]
        
        # Factorize u in dxd, s & v in dx1
        self.lam_d_vec,  self.u_d = np.linalg.eig(self.A) 
        
     
        self.D = self.D.astype(np.complex128) +  1j * hilbert(D, axis=0).astype(np.complex128)
        
        _ , self.s_d_vec, _ = np.linalg.svd(self.D)
        
        
        self.s_d = np.diag(self.s_d_vec)
        self.lam_d = np.diag(self.lam_d_vec)
        
        
        #compute the nontrivial 2d-dimensional matrices
        self.lam_2d = np.diag(np.hstack((self.lam_d_vec, self.lam_d_vec)))
        self.u_2d = np.hstack((self.u_d, -self.u_d)) # u in d x 2d
        self.s_2d = np.diag((np.hstack((self.s_d_vec, self.s_d_vec)))) # s in 2d x 2d        
                   
    def __complexify_bases__(self):
        ''' 
         If dynamical system contains complex eigenvalues,
         adjust U and lambda so that lambda is real and U forms a complex basis
        '''
        
        self.s_d = self.s_d.astype(np.complex128) 
        self.s_2d = self.s_2d.astype(np.complex128)
        self.lam_bar = self.lam_2d
        self.W_d = self.u_d
        self.W_2d = self.u_2d
        

    
    def __set_Mv__(self):
        ''' Set the Diagonal Voltage Leak Matrix using d x d Matrix A and Neurons N
        Assigns Mc =  N x N matrix having the top 2 d diagonals from the eigvals of A (repeated once)
        '''
        self.Mv =  np.zeros((self.N,self.N), dtype=np.complex128)
        for i in range(2*self.dim):
            self.Mv[i, i] = self.lam_bar[i,i]
        
    def __set_Beta__(self):
        '''
        Set the input matrix to the new basis. 
        Beta = U.T B U,    
        Assigns Beta =  N x 2d matrix, 
        '''

        dim = (self.D).shape[0]
        _, sD, _ = np.linalg.svd(self.D)
        _, uA = np.linalg.eig(self.lds.A)
        
        #sD = np.hstack((sD, sD))
        #uA = np.hstack((uA, -uA))
        
        
        
#         self.Mc[0 : dim, :] = np.diag(sD) @ self.Beta
#         self.Mc[dim : 2*dim, :] = -np.diag(sD) @ self.Beta
#         print('Remove Override in set beta')
#         return
        
        
        
        
        
        self.Mc = np.zeros((self.N, 2*dim), dtype=np.complex128)
        self.Beta= (np.conjugate(self.W_2d).T) @ self.lds.B.astype(np.complex128) @ self.W_2d       
        self.Mc[0 : 2* dim, :] = (self.s_2d) @ self.Beta
        
        
    def __set_Mr__(self):
        ''' Set the Post-synaptic matrix via SVD of D
        Assigns self.Mr = S @ (L + I) @ S.T, where D = USV.T and A =ULU.T, as an NxN matrix
        '''   
        self.Mr = np.zeros((self.N, self.N), dtype=np.complex128)
        self.Mr[0 : 2*self.dim, 0 : 2*self.dim] = np.conjugate(self.s_2d.T) @ (np.eye(2*self.dim) + self.lam_bar) @ self.s_2d 
        
    def __set_Mo__(self):
        ''' 
        Set the Voltage Fast Reset Matrix
        
        Assigns self.Mo = S.T @ S where D = USV.T, as an NxN matrix
        '''    
        self.Mo = np.zeros((self.N, self.N),dtype=np.complex128)
        
        self.Mo[0:2*self.dim, 0:2*self.dim] = -np.conjugate(self.s_2d.T) @ self.s_2d 
        
        
    def __set_vth__(self):
        ''' Set the threshold voltages for the rotated neurons'''
        self.vth = np.diag(np.conjugate(self.s_2d.T) @ self.s_2d)  
        assert(np.all(np.isclose(self.vth.imag,0))), "vth should not have any imaginary component but was {0}".format(self.vth)
        self.vth = np.real(self.vth)
        
    def __rotate_input__(self, U):
        '''Given a d x T time series of input, rotate to new basis uA'''
        return  (np.conjugate(self.W_2d).T) @ U
    
    def __rotate_decoder__(self):
        ''' 
         Move the assigned decoder to rotated basis. 
         sets self.Delta, used to decode self.r to get network estimate
         '''
        self.D = np.zeros((self.dim, self.N), dtype=np.complex128)
        self.D[:,0: 2*self.dim] = self.W_2d @ self.s_2d  
      
    def get_net_estimate(self):
        # net estimate is given by complex bases projections
        
        omegas = np.imag(np.diag(self.lam_2d))
        xhat = np.zeros((self.dim, len(self.t)), dtype=np.complex128)
        
        for j in range(2*self.dim):
            

            
            
            
            # choose Wj basis vector
            # create time series for that basis by mult e 2 i pi
            for i,t in enumerate(range(len(self.t))):
                wj = self.W_2d[:,j] * np.exp(2*np.pi * 1j * omegas[j] * t)
                xhat[:,i] += wj * self.r[j, i] 
            # portion of estimate from that basis is that times series scaled by rho j
            
            # add to network estimate
            return xhat
            
        return self.D @ self.r
   
    def run_sim(self): 
        '''
        process data and call c_solver library to quickly run sims
        '''
        self.lds_data = self.lds.run_sim()
        U = self.__rotate_input__(self.lds_data['U'])
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
        ang_vels = np.imag(np.diag(self.lam_bar))
        
    
        
     
        
        spike_trans_prob  = self.spike_trans_prob

        bot = np.hstack((np.zeros((self.N, self.N)), -np.eye(self.N)))
        top = np.hstack((Mv, Mr))
        A_exp = np.vstack((top, bot)) #matrix exponential for linear part of eq
        A_exp = ( expm(A_exp*dt) )
        
        print('Starting Simulation.')
        fast_sim(dt, vth, num_pts, t, V, r, O, U, Mr,  Mo,  Mc,  A_exp, spike_trans_prob, spike_nums, floor_voltage=True, rotating_basis=True, ang_vels=ang_vels)
        print('Simulation Complete.')
        
        
        
        return self.pack_data()


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
    
    
 
  
N = 4
p= 1
T = 20
k = 1
dt = 1e-3
d = 2
A =  np.zeros((d,d))
A[0,1] = -1
A[1,0] = 1 

#A = -np.eye(d) 
B = np.eye(d)
 



D = .1 * np.hstack((
         np.eye(d),
         np.zeros((d, N - d))
         ))

D =   .1 *gen_decoder(d, N, mode='2d cosine')

plot_step = 10

stim = lambda t : np.asarray([np.sin(t), np.cos(t)]) * 0 
 
x0 = np.asarray([.5, 0])
lds = sat.LinearDynamicalSystem(x0, A, B, u = stim , T = T, dt = dt)
net = ComplexSelfCoupledNet(T=T, dt=dt, N=N, D=D, lds=lds, t0=0, spike_trans_prob = p)
data = net.run_sim()

print(data['spike_nums'])

# v = Sj * epsilon j 


# for i in range(4):
#         plt.axhline(y=data['vth'][i])
#         diffs = np.abs(data['V'][i,:]) - data['vth'][i]
#         print(np.any(diffs>0))
#         #plt.plot(data['t'],np.abs(data['V'][i,:]), label='Neuron %i Voltage'%i)
#         plt.plot(data['t'],diffs)
# plt.legend()
# plt.title("Neuron Membrane Potentials")
# plt.xlabel(r"Simulation Time (Dimensionless Units of $\tau$")
# plt.ylabel('Membrane Potential')
#  

#plt.plot(data['t'],data['r'][0,:])
#plt.show()

# print(np.max(data['O']))
#  
vmin = -np.min(np.real(data['vth']))
vmax = np.max(np.real(data['vth']))
# 
# plt.plot(data['t'],(data['Mr']@ data['r'])[0,:])
# plt.show()
#          
plt.figure()
for i in range(1):
    plt.plot(data['t'],np.real(data['V'][i,:]), label='Neuron %i Voltage'%i)
    plt.plot(data['t'],np.imag(data['V'][i,:]), label='Imaginary')
plt.legend()
plt.title("Neuron Membrane Potentials")
plt.xlabel(r"Simulation Time (Dimensionless Units of $\tau$")
plt.ylabel('Membrane Potential')
       
plt.figure()
cbar_ticks = np.linspace( start = vmin, stop = vmax,  num = 8, endpoint = True)
plt.imshow(np.real(data['V']),extent=[0,data['t'][-1], 0,3],vmax=vmax, vmin=vmin)
plt.xlabel(r"Dimensionless Units of $\tau$")
plt.axis('auto')
cbar = plt.colorbar(ticks=cbar_ticks)
cbar.set_label(r'$\frac{v_j}{v_{th}}$')
cbar.ax.set_yticklabels(np.round(np.asarray([c / vmax for c in cbar_ticks]), 2))
plt.title('Neuron Membrane Potentials')
plt.ylabel('Neuron #')
plt.yticks([.4,1.15,1.85,2.6], labels=[1, 2, 3, 4])
       
       
plt.figure()
plt.plot(data['t'][0:-1:plot_step], data['x_hat'][0,0:-1:plot_step]/np.max(data['x_hat'][0,:]),c='r',label='Decoded Network Estimate (Dimension 0)' )
plt.plot(data['t'][0:-1:plot_step], data['x_hat'][1,0:-1:plot_step]/np.max(data['x_hat'][1,:]),c='g',label='Decoded Network Estimate (Dimension 1)' )
plt.plot(data['t_true'][0:-1:plot_step], data['x_true'][0,0:-1:plot_step],c='k')
plt.plot(data['t_true'][0:-1:plot_step], data['x_true'][1,0:-1:plot_step],c='k',label='True Dynamical System')
plt.title('Network Decode')
plt.legend()
plt.ylim([-1.1, 1.1])
plt.xlabel(r'Dimensionless Time $\tau_s$')
plt.ylabel('Decoded State')
       
plt.figure()
plt.plot(data['t'][0:-1:plot_step], data['x_hat'][0,0:-1:plot_step] - data['x_true'][0,0:-1:plot_step],c='r',label='Estimation Error (Dimension 0)' )
plt.plot(data['t'][0:-1:plot_step], data['x_hat'][1,0:-1:plot_step] - data['x_true'][1,0:-1:plot_step],c='g',label='Estimation Error (Dimension 1)' )
plt.title('Decode Error')
plt.legend()
plt.ylim([-1.1, 1.1])
plt.xlabel(r'Dimensionless Time $\tau_s$')
plt.ylabel('Decode Error')
  
plt.show()