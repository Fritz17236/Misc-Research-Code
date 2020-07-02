'''
Created on Feb 19, 2020

@author: cbfritz

Class definitions for Spiking Neural Networks Simulations
'''
import Simulation_Analysis_Toolset as sat 
import numpy as np
from abc import abstractmethod
import matplotlib.pyplot as plt #@UnusedImport squelch
from numba import jit
from scipy.optimize import nnls
from scipy.linalg import expm
from utils import pad_to_N_diag_matrix
'''
Class Definitions
'''
   
#todo:
# exponential integrator for better accuracy 
# integrate O 
# clean up
# dictionary packing for passing params (kwargs)


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
        self.O = np.asarray([[] for i in np.arange(N)])
        self.vth = np.asarray(
            [ D[:,i].T @ D[:,i] for i in np.arange(N)],
            dtype = np.double
            )
        
        if thresh == 'full':
            self.vth = self.vth * .5
            
        
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
        @jit(nopython=True)     
        def fast_sim(dt, vth, num_pts, t, V, r, O, U, Mv, Mo, Mr, Mc,  lam, A_exp):
            '''run the sim quickly using numba just-in-time compilation'''
            N = V.shape[0]
            for count in np.arange(num_pts-1):
                pct =  (count+1) / num_pts
                pct *= 100 
                if (pct // 1) == 0: 
                    print( pct * 100, r'%')                #get if max voltage above thresh then spike
                diff = V[:,count] - vth
                max_v = np.max(diff)
                if max_v >= 0:
                    idx = np.argmax(diff) 
                    V[:,count] += Mo[:,idx]
                    r[idx,count] += 1
                    O[idx] = np.append(O[idx], t[-1])
                
                
                state = np.hstack((V[:,count], r[:,count]))
                state = A_exp @ state
     
                r[:,count+1] = state[N:]
                V[:,count+1] = state[0:N] +  dt * Mc @ U[:,count]
                t[count+1] =  t[count] + dt
                count += 1
                
                
        
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
        lam  = self.lam
        
        top = np.hstack((Mv, Mr))
        bot = np.hstack((np.zeros((self.N, self.N)), -lam * np.eye(self.N)))
        A_exp = np.vstack((top, bot)) #matrix exponential for linear part of eq
        A_exp = expm(A_exp* dt)
        
        fast_sim(dt, vth, num_pts, t, V, r, O, U, Mv, Mo, Mr, Mc, lam, A_exp)


        if not self.suppress_console_output:
            print('Simulation Complete.')
        
        return super().run_sim()

class ClassicDeneveNet(GapJunctionDeneveNet):
    ''' 
    Classic Deneve is a special case of Gap Junction coupling.
    The coupling matrix Mv is replaces with the identity matrix times a leak parameter lam_v
    '''
    def __init__(self, T, dt, N, D, lds, lam, lam_v, t0 = 0, thresh = 'full'):
        super().__init__(T, dt, N, D, lds, lam, t0, thresh)
        self.Mv = np.eye(N) * lam_v
    
class SelfCoupledNet(GapJunctionDeneveNet):
    '''
    Extends Gap junction net with the following changes:
        - Mv is changed to derived diagonal matrix
        - D matrices are rotated to basis of A
        - A is diagonalized 
        - Thresholds are scales by tau_s 
    '''
    def __init__(self, T, dt, N, D, lds, lam, t0 = 0, thresh = 'full'):
        super().__init__(T, dt, N, D, lds, lam, t0)

        
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
        
        tau_syn = lam**-1
        self.tau_syn = tau_syn

        # implement V as curly v in notes
        self.Mv =  lamA  
        
        self.Mr = (np.eye(N) / tau_syn + lamA) 
  
        
        sD_inv = np.zeros(sD.shape)
        for i in range(2*dim):
            sD_inv[i, i]  = sD[i, i]**-1
        
        self.sD = sD
  
        self.sD_inv = sD_inv
        uA = np.pad(uA, ((0,0),(0, N - 2 * dim)))
        
        Delta = uA @ sD_inv
        
        for i in range(2*dim):
            Delta[:,i] /= np.linalg.norm(Delta[:,i])
          
        
        self.set_initial_rs(Delta, lds.X0)
        
        self.D = Delta
        
        self.vth =  (tau_syn / 2) * np.diag(Delta.T @ Delta)
        self.Mc = np.zeros((N,2*dim)) 
        self.Mc[0:dim, 0:dim] = self.lds.B
        self.Mc[dim:2*dim, dim:2*dim] = self.lds.B
        self.Mc = self.Mc
        self.vDT = vDT
        self.Mo = - np.eye(N)
        
#         for i in range(dim):
#             self.Mo[i+dim,i] =  self.Mo[i,i]
#             self.Mo[i, i + dim] =  -self.Mo[i+dim, i+dim] 
#         
#         plt.imshow(self.Mo)
#         plt.colorbar()
#         plt.show()
  
    def run_sim(self): 
        '''
        process data and call c_solver library to quickly run sims
        '''
        @jit(nopython=True)     
        def fast_sim(dt, vth, num_pts, t, V, r, O, U,  Mo, Mc, A_exp):
            '''run the sim quickly using numba just-in-time compilation'''
            N = V.shape[0]
            for count in np.arange(num_pts-1):
                pct =  (count+1) / num_pts
                pct *= 100 
                if (pct // 1) == 0: 
                    print( pct * 100, r'%')                #get if max voltage above thresh then spike

                state = np.hstack((V[:,count], r[:,count]))
                state = A_exp @ state
     
                r[:,count+1] = state[N:]
                V[:,count+1] = state[0:N] +  dt * Mc @ U[:,count]
                
                                    
                diffs = V[:,count+1] - vth
                for idx in range(V.shape[0]):
                    if diffs[idx] > 0:
                    # a spike occurred. roll back to when it should have happened 
                        V[:,count+1] +=  tau_syn * Mo[:,idx]
                        r[idx,count+1] +=  tau_syn
                        O[idx] = np.append(O[idx], t[-1])
                    

                
                t[count+1] =  t[count] + dt 
                count += 1
                
        
        self.lds_data = self.lds.run_sim()
        U = self.lds_data['U']
        U = np.vstack((U, -1*U))
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
        tau_syn = self.tau_syn
        
        top = np.hstack((Mv, Mr))
        bot = np.hstack((np.zeros((self.N, self.N)), - (1/tau_syn) *  np.eye(self.N)))
        A_exp = np.vstack((top, bot)) #matrix exponential for linear part of eq
        A_exp = expm(A_exp*dt)
        
        fast_sim(dt, vth, num_pts, t, V, r, O, U,  Mo,  Mc,  A_exp)
                 
        

        if not self.suppress_console_output:
            print('Simulation Complete.')
        
        return SpikingNeuralNet.run_sim(self)

class SpikeDropDeneveNet(GapJunctionDeneveNet):
    
    def __init__(self, T, dt, N, D, lds, lam, p, t0 = 0, thresh = 'not full'):
         
        super().__init__(T, dt, N, D, lds, lam, t0, thresh)
        assert(p > 0 and p <= 1), 'Probability of Spike Drops must be greater than 0, and less/equal to 1, but was %f'%p
        self.p = p
        self.seed = 0
        
        

  
           
        
        
    def run_sim(self):
        
        @jit(nopython=True)
        def fast_sim(dt, vth, num_pts, t, V, r, O, U, Mv, Mo, Mr, Mc, lam, A_exp, p):
            '''run the sim quickly using numba just-in-time compilation'''
            #run sim
            seed = 0
            N = len(V[:,0])
            for count in np.arange(num_pts - 1):
                print(int((count+1)/num_pts * 100), r'%')
                #get if max voltage above thresh then spike
                diff = V[:,count] - vth
                max_v = np.max(diff)

                if max_v >= 0:
                    idx = np.argmax(diff)
                    np.random.seed(seed)
                    draw = np.asarray(np.random.binomial(1, p, size = (len(vth),)), dtype = np.float64)
                    draw[idx] = 1 
                    draw = np.multiply(draw, Mo[:,idx])
                    V[:,count] +=  draw 
                    r[idx,count] += 1  
                    O[idx] = np.append(O[idx], t[-1])
                    seed += 1

                    
                state = np.hstack((V[:,count], r[:,count]))
                state = A_exp @ state
     
                r[:,count+1] = state[N:]
                V[:,count+1] = state[0:N] +  dt * Mc @ U[:,count]
                t[count+1] =  t[count] + dt
                count += 1

        
        self.lds_data = self.lds.run_sim()
        U = self.lds_data['U']
        p = self.p
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
        lam  = self.lam
        

        top = np.hstack((Mv, Mr))
        bot = np.hstack((np.zeros((self.N,self.N)), -lam * np.eye(self.N)))
        A_exp = np.vstack((top, bot)) #matrix exponential for linear part of eq
        A_exp = expm(A_exp * dt)
        
        fast_sim(dt, vth, num_pts, t, V, r, O, U, Mv, Mo, Mr, Mc, lam, A_exp, p)


        if not self.suppress_console_output:
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
        if mode == 'random':
            return np.random.normal(loc=0, scale=1, size=(d, N) )
        elif mode == '2d cosine':
            assert(d == 2), "2D Cosine Mode is only implemented for d = 2"
            thetas = np.linspace(0, 2 * np.pi, num=N)
            D = np.zeros((d,N), dtype = np.float64)
            D[0,:] = np.cos(thetas)
            D[1,:] = np.sin(thetas)
            
            return D
    
    
    
#       
# A =  np.zeros((2,2))
# A[0,1] = 1
# A[1,0] = -1
# A =  A
# N = 50
# lam =  1
# mode = '2d cosine'
# p = .9
# 
#     
# D = gen_decoder(A.shape[0], N, mode=mode)
# B = np.eye(2)
# u0 = D[:,0]
# x0 = np.asarray([10, 100])
# T = 5
# dt = 1e-4
# 
# lds = sat.LinearDynamicalSystem(x0, A, B, u = lambda t: u0 , T = T, dt = dt)
# 
# #net = GapJunctionDeneveNet(T=T, dt=dt, N=N, D=D, lds=lds, lam=lam, t0=0, thresh = 'not full')
# net = SpikeDropDeneveNet(T=T, dt=dt, N=N, D=D, lds=lds, lam=lam, t0=0, thresh = 'not full', p = p)
# 
# 
# data = net.run_sim()
# plt.plot(data['t'], data['x_hat'][0,:],label='xhat')
# plt.plot(data['t'], data['x_true'][0,:],label='xtru')
# 
# plt.legend()
# 
# plt.show()
