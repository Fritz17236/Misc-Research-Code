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
from utils import has_real_eigvals, cart_to_polar, is_diagonal
import warnings
from scipy.signal import hilbert


'''
Helper Functions
'''

@jit(nopython=True)
def fast_sim(dt, vth, num_pts, t, V, r, O, U, Mr, Mo, Mc, A_exp, spike_trans_prob, spike_nums):
    '''run the sim quickly using numba just-in-time compilation'''
    N = V.shape[0]
    max_spikes = len(O[0, :])
    for count in np.arange(num_pts-1):
        state = np.hstack((V[:,count], r[:,count]))
        state = A_exp @ state
        r[:,count+1] = state[N:]
        V[:,count+1] = state[0:N] + dt * Mc @ U[:,count]

        diffs = np.real(V[:,count+1]) - vth
        if np.any(diffs > 0):
            idx = np.argmax(diffs)
            V[:,count+1] += Mo[:,idx]

            if spike_trans_prob == 1:
                r[idx,count+1] +=  1
            else:
                sample = np.random.uniform(0, 1)
                if spike_trans_prob >= sample:
                    r[idx,count+1] += 1
            spike_num = spike_nums[idx]

            if spike_num >= max_spikes:
                print("Maximum Number of Spikes Exceeded in Simulation  for neuron ",
                       idx,
                       " having ", spike_num, " spikes with spike limit ", max_spikes)
                assert(False)
            O[idx,spike_num] = t[count]
            spike_nums[idx] += 1

        t[count+1] = t[count] + dt
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
        self.dim = self.lds.A.shape[0]

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

    def set_initial_rs(self, D, X0):
        '''
        Use nonnegative least squares to set initial PSC/r values
        '''
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
        bot = np.hstack((np.zeros((self.N, self.N), dtype=SpikingNeuralNet.FLOAT_TYPE), -(np.eye(self.N))))
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

        # rotate D to same basis as A!!!!!

        _, uA = np.linalg.eig(lds.A)

        dim = lds.A.shape[0]
        uD, sD, vt = np.linalg.svd(D[:,0:dim])  # svd of 1st two cols of D
        self.sD = sD
        self.vt = vt
        self.__set_vth()
        self.__set_Mv__()
        self.__set_Beta__()
        self.__set_Mr__()
        self.__set_Mo__()
        self.set_initial_rs_rot()
        self.__rotate_decoder()

    def __set_Mv__(self):
        ''' Set the Diagonal Voltage Leak Matrix using d x d Matrix A and Neurons N
        Assigns Mc =  N x N matrix having the top 2 d diagonals from the eigvals of A (repeated once)
        '''
        Mv =  np.zeros((self.N,self.N))
        lamA_vec, _ = np.linalg.eig(self.lds.A)
        lamA_vec = np.hstack((lamA_vec, lamA_vec))

        self.Mv = np.diag(lamA_vec)

    def __set_Beta__(self):
        '''
        Set the input matrix to the new basis.
        Beta = U.T B U,
        Assigns Beta =  N x 2d matrix,
        '''
        dim = (self.D).shape[0]
        _, uA = np.linalg.eig(self.lds.A)

        self.Mc = np.zeros((self.N, dim))

        self.Beta= uA.T @ self.lds.B @ uA

        sD = np.diag(self.sD)

        self.Mc = np.vstack((sD @ self.Beta, - sD @ self.Beta))



        self.Mc = self.Mc @ self.Beta

    def __set_Mr__(self):
        ''' Set the Post-synaptic matrix via SVD of D
        Assigns self.Mr = S @ (L + I) @ S.T, where D = USV.T and A =ULU.T, as an NxN matrix
        Also Sets D = uA @ S
        '''
        lamA_vec, uA = np.linalg.eig(self.lds.A)
        L = np.diag(lamA_vec)

        S = np.diag(self.sD)
        S = np.hstack((S, -S))

        dim = self.D.shape[0]

        self.Mr = np.zeros((self.N, self.N))
        self.Mr[0 : 2*dim, 0 : 2*dim] = S.T @ (np.eye(dim) + L) @ S

    def __set_Mo__(self):
        '''
        Set the Voltage Fast Reset Matrix

        Assigns self.Mo = S.T @ S where D = USV.T, as an NxN matrix
        '''

        lamA_vec, uA = np.linalg.eig(self.lds.A)
        L = np.diag(lamA_vec)

        S = np.diag(self.sD)
        S = np.hstack((S, -S))

        dim = self.D.shape[0]

        self.Mo = np.zeros((self.N, self.N))
        self.Mo[0 : 2*dim, 0 : 2*dim] = -S.T @ S

    def __set_vth(self):
        ''' Set the threshold voltages for the rotated neurons'''

        dim = self.D.shape[0]
        sD = np.hstack((self.sD, self.sD, np.zeros((self.N - 2*dim,))))

        self.vth = np.square(sD) / 2

    def __rotate_input(self, U):
        '''Given a d x T time series of input, rotate to new basis uA'''
        _, uA = np.linalg.eig(self.lds.A)
        return  uA.T @ U

    def set_initial_rs_rot(self):
        _, uA = np.linalg.eig(self.lds.A)

        y0 = uA.T @ self.lds.X0


        self.r[:,0] = nnls(self.D, y0)[0]




    def __rotate_decoder(self):
        ''' Move the assigned decoder to rotated basis. overwrites self.D!'''
        _, uA = np.linalg.eig(self.lds.A)
        dim = self.lds.A.shape[0]
        uA = np.hstack((uA, uA))




        self.D = np.zeros((dim, self.N))
        #sD = np.hstack((self.sD, -self.sD))
        #sD = np.vstack((sD , -sD)) / 2
        sD = self.sD
        sD = np.diag(np.hstack((sD, -sD)))

        #np.hstack((sD, -sD))
        # sD = np.vstack((
        #     np.hstack(( sD,  sD)),
        #     np.hstack((sD,  -sD))
        # ))


        self.D = np.hstack((np.diag(self.sD), np.diag(-self.sD)))


    def  run_sim(self):
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
        self.U = U
        spike_trans_prob  = self.spike_trans_prob

        bot = np.hstack((np.zeros((self.N, self.N)), -np.eye(self.N)))
        top = np.hstack((Mv, Mr))
        A_exp = np.vstack((top, bot)) #matrix exponential for linear part of eq
        A_exp = ( expm(A_exp*dt) ).astype(SpikingNeuralNet.FLOAT_TYPE)

        print('Starting Simulation.')
        fast_sim(dt, vth, num_pts, t, V, r, O, U, Mr,  Mo,  Mc,  A_exp, spike_trans_prob, spike_nums)
        print('Simulation Complete.')



        return self.pack_data()

class SymSelfCoupledNet(GapJunctionDeneveNet):
    '''
    Extends Gap junction net with the following changes:
        - Mv is changed to derived diagonal matrix
        - D matrices are rotated to basis of A
        - A is diagonalized
        - Thresholds are scales by tau_s
    '''
    def __init__(self, T, dt, N, D, lds, t0 = 0, spike_trans_prob=1):
        super().__init__(T, dt, N, D, lds, t0, spike_trans_prob)

        assert(self.__network_big_enough__())

        self.__set_LK__()
        self.__set_s__()
        self.__set_uLam__()
        self.__set_new_decoder__()
        self.__set_MvSelf__()
        self.__set_MvGJ__()
        self.__set_Mr__()
        self.__set_Mo__()
        self.__set_vth()
        self.__set_initial_conditions__()


    def __network_big_enough__(self):
        ''' check if the network is big enough to fully represent the d-dimensional state space'''
        return self.N >= 2 * self.dim

    def __set_LK__(self):
        '''
        Set L and K symmetric and skew symmetric matrices of A
        '''
        self.L = (1 / 2) * (self.lds.A + self.lds.A.T)
        self.K = (1 / 2) * (self.lds.A - self.lds.A.T)

    def __set_s__(self):
        '''
         set the decoder scaling factor s
        '''
        uD, sD, vt = np.linalg.svd(self.D[:,0:self.dim])  # svd of 1st d cols of D
        self.s = sD[0]

    def __set_uLam__(self):
        ''' Diagonalize symmetric L matrix and set its bases and eigenvalues as u, lam respectively'''
        self.lam, self.uA = np.linalg.eig(self.L)

    def __set_new_decoder__(self):
        ''' Set the new matrix used to decode estimate from network'''
        zero = np.zeros(self.uA.shape)
        I = np.eye(self.uA.shape[0])
        u2 = np.vstack((
            np.hstack((self.uA, zero)),
            np.hstack((zero, self.uA))
        ))
        s2 = self.s * np.vstack((
            np.hstack((I, zero)),
            np.hstack((zero, I))
        ))
        self.D = u2 @ s2

    def __set_MvSelf__(self):
        ''' Set the Diagonal Voltage Leak Matrix '''
        Lamb = np.diag(self.lam)
        zero = np.zeros(Lamb.shape)
        self.MvSelf = np.vstack((
            np.hstack((Lamb, zero)),
            np.hstack((zero, Lamb))
        ))

    def __set_MvGJ__(self):
        ''' Set the skew symmetric gap junction voltage coupling matrix'''
        utKu = self.uA.T @ self.K @ self.uA
        zero = np.zeros(utKu.shape)
        self.MvGJ = np.vstack((
            np.hstack((utKu, zero)),
            np.hstack((zero, utKu))
        ))

    def __set_Mr__(self):
        ''' set the slow synaptic coupling matrix'''
        luki = np.diag(self.lam) + self.uA.T @ self.K @ self.uA + np.eye(self.L.shape[0])
        self.Mr = self.s**2 * np.vstack((
            np.hstack((luki, -luki)),
            np.hstack((-luki, luki))
        ))

    def __set_Mo__(self):
        ''' set the fast synaptic voltage coupling & self reset matrix'''
        I = np.eye(self.L.shape[0])
        self.Mo = self.s**2 * np.vstack((
            np.hstack((I, -I)),
            np.hstack((-I, I))
        ))

    def __set_Beta__(self):
        '''
        Set the input matrix to the new basis.
        Beta = U.T B U,
        Assigns Beta =  N x 2d matrix,
        '''
        dim = (self.D).shape[0]
        _, uA = np.linalg.eig(self.lds.A)

        self.Mc = np.zeros((self.N, dim))

        self.Beta= uA.T @ self.lds.B @ uA

        sD = np.diag(self.sD)

        self.Mc = np.vstack((sD @ self.Beta, - sD @ self.Beta))



        self.Mc = self.Mc @ self.Beta

    def __set_vth(self):
        ''' Set the threshold voltages'''
        I = np.eye(self.L.shape[0])
        II = np.hstack((I, -I)) / 2
        self.vth = np.asarray([
            np.linalg.norm(II @ self.D[:, j]) / 2
            for j in range(self.N)
        ])

    def __set_initial_conditions__(self):
        ''' Set initial V and r so that network estimate is initial condition of dynamical system'''
        I = np.eye(len(self.lds.X0))
        II = np.hstack((I, -I))
        self.r[:, 0] = nnls(II @ self.D, self.lds.X0)[0]
        xHat0 = II @ self.D @ self.r[:,0]
        assert(np.all(np.isclose(self.lds.X0,xHat0)))

    def get_net_estimate(self):
        ''' Get the network estimate by decoding activity'''
        I = np.eye(self.L.shape[0])
        II = np.hstack((I, -I))  # [I -I]
        return II @ self.D @ self.r

    def  run_sim(self):
        '''
        process data and call c_solver library to quickly run sims
        '''

        @jit(nopython=True)
        def fast_sim(dt, vth, num_pts, t, V, r, O, U, Mo, Mc, A_exp, spike_trans_prob, spike_nums):
            '''run the sim quickly using numba just-in-time compilation'''
            N = V.shape[0]
            max_spikes = len(O[0, :])
            for count in np.arange(num_pts - 1):
                state = np.hstack((V[:, count], r[:, count]))
                state = A_exp @ state
                r[:, count + 1] = state[N:]
                V[:, count + 1] = state[0:N] + dt * Mc @ U[:, count]

                diffs = (V[:, count + 1]) - vth
                if np.any(diffs > 0):
                    idx = np.argmax(diffs)
                    V[:, count + 1] -= Mo[:, idx]

                    if spike_trans_prob == 1:
                        r[idx, count + 1] += 1
                    else:
                        sample = np.random.uniform(0, 1)
                        if spike_trans_prob >= sample:
                            r[idx, count + 1] += 1
                    spike_num = spike_nums[idx]

                    if spike_num >= max_spikes:
                        print("Maximum Number of Spikes Exceeded in Simulation  for neuron ",
                              idx,
                              " having ", spike_num, " spikes with spike limit ", max_spikes)
                        assert (False)
                    O[idx, spike_num] = t[count]
                    spike_nums[idx] += 1

                t[count + 1] = t[count] + dt
                count += 1

        self.lds_data = self.lds.run_sim()
        #U = self.__rotate_input(self.lds_data['U'])
        U = np.zeros(self.lds_data['U'].shape)
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
        self.U = U
        spike_trans_prob  = self.spike_trans_prob

        bot = np.hstack((np.zeros((self.N, self.N)), -np.eye(self.N)))
        top = np.hstack((self.MvSelf + self.MvGJ, self.Mr))
        A_exp = np.vstack((top, bot)) #matrix exponential for linear part of eq
        A_exp = ( expm(A_exp*dt) ).astype(SpikingNeuralNet.FLOAT_TYPE)

        print('Starting Simulation.')
        fast_sim(dt, vth, num_pts, t, V, r, O, U, Mo, Mc, A_exp, spike_trans_prob, spike_nums)
        print('Simulation Complete.')

        return self.pack_data()

class SecondOrderSymSCNet(GapJunctionDeneveNet):
    def __init__(self, T, dt, N, D, lds, t0=0, spike_trans_prob=1):
        super().__init__(T, dt, N, D, lds, t0, spike_trans_prob)

        assert (self.__network_big_enough__())


        self.__add_second_order_vars__()
        self.__set_LK__()
        self.__set_uLam__()
        self.__set_IZero__()
        self.__set_s__()
        self.__set_new_decoder__()
        self.__set_Mv__()
        self.__set_MCa__()
        self.__set_Mr__()
        self.__set_Mu__()
        self.__set_Mo__()
        self.__set_vth__()
        self.__set_first_order_derivs__()
        self.__set_initial_conditions__()

    def __network_big_enough__(self):
        ''' check if the network is big enough to fully represent the d-dimensional state space'''
        return self.N >= 2 * self.dim

    def __add_second_order_vars__(self):
        ''' Allocate space for second order variables '''
        self.us = np.zeros(self.V.shape)
        self.v_dots = np.zeros(self.V.shape)

    def __set_IZero__(self):
        ''' sets I, II and Zero matrices of dims, dxd, dx2d, dxd respectively'''
        self.I = np.eye(self.dim)
        self.II = np.hstack((self.I, -self.I))
        self.zero = np.zeros(self.uA.shape)

    def __set_LK__(self):
        '''
        Set L and K symmetric and skew symmetric matrices of A
        '''
        self.L = (1 / 2) * (self.lds.A + self.lds.A.T)
        self.K = (1 / 2) * (self.lds.A - self.lds.A.T)
        self.M = - self.lds.A.T @ self.lds.A

    def __set_s__(self):
        '''
         set the decoder scaling factor s
        '''
        uD, sD, vt = np.linalg.svd(self.D[:, 0:self.dim])  # svd of 1st d cols of D
        self.s = sD[0]

    def __set_uLam__(self):
        ''' Diagonalize symmetric L matrix and set its bases and eigenvalues as u, lam respectively'''
        self.lam, self.uA = np.linalg.eig(self.M)

    def __set_new_decoder__(self):
        ''' Set the new matrix used to decode estimate from network'''
        u2 = np.vstack((
            np.hstack((self.uA, self.zero)),
            np.hstack((self.zero, self.uA))
        ))
        s2 = self.s * np.vstack((
            np.hstack((self.I, self.zero)),
            np.hstack((self.zero, self.I))
        ))
        self.D = u2 @ s2

    def __set_Mv__(self):
        ''' Set the Voltage Coupling Matrix '''
        ulu = self.uA.T @ self.L @ self.uA
        self.ulu = ulu
        self.Mv = 2 * np.vstack((
            np.hstack((ulu, self.zero)),
            np.hstack((self.zero, ulu))
        ))

        assert(is_diagonal(self.Mv))

    def __set_MCa__(self):
        ''' Set the skew symmetric gap junction voltage coupling matrix'''
        Lamb  = np.diag(self.lam)
        self.MCa = np.vstack((
            np.hstack((Lamb, self.zero)),
            np.hstack((self.zero, Lamb))
        ))
        self.Lamb = Lamb

        assert(is_diagonal(self.Mv))

    def __set_Mr__(self):
        ''' set the slow synaptic coupling matrix'''
        LamIulu = self.Lamb - np.eye(self.dim) - 2 * self.ulu

        self.Mr = self.s ** 2 * np.vstack((
            LamIulu @ self.II,
            -LamIulu @ self.II
        ))

    def __set_Mu__(self):
        ''' Set tilde{u} (2nd order filter) matrix'''
        self.Mu = 2 * self.s**2 * np.vstack((
            (self.ulu + self.I) @ self.II,
            -(self.ulu + self.I) @ self.II
        ))

    def __set_Mo__(self):
        ''' set the fast synaptic voltage coupling & self reset matrix'''
        self.Mo = self.s ** 2 * np.vstack((self.II, -self.II))

    def __set_Beta__(self):
        '''
        Set the input matrix to the new basis.
        Beta = U.T B U,
        Assigns Beta =  N x 2d matrix,
        '''
        dim = (self.D).shape[0]
        _, uA = np.linalg.eig(self.lds.A)

        self.Mc = np.zeros((self.N, dim))

        self.Beta = uA.T @ self.lds.B @ uA

        sD = np.diag(self.sD)

        self.Mc = np.vstack((sD @ self.Beta, - sD @ self.Beta))

        self.Mc = self.Mc @ self.Beta

    def __set_vth__(self):
        ''' Set the threshold voltages'''
        I = np.eye(self.L.shape[0])
        II = np.hstack((I, -I)) / 2
        self.vth = np.asarray([
            np.linalg.norm(II @ self.D[:, j]) / 2
            for j in range(self.N)
        ])

    def __set_first_order_derivs__(self):
        ''' set the first order derivative matrices for computing initial condition v_dot(0)'''
        D = self.D
        D_inv = np.linalg.pinv(D.T)
        self.fd_Mv = D.T @ self.II.T @ (self.L + self.K) @ self.II @ D_inv
        self.fd_Mr = D.T @ self.II.T @ (self.L + self.K + np.eye(self.dim)) @ self.II @ D
        self.fd_Mu = D.T @ self.II.T @ self.II @ D

    def __set_initial_conditions__(self):
        ''' Set initial V and r so that network estimate is initial condition of dynamical system'''
        self.r[:, 0] = nnls(self.II @ self.D, self.lds.X0)[0]
        self.us[:, 0] = self.r[:, 0]
        xHat0 = self.II @ self.D @ self.r[:, 0]
        assert (np.all(np.isclose(self.lds.X0, xHat0)))
        self.v_dots[:, 0] = (self.fd_Mv @ self.V[:, 0] + self.fd_Mr @ self.r[:, 0] - self.fd_Mu @ self.us[:, 0])

    def get_net_estimate(self):
        ''' Get the network estimate by decoding activity'''
        return self.II @ self.D @ self.r

    def run_sim(self):
        '''
        process data and call c_solver library to quickly run sims
        '''

        @jit(nopython=True)
        def fast_sim(dt, vth, num_pts, t, v_dots, V, r, us, O, Mo, state_transition, spike_nums):
            '''run the sim quickly using numba just-in-time compilation'''
            N = V.shape[0]
            max_spikes = len(O[0, :])
            for count in np.arange(num_pts - 1):

                state = np.hstack((v_dots[:, count], V[:, count], r[:, count], us[:, count]))
                state = state_transition @ state
                v_dots[:, count + 1] = state[0:N]
                V[:, count + 1] = state[N:2 * N]
                r[:, count + 1] = state[2 * N:3 * N]
                us[:, count + 1] = state[3 * N:]

                diffs = v_dots[:, count + 1] - vth
                if np.any(diffs > 0):
                    idx = np.argmax(diffs)
                    v_dots[:, count + 1] -= Mo[:, idx]
                    us[idx, count + 1] += 1

                    spike_num = spike_nums[idx]
                    if spike_num >= max_spikes:
                        print("Maximum Number of Spikes Exceeded in Simulation  for neuron ",
                              idx,
                              " having ", spike_num, " spikes with spike limit ", max_spikes)
                        assert (False)
                    O[idx, spike_num] = t[count]
                    spike_nums[idx] += 1

                t[count + 1] = t[count] + dt
                count += 1

        self.lds_data = self.lds.run_sim()
        vth = self.vth
        num_pts = self.num_pts
        t = np.asarray(self.t)
        V = self.V
        r = self.r
        us = self.us
        O = self.O
        v_dots = self.v_dots
        spike_nums = self.spike_nums
        Mv = self.Mv
        MCa = self.MCa
        Mo = self.Mo
        Mr = self.Mr
        Mu = self.Mu
        dt = self.dt
        ##self.U = U

        #          [N,    N, N, N]
        # state is [vdot, v, r, u] <==>[ Mv1, Mv2, -I, -I]
        zeroN = np.zeros((self.N, self.N))
        IN = np.eye(self.N)

        v_double_dot = np.hstack((Mv, MCa, Mr, Mu))
        v_dot = np.hstack((IN, zeroN, zeroN, zeroN))
        r_dot = np.hstack((zeroN, zeroN, -IN, IN))
        u_dot = np.hstack((zeroN, zeroN, zeroN, -IN))

        state_transition = np.vstack((v_double_dot, v_dot, r_dot, u_dot))
        state_transition = expm(state_transition * dt)

        print('Starting Simulation.')
        fast_sim(dt, vth, num_pts, t, v_dots, V, r, us, O, Mo, state_transition, spike_nums)
        print('Simulation Complete.')

        return self.pack_data()

class CSelfCoupledNet(GapJunctionDeneveNet):
    '''
    Extends Gap junction net with the following changes:
        - Mv is changed to derived diagonal matrix
        - D matrices are rotated to basis of A
        - A is diagonalized
        - Thresholds are scales by tau_s
    '''

    def __init__(self, T, dt, N, D, lds, t0=0, spike_trans_prob=1):
        super().__init__(T, dt, N, D, lds, t0, spike_trans_prob)

        dim = np.linalg.matrix_rank(D)

        assert (np.linalg.matrix_rank(lds.A) == np.min(
            lds.A.shape)), "Dynamics Matrix A is  not full rank, has dims (%i, %i) but rank %i" % (
        lds.A.shape[0], lds.A.shape[1], np.linalg.matrix_rank(lds.A))
        assert (N >= 2 * dim), "Must have at least 2 * dim neurons but rank(D) = %i and N = %i" % (dim, N)


        self.V = self.V.astype(np.complex128)
        self.r = self.r.astype(np.complex128)

        dim = lds.A.shape[0]
        uD, sD, vt = np.linalg.svd(D[:, 0:dim])  # svd of 1st two cols of D
        self.sD = sD
        self.vt = vt

        self.__set_r_theta__()
        self.__set_vth()
        self.__set_Mv__()
        self.__set_Beta__()
        self.__set_Mr__()
        self.__set_Mo__()
        self.set_initial_rs_rot()
        self.__rotate_decoder()

    def __set_r_theta__(self):
        ''' set the rotation matrix '''
        lam_A, uA = np.linalg.eig(self.lds.A)
        self.dim = self.lds.A.shape[0]
        sigma = np.real(lam_A)
        omega = np.imag(lam_A)



        # compute number of complex eigenvalues
        self.nc = np.sum([1 if np.iscomplex(lam) else 0 for lam in lam_A])
        assert(np.mod(self.nc, 2) == 0), "Odd Number of Complex Eigenvalues for dynamics A"
        self.nc = int(self.nc / 2)


        # for j in range(self.nc):
        #     omega[2*j] = -np.cos(omega[2*j])
        #     omega[2*j + 1] = np.sin(omega[2*j])

        sigma = np.abs(lam_A)
        omega = np.angle(lam_A)

        self.uA = np.hstack(( np.real(uA[:,0:1]), np.imag(uA[:,0:1]) ))
        for i in range(self.uA.shape[1]):
            self.uA[:,i] /= np.linalg.norm(self.uA[:,i])

        self.Sigma = np.diag(np.hstack((sigma, sigma)))
        self.Omega = (np.diag(np.hstack((omega, omega))))

        dim = self.dim


        #me
        # self.uA = np.zeros(self.uA.shape)
        # self.uA[0,1] = -1
        # self.uA[1,0] = 1



        # ##kwabena
        # self.uA = np.ones(self.uA.shape)
        # self.uA *= (1 / np.sqrt(2))
        # self.uA[0,1] *= -1
        # for each pair create a block matrix
        R_theta = np.eye(self.Sigma.shape[0])
        if self.nc:
            for j in np.arange(self.nc):
                dim_a = int(2 * j)
                dim_b = int(2 * j + 1)
                omega = self.Omega[dim_a, dim_a]
                R_theta[dim_a, dim_a] = np.cos(omega)
                R_theta[dim_b, dim_b] = np.cos(omega)
                R_theta[dim_a, dim_b] = np.sin(omega)
                R_theta[dim_b, dim_a] = -np.sin(omega)



                # add dim to j and repeat
                d = self.dim
                R_theta[dim_a + d : dim_a + d + 2, dim_a + d : dim_a + d + 2] \
                    = \
                    R_theta[dim_a : dim_a + 2, dim_a : dim_a + 2]


        self.R_theta = R_theta





        self.uA_tilde = self.uA
        # u s rtheta uinv = a
    def __set_Mv__(self):
        ''' Set the Diagonal Voltage Leak Matrix using d x d Matrix A and Neurons N
        Assigns Mc =  N x N matrix having the top 2 d diagonals from the eigvals of A (repeated once)
        '''
        uA = self.uA
        utu = np.linalg.inv(uA.T @ uA)
        zs = np.zeros(utu.shape)
        utu = np.vstack((
                        np.hstack((utu, zs)),
                        np.hstack((utu, zs))
        ))
        S = np.diag(self.sD)
        S = np.hstack((S, -S))
        Sinv = np.diag(1/self.sD)
        Sinv = np.hstack((Sinv, -Sinv))
        dim = self.dim
        #self.Mv =  S.T @ self.Sigma[0:dim, 0:dim] @ self.R_theta[0:dim,0:dim] @ Sinv
        self.Mv =  self.Sigma @ self.R_theta



        #self.Mv = self.Sigma.astype(np.complex128) + self.Omega

    def __set_Beta__(self):
        '''
        Set the input matrix to the new basis.
        Beta = U.T B U,
        Assigns Beta =  N x 2d matrix,
        '''
        dim = (self.D).shape[0]

        self.Mc = np.zeros((self.N, dim))

        self.Beta = self.uA_tilde.T@ (self.lds.B) @ self.uA_tilde

        sD = np.diag(self.sD)

        self.Mc = np.vstack((sD @ self.Beta, - sD @ self.Beta))
        self.Mc = self.Mc @ self.Beta

    def __set_Mr__(self):
        ''' Set the Post-synaptic matrix via SVD of D
        Assigns self.Mr = S @ (L + I) @ S.T, where D = USV.T and A =ULU.T, as an NxN matrix
        Also Sets D = uA @ S
        '''
        S = np.diag(self.sD)
        S = np.hstack((S, -S))
        dim = self.D.shape[0]
        self.Mr = np.zeros((self.N, self.N))
        self.Mr[0: 2 * dim, 0: 2 * dim] =  S.T @ (self.Mv[0:dim, 0:dim] + np.eye(dim)) @ S


        #self.Mr[0: 2 * dim, 0: 2 * dim] = S.T @ (np.eye(dim) + self.Sigma[0:dim, 0:dim] + self.Omega[0:dim, 0:dim]) @ S

    def __set_Mo__(self):
        '''
        Set the Voltage Fast Reset Matrix
        Assigns self.Mo = S.T @ S where D = USV.T, as an NxN matrix
        '''

        S = np.diag(self.sD)
        S = np.hstack((S, -S))
        dim = self.D.shape[0]
        self.Mo = np.zeros((self.N, self.N))
        self.Mo[0: 2 * dim, 0: 2 * dim] = - S.T @ S

    def __set_vth(self):
        ''' Set the threshold voltages for the rotated neurons'''

        dim = self.D.shape[0]
        sD = np.hstack((self.sD, self.sD, np.zeros((self.N - 2 * dim,))))
        self.vth = np.square(sD) / 2

    def __rotate_input(self, U):
        '''Given a d x T time series of input, rotate to new basis uA'''

        return self.uA_tilde.T @ U

    def set_initial_rs_rot(self):

        dim = self.lds.A.shape[0]
        S = np.hstack((np.diag(self.sD), np.diag(-self.sD), np.zeros((dim, self.N - 2 * dim))))

        lam_vec, uA = np.linalg.eig(self.lds.A)
        ux0 = uA.conjugate().T @ self.lds.X0
        phi0s = np.angle(ux0)



        #y0 =  np.asarray(self.lds.X0.copy())

        #y0 = np.asarray([ y0[0] * np.cos(phi0s)[0], y0[1] * np.sin(phi0s)[0]]) * np.sqrt(2)
        y0 = np.linalg.inv(self.uA) @ self.lds.X0

        self.r[:, 0] = nnls(S, y0)[0]
        self.V[:,0] = S.T @ (y0 - S @ self.r[:,0])

        #print((y0 - S @ self.r[:,0]))








    def __rotate_decoder(self):
        ''' Move the assigned decoder to rotated basis. overwrites self.D!'''
        rot_mtx = self.R_theta[0:self.dim, 0:self.dim]
        self.D = np.hstack(( np.diag(self.sD), np.diag(-self.sD)))

    def run_sim(self):
        '''
        process data and call c_solver library to quickly run sims
        '''
        self.lds_data = self.lds.run_sim()
        U = self.__rotate_input(self.lds_data['U']).astype(np.complex128)
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
        Mc = self.Mc.astype(np.complex128)
        dt = self.dt
        self.U = U
        spike_trans_prob = self.spike_trans_prob

        bot = np.hstack((np.zeros((self.N, self.N)), -np.eye(self.N))).astype(np.complex128)
        top = np.hstack((Mv, Mr))
        A_exp = np.vstack((top, bot))
        A_exp = (expm(A_exp * dt))


        print('Starting Simulation.')
        fast_sim(dt, vth, num_pts, t, V, r, O, U, Mr, Mo, Mc, A_exp, spike_trans_prob, spike_nums)
        print('Simulation Complete.')

        return self.pack_data()

class OLDSecondOrderSCNet(GapJunctionDeneveNet):
    '''
      Extends Gap junction net with the following changes:
          - Mv is changed to derived diagonal matrix
          - D matrices are rotated to basis of A
          - A is diagonalized
          - Thresholds are scales by tau_s
      '''

    def __init__(self, T, dt, N, D, lds, t0=0, spike_trans_prob=1):
        super().__init__(T, dt, N, D, lds, t0, spike_trans_prob)

        self.dim = lds.A.shape[0]
        assert (np.linalg.matrix_rank(lds.A) == np.min(
            lds.A.shape)), "Dynamics Matrix A is  not full rank, has dims (%i, %i) but rank %i" % (
            lds.A.shape[0], lds.A.shape[1], np.linalg.matrix_rank(lds.A))
        assert (N >= 2 * self.dim), "Must have at least 2 * dim neurons but rank(D) = %i and N = %i" % (dim, N)


        self.__set_lambda__()
        self.__set_Mv1__()
        self.__set_Mv2__()
        self.__set_Mr__()
        self.__set_Mu__()
        self.__set_Mo__()
        self.__set_vth__()
        self.__set_first_order_derivs__()
        self.__set_initial_conditions__()
        self.__set_decoder__()
        #self.__set_Beta__()
        #self.__rotate_decoder__()

    def __set_lambda__(self):
        '''
        compute and set lambda dynamics matrix & u bases.
        Only implemented for 2x2 complex eigenvalued A
        '''
        eig_A, uA = np.linalg.eig(self.lds.A)
        uA = np.hstack((np.real(uA[:,0:1]), np.imag(uA[:,0:1])))
        uA_inv = np.linalg.inv(uA)
        self.uA = uA
        self.uA_inv = uA_inv

        theta = np.angle(eig_A[0])
        sin = np.sin(theta)
        cos = np.cos(theta)
        L = np.vstack((
            np.hstack((
                0, 0
            )),
            np.hstack((
                -sin - (cos**2) / sin,  2 * cos
            ))
        ))

        L *= np.abs(eig_A[0])

        Lam, uLam = np.linalg.eig(L)

        uLam =  uA @ uLam


        self.Lam = np.diag(Lam)
        self.uLam = uLam
        self.uTu = uLam.T @ uLam

        if np.allclose(self.Lam, 0):
            self.Lam = self.uTu_inv = np.zeros((self.dim, self.dim))
        else:
            self.uLam = uLam
            self.uTu_inv = np.linalg.inv(self.uTu)

    def __set_Mv1__(self):
        ''' Set Mv derivative coupling'''
        Mv = self.uTu @ self.Lam @ self.uTu_inv
        self.Mv1 = np.vstack((
            np.hstack(( Mv, -Mv)),
            np.hstack((-Mv,  Mv))
        ))

    def __set_Mv2__(self):
        ''' Set Mv coupling'''
        Mv = (1/2) * self.uTu @ np.square(self.Lam) @ self.uTu_inv
        self.Mv2 =  np.vstack((
            np.hstack((Mv, -Mv)),
            np.hstack((-Mv, Mv))
        ))

    def __set_Mr__(self):
        ''' set rho coupling matrix'''
        uD, sD, vt = np.linalg.svd(self.D[:, 0 : self.dim])  # svd of 1st two cols of D
        self.sD = sD
        self.vt = vt
        self.s = sD[0]

        Mr = self.uTu @ (
             np.square(self.Lam) + 2 * self.Lam + np.eye(self.dim)
        )
        self.Mr = (self.s**2 / 4) * np.vstack((
            np.hstack(( Mr, -Mr)),
            np.hstack((-Mr,  Mr))
        ))

    def __set_Mu__(self):
        Mu = (self.s**2 / 2) * self.uTu @ (self.Lam + np.eye(self.dim))
        self.Mu = np.vstack((
            np.hstack(( Mu, -Mu)),
            np.hstack((-Mu,  Mu))
        ))

    def __set_Beta__(self):
        '''
        Set the input matrix to the new basis.
        Beta = U.T B U,
        Assigns Beta =  N x 2d matrix,
        '''
        dim = (self.D).shape[0]

        self.Mc = np.zeros((self.N, dim))

        self.Beta = self.uA_tilde.T @ (self.lds.B) @ self.uA_tilde

        sD = np.diag(self.sD)

        self.Mc = np.vstack((sD @ self.Beta, - sD @ self.Beta))
        self.Mc = self.Mc @ self.Beta

    def __set_Mo__(self):
        '''
        Set the Voltage Fast Reset Matrix
        '''
        self.Mo = (self.s ** 2 / 4 ) * np.vstack((
            np.hstack(( self.uTu, -self.uTu)),
            np.hstack((-self.uTu,  self.uTu))
        ))

    def __set_vth__(self):
        ''' Set the threshold voltages for the rotated neurons'''
        self.vth = self.s**2 / 2 * np.ones((self.Mv.shape[0],))

    def __rotate_input(self, U):
        '''Given a d x T time series of input, rotate to new basis uA'''
        return self.uA_tilde.T @ U

    def __set_first_order_derivs__(self):
        S = self.s * np.eye(self.dim)
        S = np.hstack((S, -S))
        D = S
        D_inv = np.linalg.pinv(D.T)
        self.fd_Mv = D.T @ self.Lam @ D_inv
        self.fd_Mr = D.T @ (self.Lam + np.eye(2)) @ D
        self.fd_Mu = D.T @ D

    def __set_initial_conditions__(self):
        ''' set rho(0) and u(o) '''
        S = self.s * np.eye(self.dim)
        S = np.hstack((S, -S))

        self.r[:, 0] = nnls(S, self.uLam.T @ self.lds.X0)[0]
        
        self.us = np.zeros(self.V.shape)
        self.us[:,0] = self.r[:,0]

        self.v_dots = np.zeros(self.V.shape)
        self.v_dots[:, 0] = (self.fd_Mv @ self.V[:, 0] + self.fd_Mr @ self.r[:, 0] - self.fd_Mu @ self.us[:, 0])

    def __set_decoder__(self):
        ''' compute the new decoder using u [s -s] [vt, 0; 0, vt]'''
        S = self.s * np.eye(self.dim)
        self.D = .5 * self.uLam @ np.hstack((S, -S))

    def run_sim(self):
        '''
        process data and call c_solver library to quickly run sims
        '''

        @jit(nopython=True)
        def fast_sim(dt, vth, num_pts, t, v_dots, V, r, us, O, Mo, state_transition, spike_nums):
            '''run the sim quickly using numba just-in-time compilation'''
            N = V.shape[0]
            max_spikes = len(O[0, :])
            for count in np.arange(num_pts - 1):
                state = np.hstack((v_dots[:,count], V[:, count], r[:, count], us[:,count]))
                state = state_transition @ state
                v_dots[:,count + 1] = state[0:N]
                V[:, count + 1] = state[N:2 * N]
                r[:, count + 1] = state[2 * N:3 * N]
                us[:,count + 1] = state[3 * N:]

                diffs = v_dots[:, count + 1] - vth
                if np.any(diffs > 0):
                    idx = np.argmax(diffs)
                    v_dots[:,count + 1] -= Mo[:,idx]
                    us[idx, count + 1] += 1

                    spike_num = spike_nums[idx]
                    if spike_num >= max_spikes:
                        print("Maximum Number of Spikes Exceeded in Simulation  for neuron ",
                              idx,
                              " having ", spike_num, " spikes with spike limit ", max_spikes)
                        assert (False)
                    O[idx, spike_num] = t[count]
                    spike_nums[idx] += 1

                t[count + 1] = t[count] + dt
                count += 1

        self.lds_data = self.lds.run_sim()
        #U = self.__rotate_input(self.lds_data['U']).astype(np.complex128)
        vth = self.vth
        num_pts = self.num_pts
        t = np.asarray(self.t)
        V = self.V
        r = self.r
        us = self.us
        O = self.O
        v_dots = self.v_dots
        spike_nums = self.spike_nums
        Mv1 = self.Mv1
        Mv2 = self.Mv2
        Mo = self.Mo
        Mr = self.Mr
        Mu = self.Mu

        #Mc = self.Mc.astype(np.complex128)
        dt = self.dt
        ##self.U = U
        ##         [N,    N, N, N]
        ##state is [vdot, v, r, u] <==>[ Mv1, Mv2, -I, -I]
        zero = np.zeros((self.N, self.N))
        I = np.eye(self.N)
        v_double_dot = np.hstack((zero, Mv2, -Mr, Mu))
        v_dot = np.hstack((I, zero, zero , zero))
        r_dot = np.hstack((zero, zero, -I, I))
        u_dot = np.hstack((zero, zero, zero, -I))

        state_transition = np.vstack((v_double_dot, v_dot, r_dot, u_dot))
        state_transition = expm(state_transition * dt)

        print('Starting Simulation.')
        fast_sim(dt, vth, num_pts, t, v_dots, V, r, us, O, Mo, state_transition, spike_nums)
        print('Simulation Complete.')

        return self.pack_data()

# class SecondOrderSCNet(GapJunctionDeneveNet):
#     '''
#       Extends Gap junction net with the following changes:
#           - Mv is changed to derived diagonal matrix
#           - D matrices are rotated to basis of A
#           - A is diagonalized
#           - Thresholds are scales by tau_s
#       '''
#
#     def __init__(self, T, dt, N, D, lds, t0=0, spike_trans_prob=1):
#         super().__init__(T, dt, N, D, lds, t0, spike_trans_prob)
#
#         self.dim = lds.A.shape[0]
#         assert (np.linalg.matrix_rank(lds.A) == np.min(
#             lds.A.shape)), "Dynamics Matrix A is  not full rank, has dims (%i, %i) but rank %i" % (
#             lds.A.shape[0], lds.A.shape[1], np.linalg.matrix_rank(lds.A))
#         assert (N >= 2 * self.dim), "Must have at least 2 * dim neurons but rank(D) = %i and N = %i" % (dim, N)
#
#         self.s = np.linalg.svd(D[:,0:self.dim])[1][0]
#
#         self.__set_LKM__()
#         self.__set_Mv1__()
#         self.__set_Mv2__()
#         self.__set_Mr__()
#         self.__set_Mu__()
#         self.__set_Mo__()
#         self.__set_first_order_derivs__()
#         self.__set_decoder__()
#         self.__set_vth__()
#
#
#         self.__set_initial_conditions__()
#         # self.__set_Beta__()
#         # self.__rotate_decoder__()
#
#     #TODO: Set Matrices all Symmetric
#
#     def __set_LKM__(self):
#         '''
#         Set L and K symmetric and skew symmetric matrices of A
#         '''
#         self.L = (1 / 2) * (self.lds.A + self.lds.A.T)
#         self.K = (1 / 2) * (self.lds.A - self.lds.A.T)
#
#         self.M = (self.K @ self.K) - (self.L @ self.L) - (self.L @ self.K + (self.L @ self.K).T)
#         assert (np.all(self.M) == np.all(self.M.T)), "M={0} not Symmetric".format(self.M)
#
#     def __set_Mv1__(self):
#         ''' Set Mv derivative coupling'''
#         L = np.hstack((self.L, -self.L))
#         self.Mv1 = np.vstack((L, -L))
#         assert(np.all(self.Mv1)==np.all(self.Mv1.T)), "Matrix Mv1={0} not symmetric".format(self.Mv1)
#
#     def __set_Mv2__(self):
#         ''' Set Mv coupling'''
#         M = np.hstack((self.M, -self.M))
#         self.Mv2 = (1 / 2) * np.vstack((M, -M))
#         assert(np.all(self.Mv2)==np.all(self.Mv2.T)), "Matrix Mv1={0} not symmetric".format(self.Mv2)
#
#     def __set_Mr__(self):
#         ''' set rho coupling matrix'''
#         M2 = self.M - (np.eye(self.dim) + 2 * self.L)
#         M2 = np.hstack((M2, -M2))
#         self.Mr = (self.s**2 / 4) * np.vstack((M2, -M2))
#
#     def __set_Mu__(self):
#         L = self.L + np.eye(self.dim)
#         L = np.hstack((L, -L))
#         self.Mu = (self.s**2 / 2) * np.vstack((L, -L))
#
#     def __set_Beta__(self):
#         '''
#         Set the input matrix to the new basis.
#         Beta = U.T B U,
#         Assigns Beta =  N x 2d matrix,
#         '''
#         dim = (self.D).shape[0]
#
#         self.Mc = np.zeros((self.N, dim))
#
#         self.Beta = self.uA_tilde.T @ (self.lds.B) @ self.uA_tilde
#
#         sD = np.diag(self.sD)
#
#         self.Mc = np.vstack((sD @ self.Beta, - sD @ self.Beta))
#         self.Mc = self.Mc @ self.Beta
#
#     def __set_Mo__(self):
#         '''
#         Set the Voltage Fast Reset Matrix
#         '''
#         I = np.eye(self.dim)
#         I = np.hstack((I, -I))
#         self.Mo = (self.s**2 / 4) * np.vstack((I, -I))
#
#     def __set_vth__(self):
#         ''' Set the threshold voltages for the rotated neurons'''
#         self.vth = (self.s**2 / 2) * np.ones((self.N,))
#
#     def __set_first_order_derivs__(self):
#
#         self.fd_Mv =  (1 / 2) * (self.L + self.K)
#         self.fd_Mv = np.hstack((self.fd_Mv, -self.fd_Mv))
#         self.fd_Mv = np.vstack((self.fd_Mv, -self.fd_Mv))
#
#         ILK = np.eye(self.dim) + self.L + self.K
#         ILK = np.hstack((ILK, -ILK))
#         self.fd_Mr = (self.s**2 / 4) * np.vstack((ILK, -ILK))
#
#         I = np.eye(self.dim)
#         I = np.hstack((I, -I))
#         self.fd_Mu = (self.s**2 / 4) * np.vstack((I, -I))
#
#     def __set_initial_conditions__(self):
#         ''' set rho(0) and u(o) '''
#         self.r[:, 0] = nnls(self.D, self.lds.X0)[0]
#         self.us = np.zeros(self.V.shape)
#         self.us[:,0] = self.r[:,0]
#         self.v_dots = np.zeros(self.V.shape)
#         #self.V[:, 0] = self.D.T @ (self.lds.X0 - self.D @ self.r[:,0])
#         self.v_dots[:,0] = (self.fd_Mv @ self.V[:,0] + self.fd_Mr @ self.r[:,0] - self.fd_Mu @ self.us[:,0])
#
#     def __set_decoder__(self):
#         ''' compute the new decoder using u [s -s] [vt, 0; 0, vt]'''
#         sI = self.s * np.eye(self.dim)
#         self.D = .5 * np.hstack((sI, -sI))
#
#     def run_sim(self):
#         '''
#         process data and call c_solver library to quickly run sims
#         '''
#
#         @jit(nopython=True)
#         def fast_sim(dt, vth, num_pts, t, v_dots, V, r, us, O, Mo, state_transition, spike_nums):
#             '''run the sim quickly using numba just-in-time compilation'''
#             N = V.shape[0]
#             max_spikes = len(O[0, :])
#             for count in np.arange(num_pts - 1):
#
#                 state = np.hstack((v_dots[:, count], V[:, count], r[:, count], us[:, count]))
#                 state = state_transition @ state
#                 v_dots[:, count + 1] = state[0:N]
#                 V[:, count + 1] = state[N:2 * N]
#                 r[:, count + 1] = state[2 * N:3 * N]
#                 us[:, count + 1] = state[3 * N:]
#
#                 diffs = v_dots[:, count + 1] - vth
#                 if np.any(diffs > 0):
#                     idx = np.argmax(diffs)
#                     v_dots[:, count + 1] -= Mo[:, idx]
#                     us[idx, count + 1] += 1
#
#                     spike_num = spike_nums[idx]
#                     if spike_num >= max_spikes:
#                         print("Maximum Number of Spikes Exceeded in Simulation  for neuron ",
#                               idx,
#                               " having ", spike_num, " spikes with spike limit ", max_spikes)
#                         assert (False)
#                     O[idx, spike_num] = t[count]
#                     spike_nums[idx] += 1
#
#
#
#                 t[count + 1] = t[count] + dt
#                 count += 1
#
#         self.lds_data = self.lds.run_sim()
#         vth = self.vth
#         num_pts = self.num_pts
#         t = np.asarray(self.t)
#         V = self.V
#         r = self.r
#         us = self.us
#         O = self.O
#         v_dots = self.v_dots
#         spike_nums = self.spike_nums
#         Mv1 = self.Mv1
#         Mv2 = self.Mv2
#         Mo = self.Mo
#         Mr = self.Mr
#         Mu = self.Mu
#         fd_Mv = self.fd_Mv
#         fd_Mr = self.fd_Mr
#         fd_Mu = self.fd_Mu
#         dt = self.dt
#
#         ##self.U = U
#         ##         [N,    N, N, N]
#         ##state is [vdot, v, r, u] <==>[ Mv1, Mv2, -I, -I]
#         zero = np.zeros((self.N, self.N))
#         I = np.eye(self.N)
#         v_double_dot = np.hstack((Mv1, Mv2, Mr, Mu))
#         v_dot = np.hstack((I, zero, zero, zero))
#         r_dot = np.hstack((zero, zero, -I, I))
#         u_dot = np.hstack((zero, zero, zero, -I))
#
#         state_transition = np.vstack((v_double_dot, v_dot, r_dot, u_dot))
#         state_transition = expm(state_transition * dt)
#
#         print('Starting Simulation.')
#         fast_sim(dt, vth, num_pts, t, v_dots, V, r, us, O, Mo, state_transition, spike_nums)
#         print('Simulation Complete.')
#
#         return self.pack_data()

class SecondOrderGJNet(GapJunctionDeneveNet):
    '''
      Extends Gap junction net with the following changes:
          - Mv is changed to derived diagonal matrix
          - D matrices are rotated to basis of A
          - A is diagonalized
          - Thresholds are scales by tau_s
      '''

    def __init__(self, T, dt, N, D, lds, t0=0, spike_trans_prob=1):
        super().__init__(T, dt, N, D, lds, t0, spike_trans_prob)

        self.dim = lds.A.shape[0]
        assert (np.linalg.matrix_rank(lds.A) == np.min(
            lds.A.shape)), "Dynamics Matrix A is  not full rank, has dims (%i, %i) but rank %i" % (
            lds.A.shape[0], lds.A.shape[1], np.linalg.matrix_rank(lds.A))
        assert (N >= 2 * self.dim), "Must have at least 2 * dim neurons but rank(D) = %i and N = %i" % (dim, N)

        self.__set_LK__()
        self.__set_Mv1__()
        self.__set_Mv2__()
        self.__set_Mr__()
        self.__set_Mu__()
        self.__set_Mo__()
        self.__set_vth__()
        self.__set_first_order_derivs__()
        self.__set_initial_conditions__()
        # self.__set_Beta__()
        # self.__rotate_decoder__()

    def __set_LK__(self):
        '''
        Set L and K symmetric and skew symmetric matrices of A
        '''
        self.L = (1 / 2) * (self.lds.A + self.lds.A.T)
        self.K = (1 / 2) * (self.lds.A - self.lds.A.T)

    def __set_Mv1__(self):
        ''' Set Mv derivative coupling'''
        D = self.D
        D_inv = np.linalg.pinv(D.T)
        self.Mv1 =  2 * D.T @ self.L @ D_inv

    def __set_Mv2__(self):
        ''' Set Mv coupling'''
        D = self.D
        D_inv = np.linalg.pinv(D.T)

        M = (self.K @ self.K) - (self.L @ self.L) - (self.L @ self.K + (self.L @ self.K).T)
        assert(np.all(M)==np.all(M.T)), "M={0} not Symmetric".format(M)

        self.Mv2 = D.T @ M @ D_inv
        self.M = M

    def __set_Mr__(self):
        ''' set rho coupling matrix'''
        D = self.D
        D_inv = np.linalg.pinv(D.T)
        self.Mr = D.T @ (self.M - (np.eye(self.dim) + 2 * self.L)) @ D

    def __set_Mu__(self):
        D = self.D
        D_inv = np.linalg.pinv(D.T)
        self.Mu = 2 * D.T @ (self.L + np.eye(2)) @ D

    def __set_Beta__(self):
        '''
        Set the input matrix to the new basis.
        Beta = U.T B U,
        Assigns Beta =  N x 2d matrix,
        '''
        dim = (self.D).shape[0]

        self.Mc = np.zeros((self.N, dim))

        self.Beta = self.uA_tilde.T @ (self.lds.B) @ self.uA_tilde

        sD = np.diag(self.sD)

        self.Mc = np.vstack((sD @ self.Beta, - sD @ self.Beta))
        self.Mc = self.Mc @ self.Beta

    def __set_Mo__(self):
        '''
        Set the Voltage Fast Reset Matrix
        '''
        self.Mo = self.D.T @ self.D

    def __set_vth__(self):
        ''' Set the threshold voltages for the rotated neurons'''
        self.vth = np.diag(self.D.T @ self.D) / 2

    def __set_first_order_derivs__(self):
        D = self.D
        D_inv = np.linalg.pinv(D.T)
        self.fd_Mv = D.T @ (self.L + self.K) @ D_inv
        self.fd_Mr = D.T @ (self.L + self.K + np.eye(self.dim)) @ D
        self.fd_Mu = D.T @ D

    def __set_initial_conditions__(self):
        ''' set rho(0) and u(o) '''
        self.r[:, 0] = nnls(self.D, self.lds.X0)[0]
        self.us = np.zeros(self.V.shape)
        self.us[:,0] = self.r[:,0]
        self.v_dots = np.zeros(self.V.shape)
        #self.V[:, 0] = self.D.T @ (self.lds.X0 - self.D @ self.r[:,0])
        self.v_dots[:,0] = (self.fd_Mv @ self.V[:,0] + self.fd_Mr @ self.r[:,0] - self.fd_Mu @ self.us[:,0])


    def run_sim(self):
        '''
        process data and call c_solver library to quickly run sims
        '''

        @jit(nopython=True)
        def fast_sim(dt, vth, num_pts, t, v_dots, V, r, us, O, Mo, state_transition, spike_nums):
            '''run the sim quickly using numba just-in-time compilation'''
            N = V.shape[0]
            max_spikes = len(O[0, :])
            for count in np.arange(num_pts - 1):

                state = np.hstack((v_dots[:, count], V[:, count], r[:, count], us[:, count]))
                state = state_transition @ state
                v_dots[:, count + 1] = state[0:N]
                V[:, count + 1] = state[N:2 * N]
                r[:, count + 1] = state[2 * N:3 * N]
                us[:, count + 1] = state[3 * N:]

                diffs = v_dots[:, count + 1] - vth
                if np.any(diffs > 0):
                    idx = np.argmax(diffs)
                    v_dots[:, count + 1] -= Mo[:, idx]
                    us[idx, count + 1] += 1

                    spike_num = spike_nums[idx]
                    if spike_num >= max_spikes:
                        print("Maximum Number of Spikes Exceeded in Simulation  for neuron ",
                              idx,
                              " having ", spike_num, " spikes with spike limit ", max_spikes)
                        assert (False)
                    O[idx, spike_num] = t[count]
                    spike_nums[idx] += 1



                t[count + 1] = t[count] + dt
                count += 1

        self.lds_data = self.lds.run_sim()
        vth = self.vth
        num_pts = self.num_pts
        t = np.asarray(self.t)
        V = self.V
        r = self.r
        us = self.us
        O = self.O
        v_dots = self.v_dots
        spike_nums = self.spike_nums
        Mv1 = self.Mv1
        Mv2 = self.Mv2
        Mo = self.Mo
        Mr = self.Mr
        Mu = self.Mu
        fd_Mv = self.fd_Mv
        fd_Mr = self.fd_Mr
        fd_Mu = self.fd_Mu
        dt = self.dt

        ##self.U = U
        ##         [N,    N, N, N]
        ##state is [vdot, v, r, u] <==>[ Mv1, Mv2, -I, -I]
        zero = np.zeros((self.N, self.N))
        I = np.eye(self.N)
        v_double_dot = np.hstack((Mv1, Mv2, Mr, Mu))
        v_dot = np.hstack((I, zero, zero, zero))
        r_dot = np.hstack((zero, zero, -I, I))
        u_dot = np.hstack((zero, zero, zero, -I))

        state_transition = np.vstack((v_double_dot, v_dot, r_dot, u_dot))
        state_transition = expm(state_transition * dt)

        print('Starting Simulation.')
        fast_sim(dt, vth, num_pts, t, v_dots, V, r, us, O, Mo, state_transition, spike_nums)
        print('Simulation Complete.')

        return self.pack_data()

class SecondOrderSCNet(SecondOrderGJNet):
    def __init__(self, T, dt, N, D, lds, t0=0, spike_trans_prob=1):

        self.s = np.linalg.svd(D)[1][0]
        sI = self.s * np.eye(D.shape[0])
        self.D = .5 * np.hstack((sI, -sI))
        super().__init__(T, dt, N, D, lds, t0, spike_trans_prob)

class OLDSecondOrderGJNet(GapJunctionDeneveNet):
    '''
      Extends Gap junction net with the following changes:
          - Mv is changed to derived diagonal matrix
          - D matrices are rotated to basis of A
          - A is diagonalized
          - Thresholds are scales by tau_s
      '''

    def __init__(self, T, dt, N, D, lds, t0=0, spike_trans_prob=1):
        super().__init__(T, dt, N, D, lds, t0, spike_trans_prob)

        self.dim = lds.A.shape[0]
        assert (np.linalg.matrix_rank(lds.A) == np.min(
            lds.A.shape)), "Dynamics Matrix A is  not full rank, has dims (%i, %i) but rank %i" % (
            lds.A.shape[0], lds.A.shape[1], np.linalg.matrix_rank(lds.A))
        assert (N >= 2 * self.dim), "Must have at least 2 * dim neurons but rank(D) = %i and N = %i" % (dim, N)

        self.__set_L__()
        self.__set_Mv1__()
        self.__set_Mv2__()
        self.__set_Mr__()
        self.__set_Mu__()
        self.__set_Mo__()
        self.__set_vth__()
        self.__set_first_order_derivs__()
        self.__set_initial_conditions__()
        # self.__set_Beta__()
        # self.__rotate_decoder__()

    def __set_L__(self):
        '''
        compute and set lambda dynamics matrix & u bases.
        Only implemented for 2x2 complex eigenvalued A
        '''
        eig_A, uA = np.linalg.eig(self.lds.A)
        theta = np.angle(eig_A[0])
        sin = np.sin(theta)
        cos = np.cos(theta)
        L = np.vstack((
            np.hstack((
                0, 0
            )),
            np.hstack((
                -sin - (cos ** 2) / sin, 2 * cos
            ))
        ))
        self.L = L * np.abs(eig_A[0])

        self.L = self.lds.A

    def __set_Mv1__(self):
        ''' Set Mv derivative coupling'''
        D = self.D
        D_inv = np.linalg.pinv(D.T)
        self.Mv1 = D.T @ self.L @ D_inv

    def __set_Mv2__(self):
        ''' Set Mv coupling'''
        D = self.D
        D_inv = np.linalg.pinv(D.T)
        self.Mv2 = D.T @ np.square(self.L) @ D_inv

    def __set_Mr__(self):
        ''' set rho coupling matrix'''
        D = self.D
        D_inv = np.linalg.pinv(D.T)
        self.Mr = D.T @ (self.L + np.eye(2)) @ D

    def __set_Mu__(self):
        D = self.D
        D_inv = np.linalg.pinv(D.T)
        self.Mu = D.T @ (self.L + 2 * np.eye(2)) @ D

    def __set_Beta__(self):
        '''
        Set the input matrix to the new basis.
        Beta = U.T B U,
        Assigns Beta =  N x 2d matrix,
        '''
        dim = (self.D).shape[0]

        self.Mc = np.zeros((self.N, dim))

        self.Beta = self.uA_tilde.T @ (self.lds.B) @ self.uA_tilde

        sD = np.diag(self.sD)

        self.Mc = np.vstack((sD @ self.Beta, - sD @ self.Beta))
        self.Mc = self.Mc @ self.Beta

    def __set_Mo__(self):
        '''
        Set the Voltage Fast Reset Matrix
        '''
        self.Mo = self.D.T @ self.D

    def __set_vth__(self):
        ''' Set the threshold voltages for the rotated neurons'''
        self.vth = np.diag(self.D.T @ self.D) / 2

    def __set_first_order_derivs__(self):
        D = self.D
        D_inv = np.linalg.pinv(D.T)
        self.fd_Mv = D.T @ self.L @ D_inv
        self.fd_Mr = D.T @ (self.L + np.eye(2)) @ D
        self.fd_Mu = D.T @ D

    def __set_initial_conditions__(self):
        ''' set rho(0) and u(o) '''
        self.r[:, 0] = nnls(self.D, self.lds.X0)[0]
        self.us = np.zeros(self.V.shape)
        self.us[:, 0] = self.r[:, 0]
        self.v_dots = np.zeros(self.V.shape)
        # self.V[:, 0] = self.D.T @ (self.lds.X0 - self.D @ self.r[:,0])
        self.v_dots[:, 0] = (self.fd_Mv @ self.V[:, 0] + self.fd_Mr @ self.r[:, 0] - self.fd_Mu @ self.us[:, 0])

    def run_sim(self):
        '''
        process data and call c_solver library to quickly run sims
        '''

        @jit(nopython=True)
        def fast_sim(dt, vth, num_pts, t, v_dots, V, r, us, O, Mo, state_transition, spike_nums):
            '''run the sim quickly using numba just-in-time compilation'''
            N = V.shape[0]
            max_spikes = len(O[0, :])
            for count in np.arange(num_pts - 1):

                state = np.hstack((v_dots[:, count], V[:, count], r[:, count], us[:, count]))
                state = state_transition @ state
                v_dots[:, count + 1] = state[0:N]
                V[:, count + 1] = state[N:2 * N]
                r[:, count + 1] = state[2 * N:3 * N]
                us[:, count + 1] = state[3 * N:]

                diffs = v_dots[:, count + 1] - vth
                if np.any(diffs > 0):
                    idx = np.argmax(diffs)
                    v_dots[:, count + 1] -= Mo[:, idx]
                    us[idx, count + 1] += 1

                    spike_num = spike_nums[idx]
                    if spike_num >= max_spikes:
                        print("Maximum Number of Spikes Exceeded in Simulation  for neuron ",
                              idx,
                              " having ", spike_num, " spikes with spike limit ", max_spikes)
                        assert (False)
                    O[idx, spike_num] = t[count]
                    spike_nums[idx] += 1

                t[count + 1] = t[count] + dt
                count += 1

        self.lds_data = self.lds.run_sim()
        vth = self.vth
        num_pts = self.num_pts
        t = np.asarray(self.t)
        V = self.V
        r = self.r
        us = self.us
        O = self.O
        v_dots = self.v_dots
        spike_nums = self.spike_nums
        Mv1 = self.Mv1
        Mv2 = self.Mv2
        Mo = self.Mo
        Mr = self.Mr
        Mu = self.Mu
        fd_Mv = self.fd_Mv
        fd_Mr = self.fd_Mr
        fd_Mu = self.fd_Mu
        dt = self.dt

        ##self.U = U
        ##         [N,    N, N, N]
        ##state is [vdot, v, r, u] <==>[ Mv1, Mv2, -I, -I]
        zero = np.zeros((self.N, self.N))
        I = np.eye(self.N)
        v_double_dot = np.hstack((Mv1, zero, -Mr, Mu))
        v_dot = np.hstack((I, zero, zero, zero))
        r_dot = np.hstack((zero, zero, -I, I))
        u_dot = np.hstack((zero, zero, zero, -I))

        state_transition = np.vstack((v_double_dot, v_dot, r_dot, u_dot))
        state_transition = expm(state_transition * dt)

        print('Starting Simulation.')
        fast_sim(dt, vth, num_pts, t, v_dots, V, r, us, O, Mo, state_transition, spike_nums)
        print('Simulation Complete.')

        return self.pack_data()

class OLDComplexSelfCoupledNet(GapJunctionDeneveNet):
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
            thetas = np.linspace(0, 2 * np.pi *(1 - 1 / N), num=N)
            D = np.zeros((d,N), dtype = np.float64)
            D[0,:] = np.cos(thetas)
            D[1,:] = np.sin(thetas)
        
        for i in range(D.shape[1]):
                D[:,i] /= np.linalg.norm(D[:,i])
        return D
    
