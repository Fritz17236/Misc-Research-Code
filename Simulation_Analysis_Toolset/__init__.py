'''
 This is a package of various functions used to define and analyze numerical simulations
 of limit-cycle oscillators. 
 The package defines an abstract base class, DynamicalSystemSim which must be extended
 by a client who provides specific derivative functions and initial conditions. 
 '''

#Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import argrelextrema
from abc import ABC, abstractmethod 
import cmath
from builtins import int
    

class DynamicalSystemSim(ABC):
    '''
     Abstract base class for simulating dynamical systems.
     Client must implement the following methods:
     deriv( self, t, X)
     copy_sim(cls, data
     '''
    
    def __init__(self, X0, T, dt, t0 = 0):
        ''' Initialize dynamical system parameters including 
        X0: the initial system state (must be an iterable object)
        t0: The starting time of the simulation
        T:  The ending time of the simulation
        dt: the spacing between each evaluated data point
        '''
        self.X0 = X0   
        self.t0 = t0
        self.T = T
        self.dt = dt
  
    
    def run_sim(self):
        '''
        Run the numerical integration & return the simulation data structure.
        Generally implemented as scipy RK45 solver.
        Returns a dictionary that contains all parameters assigned to 
        the class instance (such as T, X0, etc)
        '''
        data =  solve_ivp(
            self.deriv,   # Derivative function
            (self.t0, self.T),       # Total time interval
            self.X0,  # Initial State
            t_eval = np.arange(self.t0, self.T, self.dt),  # Returned evaluation time points
            method='LSODA',  #Radau solver for stiff systems
            dense_output=True
            )
        
        sim_data = {}
        
        for param in self.__dict__.keys():
            sim_data[str(param)] = self.__dict__[param]
        
        
        sim_data['X'] = data.y
        sim_data['t'] = data.t
        
        
        return sim_data
                 
    @abstractmethod
    def deriv(self, t, X):
        '''
        Compute the derivative of the system 
        at time t and state X according to desired 
        set of equations defining the dynamical system
        '''
        pass    
        
    @abstractmethod
    def copy_sim(self):
        '''
        Return a simulation with the same parameters
        as the instant
        '''
        pass

    
    
class OscillatorySimAnalyzer(ABC):
    '''
    Base class for analyzing data from DynamicalSystemSim Objects that
    contain stable limit cycles. (Oscillatory behavior)
    '''
    
    
    def perturb_limit_cycle(self, sim, limit_cycle, u, indices):
        '''
        Given sim data, and a set of points of the limit cycle, perturb each limit_cycle[i] by
        u[i] for i in indices. Simulate the trajectory and return a dictionary containing data from each perturbed
        simulation.
        '''
    
        pert_sims = {}
        print("Perturbing Simulation along limit cycle at %i points..."%len(indices))
        
        for i, idx in enumerate(indices):
            print('%i/%i...'%(i+1, len(indices)))
            
            pert_sim = sim.copy_sim()
            pert_sim.X0 = limit_cycle['X'][:, idx] + u[:, idx]
            pert_sims['%i'%idx] = pert_sim.run_sim()
        
        print('Perturbed Simulations Complete...')
        return pert_sims
            
   
    def lc_period(self, limit_cycle):
        '''
        Given a limit cycle, determine the period of limit cycle oscillation.
        '''
        return limit_cycle['t'][-1] - limit_cycle['t'][0]


    def lc_times_to_phases(self, limit_cycle, rads = False):
        ''' 
        Given a limit cycle, return an array of phases
        '''  
        T = self.lc_period(limit_cycle)
        t0 = limit_cycle['t'][0]
        
        phases = []
        
        if rads:
            for t in limit_cycle['t'] - t0:
                phases.append(t/T * 2 * np.pi)
        else:
            for t in limit_cycle['t'] - t0:
                phases.append(t/T)
                
        return np.asarray(phases)
    
    
    def get_traj_dist_from_lc(self, limit_cycle, traj):
        '''
        Given a limit cycle and a trajectory (2,N array), compute the distance
        from the limit cycle for each point on the trajectory and
        return the distances
        '''
        return np.asarray([
            self.dist_to_limit_cycle(limit_cycle, traj[:,i])
             for i in np.arange(traj.shape[1])
             ])
             
    {}  
    ## Abstract methods need implementation 
    def idx_of_lc_convergence(self, limit_cycle, trajectory, delta):
        '''
        Given a limit cycle, a distance delta, and trajectory (Dim, N array).
        Determine the smallest index such that the trajectory is within 
        a distance of delta from the limit cycle. This does not guarantee
        the trajectory is *always* within delta of the limit cycle after the
        returned index.
        '''
    
        def converged(X, limit_cycle, delta):
            ''' Check if a point X is within delta of limit_cycle '''
            
            if np.less_equal(self.dist_to_limit_cycle(limit_cycle, X), delta):

                return True
            
            else:
                return False
    
    
        def bin_search_conv_idx(trajectory, limit_cycle, delta):
            '''
            Recursively implement binary search to find smallest index of convergence
            in given trajectory
            '''
            
            if converged(trajectory[:,0], limit_cycle, delta):
                return 0
            
            
            left = 0
            right = len(trajectory[0,:])-1
            
            
            while right - left > 0:
                if left < 0 or right < 0 :
                    raise Exception("negative index during binary search: left = %i, right = %i"%(left, right))
                
                midpoint = left +  int(
                    np.ceil((right - left) / 2)
                    )
                
                if converged(trajectory[:,midpoint], limit_cycle, delta):
                    if not converged(trajectory[:,midpoint-1], limit_cycle, delta):
                        return midpoint
                    
                    else:
                        right = midpoint  
                    
                else:
                    left = midpoint
                    
                     
            return -1
         

        conv_idx = bin_search_conv_idx(trajectory, limit_cycle, delta)


        if conv_idx is  -1: #if binsearch fails
            print(
            "trajectory starting at ",trajectory[:,0], " does not converge to limit cycle."
            )
            return NaN
        
        else:
            return conv_idx    
       
       
    ## Abstract methods need implementation by subclass    
    @abstractmethod
    def get_limit_cycle(self, sim_data):
        ''' 
        This code assumes existence of a stable limit cycle and that the simulaion
        data provided captures at least one full period of limit cycle oscillation.
        Given a sim data struct, compute & return the limit cycle as a
        dictionary containing limit cycle points and associated times from the 
        original simulation. 
        limit_cycle[X]
        limit_cycle[t]
        '''
        pass
    
    @abstractmethod
    def lc_eigen_decomp(self):
        ''' 
        Given simulation data, return the eigendecomposition of the 
        linearization at each point around the limit cycle.
        Returns eigendecomposition as dictionary with the following entries:
        eig["lambda_i"] ith eigenvalues for points along the limit cycle
        eig["evec_i"] eigenvector associated with l_i each point along limit cycle
        '''
        pass


    @abstractmethod
    def dist_to_limit_cycle(self, limit_cycle, X):
        '''
        Compute the distance from the point X to the limit cycle
        '''
        pass
    
    @abstractmethod
    def nearest_lc_point(self, limit_cycle, X):
        '''
        Find the closest point on a given limit cycle to the point X.
        Return the index of the point and the associated distance
        '''
        pass
        


class PlanarLimitCycleAnalyzer(OscillatorySimAnalyzer):
    ''' 
    Analysis for Planar (2-dimensional) limit cycles.
    Implements OscillatorySimAnalyzer abstract class.
    '''
    
    def get_limit_cycle(self, sim_data):
        ''' 
        This code assumes existence of a stable limit cycle and that the simulaion
        data provided captures at least one full period of limit cycle oscillation.
        Given a sim data struct, compute & return the limit cycle as a
        dictionary containing limit cycle points and associated times from the 
        original simulation. 
        limit_cycle[X]
        limit_cycle[t]
        '''
        xs = sim_data['X'][0,:]
        ys = sim_data['X'][1,:]
        
        # search for the maximum values in the data
        extrema = argrelextrema(xs, np.greater)[0]
        
        if len(extrema) < 2:
            raise Exception("Only 1 maximum detected in trajectory - run simulation longer?")
        
        # this gives us a period of oscillation, return the state for times in this period
        idxs = np.arange(extrema[-2],extrema[-1] + 1)
        
        limit_cycle = {}
        
        limit_cycle['X'] = sim_data['X'][:,idxs]
        limit_cycle['t'] = sim_data['t'][idxs]
        
        return limit_cycle
    
    
    def nearest_lc_point(self, limit_cycle, X):
        '''
        Given a point X and a limit cycle, return
        the index and distance of the point on the limit cycle
        nearest (by euclidian distance) to X.
        '''
        N = len(limit_cycle['t'])
        xs = np.ones((N,)) * X[0]
        ys = np.ones((N,)) * X[1]
        
        dxs = limit_cycle['X'][0,:] - xs
        dys = limit_cycle['X'][1,:] - ys
        
        dists = np.sqrt(np.square(dxs) + np.square(dys))
        mindex = np.argmin(dists)
        return dists[mindex], mindex
    
    
    def dist_to_limit_cycle(self, limit_cycle, X):
        ''' 
        Compute the Euclidean Distance from X to closest point
        on the limit cycle.
        '''
        
        min_dist, _ = self.nearest_lc_point(limit_cycle, X)
    
        return min_dist

    
             
        


                
    
    
