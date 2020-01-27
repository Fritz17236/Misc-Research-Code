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

#Abstract class for simulating dynamical systems. Uses scipy.solve_ivp RK45 by default.
class DynamicalSystemSim():
    '''
     Abstract base class for simulating dynamical systems.
     Client must implement the following methods:
     deriv( self, t, X)
     copy_sim(cls, data
     '''
    

    def copy_sim(self):
        '''
        Return a simulation with the same parameters
        as the instant
        '''
        pass
   
     
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
          

    def deriv(self, t, X):
        '''
        Compute the derivative of the system 
        at time t and state X according to desired 
        set of equations defining the dynamical system
        '''
        pass
  
    
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
            t_eval = np.arange(self.t0, self.T, self.dt)  # Returned evaluation time points
            )
        
        sim_data = {}
        
        for param in self.__dict__.keys():
            sim_data[str(param)] = self.__dict__[param]
        
        
        sim_data['X'] = data.y
        sim_data['t'] = data.t
        
        
        return sim_data
    
class OscillatorySimAnalyzer():
    '''
    Base class for analyzing data from DynamicalSystemSim Objects that
    contain stable limit cycles. (Oscillatory behavior)
    '''
    
    ## Present methods
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
            pert_sims['pert_idx=%i'%idx] = pert_sim.run_sim()
        
        print('Perturbed Simulations Complete...')
        return pert_sims
            
    ## Abstract methods need implementation to use class
   
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
                
        return phases

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


 
        
    






## Run Simulation & set parameters here

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
        
        
       

    
# simulation parameters

    @abstractmethod
    def get_traj_dist_from_lc(self, limit_cycle, trajectory):
        '''
        Given a limit cycle and a trajectory, compute the distance
        from the limit cycle for each point on the trajectory and
        return the distances
        '''
        pass
        
    @abstractmethod  
    def time_to_convergence(self, limit_cycle, trajectory, delta):
        '''
        Given a limit cycle, a distance delta, and trajectory.
        Determine the smallest elapsed time such that the trajectory is within 
        a distance of delta from the limit cycle
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
        
        # this gives us a period of oscillation, return the state for times in this period
        idxs = np.arange(extrema[-2],extrema[-1] + 1)
        
        limit_cycle = {}
        
        limit_cycle['X'] = sim_data['X'][:,idxs]
        limit_cycle['t'] = sim_data['t'][idxs]
        
        return limit_cycle
    
    def nearest_lc_pt(self, limit_cycle, X):
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
        
    
    
    @abstractmethod
    def lc_eigen_decomp(self):
        pass
          
    





def evec_ang(lc_inst_x, lc_inst_y, evec_x, evec_y):
    '''
    given a vector parrallel to the limit cycle, compute the angle
    between that vector and the provided eigenvector evec
    '''
    lc = np.asarray([lc_inst_x, lc_inst_y])
    evec = np.asarray([evec_x, evec_y])
    return np.arccos(lc.T@evec)

    
    
             
        


                
    
    
