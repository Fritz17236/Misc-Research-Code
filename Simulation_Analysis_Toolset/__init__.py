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
from scipy.signal import find_peaks
from abc import ABC, abstractmethod 
import cmath
    

# check get latent phase
# perturbation at 0 should give phase 0 


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
  
    
    def run_sim(self, event = None):
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
            dense_output=True,
            events = event
            )
        
        sim_data = {}
        
        for param in self.__dict__.keys():
            sim_data[str(param)] = self.__dict__[param]
        
        
        sim_data['X'] = data.y
        sim_data['t'] = data.t
        sim_data['t_events'] = data.t_events
        
        return sim_data
           
                 
    def same_sim_at_point(self, X):
        ''' Copy a simulation sim, except begin at a different state X '''
        sim_copy = self.copy_sim()
        sim_copy.X0 = X
        return sim_copy
    
    @abstractmethod
    def deriv(self, t, X):
        '''
        Compute the derivative of the system 
        at time t and state X according to desired 
        set of equations defining the dynamical system
        '''
        pass    
        
    def copy_sim(self):
        '''
        Return a simulation with the same parameters
        as the instant
        '''
        new_sim = DynamicalSystemSim()
        for d in self.__dict__.keys():
            new_sim.d = self.__dict__[str(d)]
  
 
            
class LinearDynamicalSystem(DynamicalSystemSim):
    '''
    A LinearDynamicalSystem (LDS) object instantiates a specified dynamical 
    system. The user provides a LDS specification via initial arguments. The system begins at 
    time = 0, and is advanced  by calling the update() method. Any field listed below can be 
    directly accessed via dot notation.  
    
    ---------------- Initial Arguments: ----------------------------------------------------------------------------------------
    init_state: (M numpy array) - an initial configuration of the dynamical system
    
    A: state transition matrix (M x M numpy array) - determines how one state maps to the next 
    
    init_input: (L numpy array) - an initial configuration of the input to the dynamical system 
    
    B: input matrix ( M x L numpy array ) - maps the dependence of the state on the input vector 
    
    dt: time step (miliseconds) - the time elapsed in the dynamical system between successive updates - defaults to 0.001 ms
        

    --------------- Fields -------------------------------------------------------------------------------------------------------
    t: (T scalar) the elapsed time of the simulation
    
    A: state_transition matrix (see above)
    
    B: input matrix (see above)
    
    x: (M numpy array ) state vector at current time
    
    u: (function handle) input u at time t i.e. def u(t) returns vector 

    
    '''
    
    def __init__(self, init_state, A, init_input, B, u = None, dt = .001, T = 10):
        '''
        Initialize the Linear Dynamical System. If there are any defaults, ensure they have appropriate
        dimensions. 
        '''
        X0 = init_state  
        assert(X0.ndim == 1), "Initial State is expected to be a vector, but has %i dimensions" %self.x.ndim
        assert(X0.shape[0] == A.shape[0]), ("State vector size (%i) does not match"\
                                                " Number of State Transition Matrix Rows (%i)" %(self.x.shape[0], A.shape[0]) )
        self.A = A
        self.u0 = init_input
        if u is None:
            self.u = lambda t: np.zeros(self.u0.shape) 
        else:
            self.u = u
        assert(self.u0.ndim == 1), "Initial Input is expected to be a vector, but has %i dimensions" %self.u.ndim
        self._null_input = np.zeros(self.u0.shape)

    
        assert(B.shape[0] == A.shape[0]), "Input matrix rows (%i) should match state transition matrix rows (%i)" %(B.shape[0], A.shape[0])
        assert(B.shape[1] == len(self.u0)), "Input matrix columns (%i) should match length of provided input vector (%i)" %(B.shape[1], self.u.shape[0])
        self.B = B
        
        
        super().__init__(X0, T, dt)

    def deriv(self, t, x):
        '''
        Return the derivative of the state (x-dot) at the current time
        '''
        try:
            return self.A @ x + self.B @ self.u(t)
        except TypeError:
            return self.A@x
    
    
    
    
class OscillatorySimAnalyzer(ABC):
    '''
    Base class for analyzing data from DynamicalSystemSim Objects that
    contain stable limit cycles. (Oscillatory behavior)
    '''
    
    def __init__(self, data):
        ''' Initiate with a data object'''
        self.data = data
    
    
    def perturb_limit_cycle(self, sim, limit_cycle, u, indices):
        '''
        Given sim data, and a set of points of the limit cycle, perturb each limit_cycle[i] by
        u[i] for i in indices. Simulate the trajectory and return a dictionary containing data from each perturbed
        simulation.
        '''

        pert_datas = {}
        print("Perturbing Simulation along limit cycle at %i points..."%len(indices))
        
        for i, idx in enumerate(indices):
            
            print('%i/%i...'%(i+1, len(indices)))
            pert_X0 = limit_cycle['X'][:, idx] + u[:, idx]
            pert_sim = sim.same_sim_at_point(pert_X0)
            pert_datas['%i'%idx] = pert_sim.run_sim() 
            
        print('Perturbed Simulations Complete...')
        return pert_datas
            
   
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

 
    def trajectory_difference(self, X, Y, errfunc):
        '''
        Given two (Dim x N trajectories), X & Y, return their 
        difference according to a predefined error function. 
        '''
        return np.asarray([ errfunc(X[:,i], Y[:,i])
                              for i in np.arange(len(X[0,:]))
                               ])             


    def traj_idx_of_lc_convergence(self, limit_cycle, trajectory, delta):
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


        if conv_idx is  -1: #if binsearch fails try linear
            for i in np.arange(len(trajectory[0,:])):
                if converged(trajectory[:,i], limit_cycle, delta):
                    return conv_idx
            print(
            "trajectory starting at ",trajectory[:,0], " does not converge to limit cycle."
            )
            return -1
        
        else:
            return conv_idx    
       
       
    def concat_data_traj(self, data0, data1, get_clip_idx = False):
        ''' 
        Given two data objects, concatenate their state trajectories such that
        data0['X'] = data0['X']:data1['X'],
        data0['t'] = data0['t']:data1['t']
        '''
        clip_idx = len(data0['X'][0,:])
        data0['X'] = np.append(data0['X'], data1['X'], axis=1)
        data0['t'] = np.append(data0['t'], data1['t'] )
        
        if get_clip_idx:
            return data0, clip_idx
        else:
            return data0
    
    
    def plot(self, data, mode = 'plot', fig_name = None, **kwargs):
        '''
        Plot the 2-D data given 
        a 2 x N vector. Additional arguments are passed to the plot function.
        mode specifies the type of plot e.g. plt.[mode] is called so is plot
        default but may also be scatter, etc. 
        '''
        
        data = np.asarray(data)
        
        xs = data[0,:]
        ys = data[1,:]
        
        plt.figure(fig_name)
        

    
        if 'xlabel' in kwargs.keys():
            plt.xlabel(kwargs['xlabel'])
            kwargs.pop('xlabel')
            
        if 'ylabel' in kwargs.keys():
            plt.ylabel(kwargs['ylabel'])
            kwargs.pop('ylabel')    
        
        if 'title' in kwargs.keys():
            plt.title(kwargs['title'])
            kwargs.pop('title')
        
        if mode == 'plot':
            plt.plot(xs, ys, **kwargs)
            
        
        elif mode == 'scatter':
            plt.scatter(xs, ys, **kwargs)
        
        if 'label' in kwargs.keys():
            plt.legend()
                
        if 'c' in kwargs.keys() or 'color' in kwargs.keys():
            plt.colorbar()
            
    
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
    
    @abstractmethod
    def get_latent_phase(self, X, rads = False):
        '''
        Given a point X that is assumed to converge to a limit cycle,
        compute the latent phase of the point X. That is, compute the
        initial phase along the limit cycle that will intersect the trajectory 
        at the time and point of convergence.
        '''
        pass

    @abstractmethod
    def get_latent_error_trajectory(self, sim, limit_cycle, X, errfunc):
        '''
        Given a simulation and a point X, compute the latent phase of X
        and return the sim, its trajectory, and the latent trajectory of X
        as well as their difference according to the supplied errfunc
        '''
    


class PlanarLimitCycleAnalyzer(OscillatorySimAnalyzer):
    ''' 
    Analysis for Planar (2-dimensional) limit cycles.
    Implements OscillatorySimAnalyzer abstract class.
    Also features plotting functions
    '''
    
    def __init__(self, data):
        super().__init__(data)
        try:
            self.limit_cycle = self.get_limit_cycle(self.data)
        except:
            print("Warning: limit cycle not successfully computed")
    
    
    def get_limit_cycle(self, sim_data):
        ''' 
        going along first axis  of state ...
        This code assumes existence of a stable limit cycle and that the simulaion
        data provided captures at least one full period of limit cycle oscillation.
        Given a sim data struct, compute & return the limit cycle as a
        dictionary containing limit cycle points and associated times from the 
        original simulation. 
        limit_cycle[X]
        limit_cycle[t]
        '''
        xs = sim_data['X'][0,:]
        
        # search for the maximum values in the data
        extrema, _ = find_peaks(xs)
        
        if len(extrema) < 2:
            print ("Warning: Only 1 maximum detected in trajectory. No limit cycle returned. - run simulation longer?")
            return None
        
        # this gives us a period of oscillation, return the state for times in this period
        idxs = np.arange(extrema[-2],extrema[-1])

        
        limit_cycle = {}
        
        limit_cycle['X'] = sim_data['X'][:,idxs]
        limit_cycle['t'] = sim_data['t'][idxs]
        
        #close the limit cycle, making last data point equal to first data point and time + dt
        last_pt = limit_cycle['X'][:,-1]
        first_pt = limit_cycle['X'][:,0]
        
        
        lin_interp = np.expand_dims((last_pt + first_pt)/2,axis = 1)
        
        
        limit_cycle['X'] = np.append(limit_cycle['X'], lin_interp, axis=1)
        limit_cycle['t'] = np.append(limit_cycle['t'], limit_cycle['t'][-1] + sim_data['dt']/2)
        
        limit_cycle['X'] = np.append(limit_cycle['X'], np.expand_dims(last_pt,axis=1), axis=1)
        limit_cycle['t'] = np.append(limit_cycle['t'], limit_cycle['t'][-1] + sim_data['dt'])
        return limit_cycle
    
    
    def nearest_lc_point(self, limit_cycle, X):
        '''
        Given a point X and a limit cycle, return
        the index and distance of the point on the limit cycle
        nearest (by euclidian distance) to X.
        '''        
        dxs = np.square(limit_cycle['X'][0,:] - X[0])
        dys = np.square(limit_cycle['X'][1,:] - X[1])
        
        dists = np.sqrt(dxs +  dys)
        mindex = np.argmin(dists)
        return dists[mindex], limit_cycle['X'][:,mindex], mindex
    
    
    def dist_to_limit_cycle(self, limit_cycle, X):
        ''' 
        Compute the Euclidean Distance from X to closest point
        on the limit cycle.
        '''
        
        min_dist, _, _ = self.nearest_lc_point(limit_cycle, X)
    
        return min_dist


    def plot_phase_space(self):
        ''' Plot the trajectory of the associated data in state space'''
        print('Plotting phase space... ')

        self.plot(
            self.data['X'],
            fig_name = 'trajectory',
            xlabel = 'x',
            ylabel = 'y',
            label = 'Simulated')    
        
        self.plot(
            [self.data['t'], self.data['X'][0,:]],
            'plot','x_trace',
            label = 'Simulated',
            xlabel='t',
            ylabel='x',
            title='X-axis Trace')

        self.plot(
            [self.data['t'], self.data['X'][1,:]],
            'plot',
            'y_trace',
            label = 'Simulated',
            xlabel='t',
            ylabel='y',
            title='Y-axis Trace')
     
        
    def plot_limit_cycle(self):
        ''' Plot the limit cycle as a separate phase plot '''
        print('Plotting limit cycle ... ')
    
        self.plot(
            self.limit_cycle['X'],
            'plot',
            'limit cycle',
            xlabel='x',
            ylabel='y',
            label='Simulated',
            title='Limit Cycle Phase Portrait')
        
        
    
    
    
    
    
    
    
    
    
    
    