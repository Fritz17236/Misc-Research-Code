# Numerical Simulations for Exploring Van der Pol Oscillator

# Chris Fritz 1/15/2020

#TODO: 
# make sure latent pahse is correct: simulate trajectories and make sure they match at ttc
# compute latent phase, replace a trajectory with its latent phase, then simulate the original trajectory to determine approx error

##import statements
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import argrelextrema
import cmath
from numpy import iscomplex

## Class Definitions
class VanderPolSim:
    '''Simulation object - feed parameters and returns simulation datat structure'''
    def __init__(self, T = 50, dt = .01, mu = 3, x0 = .5, y0 = .5):
        self.T = T     # total sim length
        self.dt = dt # time step
        self.mu = mu   # damping coefficient 
        self.x0 = x0   # initial x state
        self.y0 = y0   # initial y state

    def deriv(self, t, X):
        '''
        Derivative function for use in integrator. X is 2-vector of state
        van der pol eqs:
        dx/dt = y
        dy/dt = mu * (1-x**2) * y  - x
         '''
        x = X[0]
        y = X[1]
        mu = self.mu
        
        def y_dot(x, y, mu):
            return mu * (1-x*x) * y - x
    
        def x_dot(y):
            return y
    
        num = np.asarray([x_dot(y), y_dot(x, y, mu)])
        return num
    
    def run_sim(self):
        '''
        Run the numerical integration & return the simulation data structure.
        [data].y is the state, [data].t is the time vector '''
        data =  solve_ivp(
            self.deriv,   # Derivative function
            (0, self.T),       # Total time interval
            np.asarray([self.x0, self.y0]),  # Initial State
            t_eval = np.arange(0, self.T, self.dt)  # Returned evaluation time points
        )
        data.mu = self.mu
        data.T = self.T
        data.x0 = self.x0
        data.y0 = self.y0
        data.dt = self.dt
        return data


## Data Analysis & Helper Functions

def get_limit_cycle(data, indices = False):
    ''' 
    Given a sim data struct, compute & return the limit cycle as a
    (3, N) array where N is the number of data points in 1 period.
    (2 state variables & time variable data)
    This code assumes the 2nd half of data is exclusive limit cycle data
    i.e. the system is in steady state oscillation with no transients.
    Indices are optionally returned if set to true, additional tuple val
    '''
    # take 2nd half of data
    xs = data.y[0,:]
    ys = data.y[1,:]
    ts = data.t
    
    
    # search for the next maximum within the data
    extrema = argrelextrema(xs, np.greater)[0]
    
    # this gives us a period of oscillation, return the state for times in this period
    idxs = np.arange(extrema[-2],extrema[-1] + 1)
    
    if (indices):
        return (xs[idxs], ys[idxs], ts[idxs], idxs)
    else:
        return (xs[idxs], ys[idxs], ts[idxs])
    
def l_pos(x, y, mu):
    ''' eigenvalue 0 of linear approximation'''
    num = mu * (1 - x**2) + cmath.sqrt(mu**2 * (1 - x**2)**2 + 4 * (2*mu*x*y - 1) )
    return np.real(num / 2)

def l_neg(x, y, mu):
    ''' eigenvalue 0 of linear approximation'''
    num = mu * (1 - x**2) - cmath.sqrt(mu**2 * (1 - x**2)**2 + 4 * (2*mu*x*y - 1) )
    return np.real(num / 2)

def evec_pos(x, y, mu):
    ''' get the eigenvector associated with positive eigval '''
    vec = np.asarray([1, l_pos(x, y, mu)])
    return vec / np.linalg.norm(vec)

def evec_neg(x, y, mu):
    ''' get the other eigenvector '''
    vec = np.asarray([
        l_neg(x, y , mu) - mu * (1 - x**2),
        2 * mu * x * y - 1
        ])
    return vec / np.linalg.norm(vec)

def evec_ang(lc_inst_x, lc_inst_y, evec_x, evec_y):
    '''
    given a vector parrallel to the limit cycle, compute the angle
    between that vector and the provided eigenvector evec
    '''
    lc = np.asarray([lc_inst_x, lc_inst_y])
    evec = np.asarray([evec_x, evec_y])
    return np.arccos(lc.T@evec)

def eigen_decomp(data, limit_cycle = True):
    ''' 
    Given simulation data, return the eigendecomposition of the 
    linearization at each point around the limit cycle.
    Returns data as dictionary with the following entries:
    eig["l_neg"] negative root eigenvalues at each of the N data points
    eig["evec_neg"] eigenvector associated with l_neg
    eig["angs_neg"] is the angle of the eigenvector relative to the limit cycle trajectory
    If limit_cycle is true, only return the eigenvalue along a computed limit
    cycle as opposed to the entire trajectory of data
    '''
    #loop through data
    if limit_cycle:
        xs, ys, _ = get_limit_cycle(data)
    else:
        xs = data.y[0,:]
        ys = data.y[1,:]
    mu = data.mu
    N = len(xs)
    
    l_negs = np.zeros((N,1))
    l_poss = np.zeros((N,1))
    evec_negs = np.zeros((2,N))
    evec_poss = np.zeros((2,N))
    evec_pos_angs = np.zeros(l_negs.shape)
    evec_neg_angs = np.zeros(l_negs.shape)
    for i in np.arange(N):
        l_negs[i] = l_neg(xs[i], ys[i], mu)
        l_poss[i] = l_pos(xs[i], ys[i], mu)
        
        evec_negs[:,i] = evec_neg(xs[i], ys[i], mu)
        evec_poss[:,i] = evec_pos(xs[i], ys[i], mu)
        
        lc_inst_x = lc_x[i]-lc_x[i-3]
        lc_inst_y = lc_y[i] - lc_y[i-3]
        
        evec_pos_angs[i] = evec_ang(
            lc_inst_x, lc_inst_y,
            evec_poss[0,i], evec_poss[0,i]
        )
        
        evec_neg_angs[i] = evec_ang(
            lc_inst_x, lc_inst_y,
            evec_negs[0,i], evec_negs[0,i])
        
    
    eig_dec = {
        "l_neg" : l_negs,
        "l_pos" : l_poss,
        "evec_neg" : evec_negs,
        "evec_pos" : evec_poss,
        "angs_neg" : evec_neg_angs,
        "angs_pos" : evec_pos_angs
        }  
    return eig_dec      

def perturb_limit_cycle(data, lc_x, lc_y, u, indices = [-1],  noise = False, noise_str = 1):
    '''
    Given data 
    and a perturbation vector u, compute the limit cycle, apply perturbation u to indices points
    on the limit cycle, and simulate the trajectory of each point.
    Returns a 2 x T/dt x N matrix of (x,y) points up to time T for perturbed trajectory n
    '''
    # for each point in the limit cycle specified by indices,
    N = len(lc_x)
    mu = data.mu
    T = data.T # adjust length to determine speed of simulation (number of total points) 
    dt = data.dt
    
    if noise:
        noise_vec = noise_str * np.random.normal(size = (N,1))
        lc_y += noise_vec
    
    

    # perturb point by u        
    pert_trajs = np.zeros((2, int(T/dt), len(indices)))
    
    for lc_idx, i in enumerate(indices):
        data = VanderPolSim(
            mu = mu,
            T = T,
            dt = dt,
            x0 = lc_x[i] + u[0],
            y0 = lc_y[i] + u[1]
             ).run_sim()
             
        pert_trajs[:,:,lc_idx] = data.y
        
    return pert_trajs






## Run Simulation & set parameters here

def nearest_lc_point(lc_x, lc_y, x, y):
    '''
    Find the closest point on a given limit cycle to the point (x,y).
    '''
    xs = np.ones(lc_y.shape) * x
    ys = np.ones(lc_x.shape) * y
    
    dxs = lc_x - xs
    dys = lc_y - ys
    
    dists = np.sqrt(np.square(dxs) + np.square(dys))
    
    mindex = np.argmin(dists)
    return (lc_x[mindex], lc_y[mindex], mindex)
    
def dist_to_limit_cycle(lc_x, lc_y, x, y):
    '''
    Compute the distance of (x,y) to the limit cycle (lc_x, lc_y),
    defined here as the minimum distamce of (x,y) to any point on the 
    limit cycle defined by (lc_x,lc_y)
    '''
    
    close_lc_pt = nearest_lc_point(lc_x, lc_y, x, y)
    
    return ((close_lc_pt[0]-x)**2 + (close_lc_pt[1] - y)**2)**.5
    
    
# simulation parameters

def get_traj_dist_from_lc(lc_x, lc_y, traj_x, traj_y):
    '''
    Given a limit cycle (lc_x, lc_y) and a trajectory of points (traj_x, traj_y)
    compute the distance to the limit cycle for each point (traj_x, traj_y) and 
    return this distance as a vector
    '''
    dists = np.zeros((len(traj_x,)))
    
    for i,_ in enumerate(dists):
        dists[i] = dist_to_limit_cycle(lc_x, lc_y, traj_x[i], traj_y[i])
        
    return dists

def time_to_convergence(lc_x, lc_y, traj_x, traj_y, traj_ts, delta):
    
    '''
    Given a limit cycle (lc_x, lc_y) , a distance delta,
    and trajectory in question of points (x,y,t), determine the elapsed time before
    the trajectory is within delta distance of
    the limit cycle
    '''
    def bin_search_conv_idx(idxs):
        '''
        Recursively implement binary search to find smallest index of convergence
        in given trajectory
        '''
        
        if len(idxs) == 0:
            return -1
        else:
            midpoint = int(len(idxs)/2)
            search_idx = idxs[midpoint]
            curr_conv = converged(lc_x, lc_y, traj_x[search_idx], traj_y[search_idx], delta)
            prev_conv = converged(lc_x, lc_y, traj_x[search_idx-1], traj_y[search_idx-1], delta)
            
        if curr_conv: # if current converged
                
            if not prev_conv: # if previous not converged stop
                return search_idx
            
            else: # otherwise check left half
                return bin_search_conv_idx(idxs[0:midpoint])
                    
        
        else: #check right half
            return bin_search_conv_idx(idxs[midpoint:])
            
            
        
    # find first point of convergence
    
    conv_idx = bin_search_conv_idx(np.arange(len(traj_x)))


    if conv_idx is  -1: # if binseach fails try linear search (old code)
        for i in np.arange(len(traj_x)):
            if converged(lc_x, lc_y, traj_x[i], traj_y[i], delta):
                return traj_ts[i]
        print(
        "location ",
        (traj_x[0],traj_x[1]),
        "does not converge to limit cycle within %i time units." % traj_ts[-1]
        )
        return NaN
    
    else:
        return traj_ts[conv_idx]

def converged(lc_x, lc_y, x, y, delta):
    '''
    Given a limit cycle specified by lc_x, lc_y, a point x,y and some distance delta,
    check if the distance of the closest point on the limit cycle to (x,y) is within delta.
    '''
    if dist_to_limit_cycle(lc_x, lc_y, x, y) <= delta:
        return True
    else:
        return False    

def lc_period(lc_t):
    ''' 
    Given a time vector of limit cycle points,
    return the period of the limit cycle
    '''
    return lc_t[-1] - lc_t[0]

def lc_time_to_phase(lc_t, rads = False):
    ''' Given a limit cycle time, convert to phase '''
    T = lc_period(lc_t)
    
    phases = []
    if rads:
        for t in lc_t - lc_t[0]:
            phases.append(t/T * 2 * np.pi)
    else:
        for t in lc_t - lc_t[0]:
            phases.append(t/T)
    return phases

def lc_to_phase(lc_x, lc_y, lc_t, rads = False):
    '''
    Given a limit cycle (lc_x, lc_y, lc_t)
    return its phase representation, i.e map
    each point in the limit cycle (lc_x, lc_y) to the unit
    circle and return the phase of the point, which is a fraction
    of the period.
    If rads is true then return answer in radians
    '''
    # each point (lc_x, lc_y) is a value associated with a key phase
    # given by the fraction of the period passed
    phase_lc = {}
    if rads:
        phases = lc_time_to_phase(lc_t, rads = True)
    else: 
        phases = lc_time_to_phase(lc_t, rads = False)
    

    for i in np.arange((len(lc_x))):
        phase_lc.update({phases[i] : (lc_x[i], lc_y[i]) })
    for i in np.arange((len(lc_x))):
        phase_lc.update({phases[i] : (lc_x[i], lc_y[i]) })

    return phase_lc


        
################Simulation Parameters        

def get_latent_phase(data, x, y, rads = False):
    ''' 
    Given a point (x,y) in the phase plane,  compute its latent phase.
    The latent phase is defined as the phase of the limit cycle corresponding 
    to the intersection of the trajectory starting at (x,y) with the limit cycle
    at t--> infinity. 
    Caller must provide data from VanderpolSim. Also returns time to convergence.
    '''

    
    ## make sure lc at ttc is same as latent phase after ttc
    
    # start simulation at initial condition & run simulation for T
    sim = VanderPolSim(mu = data.mu, dt = data.dt, T = data.T, x0 = x, y0 = y)
    sim_data = sim.run_sim()
    traj_x = sim_data.y[0,:]
    traj_y = sim_data.y[1,:]
    traj_ts = sim_data.t
    
    lc_x, lc_y, lc_t = get_limit_cycle(data)
    
    # get time to convergence, 
    ttc = time_to_convergence(lc_x, lc_y, traj_x, traj_y, traj_ts, delta)
    conv_idx = np.argmax(traj_ts == ttc)
    
    #get phase of convergence
    # trajectory state at conv_idx
    nr_lc_x, nr_lc_y, _ = nearest_lc_point(lc_x, lc_y, traj_x[conv_idx], traj_y[conv_idx])
 
    #phase associated with the trajectories state at ttc
    phases = lc_to_phase(lc_x, lc_y, lc_t)
    phase_list = [*phases.keys()]
    # find phase associated with nr_lc_pt (search dict by value)
    conv_phase = -1
    for ph in phase_list:
        if phases[ph] == (nr_lc_x, nr_lc_y):
            conv_phase = ph
            
    if conv_phase == -1:
        print("Could not compute convergence phase")
        return NaN
    
    #wt + phi_lat = phase of convergence
    if rads:
        phi_lat = conv_phase - (2 * np.pi * ttc / lc_period(lc_t))
        
    else:
        phi_lat = conv_phase - (ttc / lc_period(lc_t))

        
    
    closest_idx = np.argmin(np.abs(phase_list- phi_lat))
    

    return phase_list[closest_idx], ttc 

def latent_error_traj(data, x, y):
    ''' 
    Given a simulation and a point (x,y), compute the latent phase,
    then measure the deviation of the latent phase trajectory from 
    the assumed limit cycle
    '''
    # compute latent phase
    lp, _ = get_latent_phase(data, x, y)
    lc_x, lc_y, lc_t = get_limit_cycle(data)
    
    # run a simulation starting at latent phase along limit cycle, 
    phases = lc_to_phase(lc_x, lc_y, lc_t)
    
    
    lat_start = phases[lp]
    
    lat_data = VanderPolSim(mu = data.mu, T = data.T, dt = data.dt,
                           x0 = lat_start[0], y0 = lat_start[1]).run_sim()
    
    # run another starting at true trajectory
    true_data = VanderPolSim(mu = data.mu, T = data.T, dt = data.dt,
                           x0 = x, y0 = y).run_sim()
    
    
    # return latent & true trajectories
    return (lat_data, true_data)


mu = 3
T = 50
dt = .001
sim = VanderPolSim(mu = mu, T = T, dt = dt)
data = sim.run_sim()
ts = data.t
xs = data.y[0,:]
ys = data.y[1,:]

    
epsilon = 1 # perturbation strength
u =  epsilon * np.asarray([1, 0])
delta = .05  # threshold for convergence (converged if dist <= delta)
################



################### Data Analysis & Plotting Configuration 
plot_trajectory = False    # Plot the (x,y) and (t,x), (t,y) trajectories of the simulation including nullclines
plot_limit_cycle = False  # Assuming at least 2 full periods of oscillation, compute & plot the limit cycle trajectory
plot_eigen_decomp = False# Compute the eigenvalues/eigenvectors along the limit cycle & display them
plot_perturbation_analysis = False # perturb along an eigenvector & compute its linearized growth for each point on the limit cycle
plot_traj_perturbations = False # numerically simulate a given perturbation along given points of a limit cycle
plot_pert_along_evecs = False # Perturb the limit cycle along an eigenvector and plot the convergence results
plot_convergence_analysis = False# simulate given perturbation for chosen indices & compute their distance to limit cycle vs phase
ttc_vs_phase_vs_mu = False# similar to plot convergence analysis, except do so for various values of mu


print("Running Simulation")

if (plot_trajectory):    
    print('Plotting trajectory ... ')
    plt.figure()
    plt.plot(xs, ys, label='Oscillator Trajectory')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Phase Space mu = %.2f" %mu)
     
    #plot nullclines
    x_nullcline = np.linspace(-2*np.max(np.abs(xs)), 2*np.max(np.abs(xs)))
    y_nullcline = [  x / (mu * (1 - x**2))  for x in x_nullcline]
    plt.plot(x_nullcline, y_nullcline, '--',color='black', label = 'Y nullcline') 
    plt.axhline(y = 0, label= 'X-nullcline', color='green')
    plt.legend()
    #
    
    
    plt.figure()
    plt.plot(ts,xs)
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("x (voltage) trace")
    
    plt.figure()
    plt.plot(ts,ys)
    plt.xlabel("t")
    plt.ylabel("y")
    plt.title("y trace")
    
if (plot_limit_cycle):
    print('Plotting limit cycle ... ')

    lc_x, lc_y, lc_t, idxs = get_limit_cycle(data, indices = True) # get exactly 1 limit cycle
    plt.figure()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(lc_x, lc_y,label='mu = %.1f' % mu)
    plt.legend()
    plt.title("Limit Cycle Phase Portrait")
    
if (plot_eigen_decomp):
    print('Plotting eigendecomposition...')
    
    if not (plot_limit_cycle):
        lc_x, lc_y, lc_t = get_limit_cycle(data)

    ed = eigen_decomp(data)
     
    plt.figure()
    plt.plot((lc_t-lc_t[0]) / lc_period(lc_t), ed["l_neg"],label= '$\lambda_{-}$')
    plt.plot((lc_t-lc_t[0]) / lc_period(lc_t), ed["l_pos"],label= '$\lambda_{+}$')
    plt.xlabel('fraction of period time')
    plt.ylabel('$Re (\lambda) $')
    plt.legend()
    
    plt.figure()
    
    plt.quiver(lc_x, lc_y, ed["evec_neg"][0,:], ed["evec_neg"][1,:], ed["l_neg"], scale = 8, headwidth = 6)
    plt.title("Linear Stability (Negative Root Eigenvalue")
    plt.plot(lc_x, lc_y,label='mu = %.1f' % mu, alpha = .4,c = 'red')
    plt.colorbar()
    
    plt.figure()
    plt.quiver(lc_x, lc_y, ed["evec_pos"][0], ed["evec_pos"][1], ed["l_pos"], scale = 8, headwidth = 6)
    plt.plot(lc_x, lc_y,label='mu = %.1f' % mu, alpha = .4,c = 'red')
    plt.title("Linear Stability (Positive Root Eigenvalue")
    plt.colorbar()
    
if (plot_perturbation_analysis):
    print('Plotting perturbation analysis...')
    
    # Project a horizontal perturbation x onto the eigvecs scaled by eigvals & quiver plot 
    pert_x = delta * np.asarray([1, 0])
    pert_y = delta * np.asarray([0, 1])
    
    net_x = np.zeros((2, len(idxs)))
    net_y = np.zeros((2, len(idxs)))
    
    for i, idx in enumerate(idxs):
        net_x[:,i] = (
            np.exp(ed["l_pos"][idx]) * pert_x@ed["evec_pos"][:,idx] # project onto E-vecs & scale
         + np.exp(ed["l_neg"][idx]) * pert_x@ed["evec_neg"][:,idx]
         )
         
        net_y[:,i] = (
            np.exp(ed["l_pos"][idx]) * pert_y@ed["evec_pos"][:,idx] # project onto E-vecs & scale
         + np.exp(ed["l_neg"][idx]) * pert_y@ed["evec_neg"][:,idx]
         ) 
    
    plt.figure()
    plt.quiver(xs, ys, net_x[0,:], net_x[1,:], np.log(np.linalg.norm(net_x,axis=0)/epsilon), scale = 1, headwidth = 6)
    plt.plot(lc_x, lc_y,label='mu = %.1f' % mu, alpha = .4,c = 'red')
    plt.title("X - Perturbation Net Direction & Strength")
    cbar = plt.colorbar()
    cbar.set_label("$log(||A\delta|| / ||\delta||)$")
    
    
    plt.figure()
    plt.quiver(xs, ys, net_y[0,:], net_y[1,:], np.log(np.linalg.norm(net_y,axis=0)/epsilon), scale = 1, headwidth = 6)
    plt.plot(lc_x, lc_y,label='mu = %.1f' % mu, alpha = .4,c = 'red')
    plt.title("Y - Perturbation Net Direction & Strength")
    cbar = plt.colorbar()
    cbar.set_label("$log(||A\delta|| / ||\delta||)$")
    
if (plot_traj_perturbations):
    print('Plotting perturbation trajectories...')
    if not plot_limit_cycle:
        lc_x, lc_y, _ = get_limit_cycle(data)
    
    if not (plot_eigen_decomp):
        ed = eigen_decomp(data)

    #find index of point on limit cycle with maximum eigenvalue(neg)

    max_idx = np.argmax(ed["l_neg"])
    min_idx = np.argmin(ed["l_neg"])
    idxs = np.linspace(0,len(lc_x)-1,num = 100,dtype = int)
    #idxs = [max_idx, min_idx]
    #idxs = [-4] 
    pts = perturb_limit_cycle(data, lc_x, lc_y,  u, indices = idxs)
    mins = np.zeros((pts.shape[1],pts.shape[2]))
    plt.figure()    
    for i in np.arange(len(idxs)):
        print('%i/%i'%(i+1,len(idxs)))
        plt.plot(pts[0,:,i],pts[1,:,i])

    plt.plot(lc_x,lc_y,c='red')           

if plot_pert_along_evecs:
    if not plot_limit_cycle:
        lc_x, lc_y, lc_t = get_limit_cycle(data)
    
    if not plot_eigen_decomp:
        ed = eigen_decomp(data)
    
    # given indices, perturb them along an eigenvector and plot the convergence results
    idxs = np.linspace(0, len(lc_x)-1,num = 800, dtype = int )
    fig, ax1 = plt.subplots()
    ttcs = []
    for j,i in enumerate(idxs):
        print("%i/%i..."%(j+1,len(idxs)))
        evec = epsilon * ed["evec_neg"][:,i]
        pert = (lc_x[i] + evec[0], lc_y[i] + evec[1])
        # compute a trajectory starting at pert
        pert_data = VanderPolSim(mu = mu, x0 = pert[0], y0 = pert[1], T = T, dt = dt).run_sim()    
        pert_traj_x = pert_data.y[0,:]
        pert_traj_y = pert_data.y[1,:]
        pert_traj_ts = pert_data.t
        
        # compute time to convergence
        ttcs.append(time_to_convergence(lc_x, lc_y, pert_traj_x, pert_traj_y, pert_traj_ts, delta))
        
        
    ax1.scatter(lc_t[idxs] / lc_period(lc_t),ttcs, label = "time to convergence")
    
    ax2 = ax1.twinx()

    ax2.plot(lc_t[idxs]/ lc_period(lc_t), ed["l_neg"][idxs],'--',label='$\lambda_{-}$')    
    ax2.plot(lc_t[idxs]/ lc_period(lc_t), ed["l_pos"][idxs],'--',label='$\lambda_{+}$')
    ax2.axhline(y=0)
    plt.legend()

if (plot_convergence_analysis):
    print("Plotting convergence analysis...")
    if not plot_limit_cycle:
        lc_x, lc_y, lc_t = get_limit_cycle(data)
    
    
    if not plot_eigen_decomp:
        ed = eigen_decomp(data)
        

    # provide indices here
    
    idxs = np.linspace(0,len(lc_x)-1,num = 100, dtype = int) 
    phases = np.asarray([*lc_time_to_phase(lc_t)])[idxs]
    
    pert_starts = [
        (
            lc_x[idxs[i]] + u[0],
            lc_y[idxs[i]] + u[1]
        )
         for i in np.arange(len(idxs))
    ]
    
    traj_dists = []
    traj_xs = []
    traj_ys = []
    ttcs = []
    plt.figure()
    plt.xlabel("Time")
    plt.ylabel("Distance")
    plt.axhline(y = delta)

    num_pts = len(idxs)
    for  idx, init_loc in enumerate(pert_starts):
        print('%i/%i...'%(idx+1,num_pts))
        # simulate trajectory
        traj = VanderPolSim(T = data.T, dt = data.dt, mu = data.mu,
                             x0 = init_loc[0],
                             y0 = init_loc[1]
                             ).run_sim()  # copy orig sim but with diff starting point
        traj_x = traj.y[0,:]
        traj_y = traj.y[1,:]
        traj_xs.append(traj_x)
        traj_ys.append(traj_y)
        ttcs.append(time_to_convergence(lc_x, lc_y, traj_x, traj_y, traj.t, delta))
        traj_dists.append(get_traj_dist_from_lc(lc_x, lc_y, traj_x, traj_y))
        
        # plot dist to limit cycle, ttc as vert line, delta as horz line
        plt.plot(data.t, traj_dists[-1])
        plt.axvline(x = ttcs[-1])
        
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx() 
    ax1.scatter(phases, ttcs / lc_period(lc_t))
    ax1.set_ylabel('Time to convergence (distance <= %.3f) / period T' %delta)

    ax2.plot(phases, ed['l_neg'][idxs])
    ax2.plot(phases, ed['l_pos'][idxs])
    ax2.set_ylabel('$\lambda$')
    plt.xlabel('Starting index along limit cycle')

if (ttc_vs_phase_vs_mu):
    print('Beginning convergence vs phase mu sweep...')
    mus = np.asarray([4])
    num_pts = 100
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    if not plot_eigen_decomp:
        ed = eigen_decomp(data)
    for mu in mus:
        sim = VanderPolSim(mu = mu)
        data = sim.run_sim()
        ts = data.t
        xs = data.y[0,:]
        ys = data.y[1,:]
        
        lc_x, lc_y, lc_t = get_limit_cycle(data)
        idxs = np.linspace(0,len(lc_x)-1,num = num_pts, dtype = int) 
        phases = np.asarray([*lc_time_to_phase(lc_t)])[idxs]
        
        u = np.zeros((len(idxs, 2)))
        for i in np.arange(len(idxs)):
            u[i,0] = ed["evec_pos"][i,0]
            u[i,1] = ed["evec_pos"][i,1]
        
        pert_starts = [
            (
                lc_x[idxs[i]] + u[i,0],
                lc_y[idxs[i]] + u[i,1]
            )
             for i in np.arange(len(idxs))
        ]
        
        traj_dists = []
        traj_xs = []
        traj_ys = []
        ttcs = []
    
        num_pts = len(idxs)
        
        for  idx, init_loc in enumerate(pert_starts):
            print('mu = %.3f, index %i/%i...'%(mu,idx+1,num_pts))
            # simulate trajectory
            traj = VanderPolSim(T = data.T, dt = data.dt, mu = data.mu,
                                 x0 = init_loc[0],
                                 y0 = init_loc[1]
                                 ).run_sim()  # copy orig sim but with diff starting point
            traj_x = traj.y[0,:]
            traj_y = traj.y[1,:]
            ttcs.append(time_to_convergence(lc_x, lc_y, traj_x, traj_y, traj.t, delta))
            
        
        ax1.plot(phases, ttcs / lc_period(lc_t) ,'-o',label='$\mu = %.3f$, period = %.3f'%(mu,lc_period(lc_t)))
      #  ax2.plot([*lc_to_phase(lc_x, lc_y, lc_t).keys()],lc_x)
    ax1.set_ylabel('Time to convergence (distance <= %.3f) / period T' %delta)
    plt.xlabel('Starting phase of perturbation limit cycle')
    ax1.legend()

lc_x, lc_y, lc_t = get_limit_cycle(data)
lat_data, true_data = latent_error_traj(data, lc_x[-1]+ u[0], lc_y[-1] + u[1])


plt.figure()
plt.plot(lat_data.y[0,:],lat_data.y[1,:],label='Approximated Trajectory')
plt.plot(true_data.y[0,:], true_data.y[1,:],label='True Trajectory')
plt.legend()

diff_x = np.square(lat_data.y[0,:] - true_data.y[0,:])
diff_y = np.square(lat_data.y[1,:] - true_data.y[1,:])
#app_err = np.sqrt(diff_x + diff_y)
app_err = np.abs(lat_data.y[0,:] - true_data.y[0,:])
plt.figure()
plt.plot(true_data.t, true_data.y[0,:],label='True Trajectory')
plt.plot(true_data.t, lat_data.y[0,:],label='Approx Trajectory')
plt.plot(true_data.t, app_err, label='Error')
plt.legend()
print("Simulation Complete.")
plt.show()




#Workspace down here




