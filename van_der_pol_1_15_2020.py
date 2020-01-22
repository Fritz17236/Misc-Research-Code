# Numerical Simulations for Exploring Van der Pol Oscillator

# Chris Fritz 1/15/2020

#TODO: 
# Plot nullclines
# Modify perturb_limit_cycle to only iterate ver provided indices
# Compute Latent Phase

##import statements
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import argrelextrema
import cmath

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

def eigen_decomp(data, limit_cycle = True):
    ''' 
    Given simulation data, return the eigendecomposition of the 
    linearization at each point around the limit cycle.
    Returns data as dictionary with the following entries:
    eig["l_neg"] negative root eigenvalues at each of the N data points
    eig["l_pos"] positive root eigenvalues
    eig["evec_neg"] eigenvector associated with l_neg
    eig["evec_pos"] eigenvector associated with l_pos
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
    
    for i in np.arange(N):
        l_negs[i] = l_neg(xs[i], ys[i], mu)
        l_poss[i] = l_pos(xs[i], ys[i], mu)
        
        evec_negs[:,i] = evec_neg(xs[i], ys[i], mu)
        evec_poss[:,i] = evec_pos(xs[i], ys[i], mu)
    
    eig_dec = {
        "l_neg" : l_negs,
        "l_pos" : l_poss,
        "evec_neg" : evec_negs,
        "evec_pos" : evec_poss
        }  
    return eig_dec      

def perturb_limit_cycle(data, lc_x, lc_y, u, indices = [-1],  noise = False, noise_str = 1):
    '''
    Given data 
    and a perturbation vector u, compute the limit cycle, apply perturbation u to every
    point on the limit cycle, and simulate the trajectory of each point.
    Returns a 2 x T/dt x N matrix of (x,y) points up to time T for perturbed trajectory n
    Step specifies the number of points to skip between simulations
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

def dist_to_limit_cycle(lc_x, lc_y, x, y, x_only = False):
    '''
    Compute the distance of (x,y) to the limit cycle (lc_x, lc_y),
    defined here as the minimum distamce of (x,y) to any point on the 
    limit cycle defined by (lc_x,lc_y)
    If x_only = True, the distance is the absolute value in the x dimension 
    '''
    N = len(lc_x)
    
    xs = np.ones((N,)) * x
    ys = np.ones((N,)) * y
    
    dxs = lc_x - xs
    dys = lc_y - ys
    
    if x_only:
        min_dist = np.min(np.abs(dxs))
        
    else:
        d_squareds = np.square(dxs)+ np.square(dys)
        min_dist = np.sqrt(np.min(d_squareds))
        
    return min_dist
    

mu = 2
T = 25
dt = .0001
sim = VanderPolSim(mu = mu, T = T, dt = dt)
data = sim.run_sim()
ts = data.t
xs = data.y[0,:]
ys = data.y[1,:]

## Data Analysis & Plotting
# Configure by setting boolean values 
plot_trajectory = False    # Plot the (x,y) and (t,x), (t,y) trajectories of the simulation
plot_limit_cycle = False  # Assuming at least 2 full periods of oscillation, compute & plot the limit cycle trajectory
plot_eigen_decomp = False # Compute the eigenvalues/eigenvectors along the limit cycle & display them
plot_perturbation_analysis = False # comment here 
plot_noisy_start = True  # comment here

if (plot_trajectory):    
    
    plt.figure()
    plt.scatter(xs, ys)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Phase Space mu = %.2f" %mu)
    
    
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
    lc_x, lc_y, lc_t, idxs = get_limit_cycle(data, indies = True) # get exactly 1 limit cycle
    plt.figure()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(lc_x, lc_y,label='mu = %.1f' % mu)
    plt.legend()
    plt.title("Limit Cycle Phase Portrait")
    
if (plot_eigen_decomp):

    ed = eigen_decomp(data)
     
    plt.figure()
    plt.plot(ts, ed["l_neg"],label= '$\lambda_{-}$')
    plt.plot(ts, ed["l_pos"],label= '$\lambda_{+}$')
    plt.xlabel('t')
    plt.ylabel('$Re (\lambda) $')
    plt.legend()
    
    plt.figure()
    
    if not (plot_limit_cycle):
        lc_x, lc_y, lc_t, idxs = get_limit_cycle(data, indices=True)
    
    plt.quiver(lc_x, lc_y, ed["evec_neg"][0,idxs], ed["evec_neg"][1,idxs], ed["l_neg"][idxs], scale = 8, headwidth = 6)
    plt.title("Linear Stability (Negative Root Eigenvalue")
    plt.plot(lc_x, lc_y,label='mu = %.1f' % mu, alpha = .4,c = 'red')
    plt.colorbar()
    
    plt.figure()
    plt.quiver(lc_x, lc_y, ed["evec_pos"][0,idxs], ed["evec_pos"][1,idxs], ed["l_pos"][idxs], scale = 8, headwidth = 6)
    plt.plot(lc_x, lc_y,label='mu = %.1f' % mu, alpha = .4,c = 'red')
    plt.title("Linear Stability (Positive Root Eigenvalue")
    plt.colorbar()
    
if (plot_perturbation_analysis):
    delta = .1  # Perturbation Strength
    
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
    plt.quiver(xs, ys, net_x[0,:], net_x[1,:], np.log(np.linalg.norm(net_x,axis=0)/delta), scale = 1, headwidth = 6)
    plt.plot(lc_x, lc_y,label='mu = %.1f' % mu, alpha = .4,c = 'red')
    plt.title("X - Perturbation Net Direction & Strength")
    cbar = plt.colorbar()
    cbar.set_label("$log(||A\delta|| / ||\delta||)$")
    
    
    plt.figure()
    plt.quiver(xs, ys, net_y[0,:], net_y[1,:], np.log(np.linalg.norm(net_y,axis=0)/delta), scale = 1, headwidth = 6)
    plt.plot(lc_x, lc_y,label='mu = %.1f' % mu, alpha = .4,c = 'red')
    plt.title("Y - Perturbation Net Direction & Strength")
    cbar = plt.colorbar()
    cbar.set_label("$log(||A\delta|| / ||\delta||)$")
    
if (plot_noisy_start):
    
    if not plot_limit_cycle:
        lc_x, lc_y, _ = get_limit_cycle(data)
    
    if not (plot_eigen_decomp):
        ed = eigen_decomp(data)
    
    epsilon = .1 # perturbation strength
    u = epsilon * np.asarray([1, 0])
    
    #find index of point on limit cycle with maximum eigenvalue(neg)

    max_idx = np.argmax(ed["l_neg"])
    
    pts = perturb_limit_cycle(data, lc_x, lc_y,  u, indices = [max_idx])
    mins = np.zeros((pts.shape[1],))
    plt.figure()    
    for i in np.arange(pts.shape[2]):
        plt.plot(pts[0,:,i],pts[1,:,i])

#         for j in np.arange(pts.shape[1]):
#             mins[j] = dist_to_limit_cycle(lc_x, lc_y, pts[0,j,i], pts[1,j,i], x_only = False)
                 
    plt.plot(lc_x,lc_y,c='red')
    
    plt.figure()
    plt.plot(ts,mins)
        


    
plt.show()



#Workspace down here




def converged(lc_x, lc_y, x, y, delta):
    '''
    Given a limit cycle specified by lc_x, lc_y, a point x,y and some distance delta,
    check if the distance of the closest point on the limit cycle to (x,y) is within delta.
    '''
    if dist_to_limit_cycle(lc_x, lc_y, x, y) <= delta:
        return True
    else:
        return False
       
def time_to_convergence(sim_data, delta, init_loc):
    
    '''
    Given a simulation along a limit cycle, a distance delta,
    and an initial point (x,y), determine the elapsed time before
    the trajectory starting at (x,y) reaches within delta distance of
    the limit cycle
    '''
    Ts = np.asarray([1, 10, 10**2, 10**4])
    
    lc_x, lc_y, lc_t = get_limit_cycle(data)
    for T in Ts:
        # simulate for T
        new_sim = VanderPolSim(x0 = init_loc[0], y0 = init_loc[1], T = T)
        # find first o
        
    print(
        "location ",
          init_loc,
          "does not converge to limit cycle within %i time units." % Ts[-1]
          )
    

def get_latent_phase(lc_x, lc_y, lc_t, x, y):
    ''' 
    Given a point (x,y) in the phase plane,  compute its latent phase.
    The latent phase is defined as the phase of the limit cycle corresponding 
    to the intersection of the trajectory starting at (x,y) with the limit cycle
    at t--> infinity. 
    Caller must provide 3 vectors of x and y coords of limit cycle lc_x, lc_y,
    and lc_t times for each point.
    '''
    
    # start simulation at initial condition & run simulation for T
    sim = VanderPolSim(x0 = x, y0 = y)
    data = sim.run_sim()
    x_traj = data.y[0,:]
    y_traj = data.y[1,:]
    
    conv_idx = -1
    plt.figure()
    plt.plot(x_traj,y_traj)
    plt.show()
    #while not converged(sim state, limit cycle)
        # compute next sim step & update
    
    # find phase    
     
    # once converged, note time and location.
    
    # given time and point x,y of intersection, compute the  
    

# run simulation to determine when min(||state traj(x,y) - limit_cycle(a,b, t)||) < delta 
# given this time, compute the latent phase:
    # theta( w * t + phi_lat ) =theta( limit_cycle(a,b, t) )
    
      
#get_latent_phase(lc_x, lc_y, lc_t, 3, 0)



