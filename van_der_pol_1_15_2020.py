# Numerical Simulations for Exploring Van der Pol Oscillator

# Chris Fritz 1/15/2020

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
        return solve_ivp(
            self.deriv,   # Derivative function
            (0, self.T),       # Total time interval
            np.asarray([self.x0, self.y0]),  # Initial State
            t_eval = np.arange(0, self.T, self.dt)  # Returned evaluation time points
        )


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
    
    xs = xs[int(len(xs)/2):]
    ys = ys[int(len(ys)/2):]
    ts = ts[int(len(ts)/2):]
    
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

def eigen_decomp(data, mu):
    ''' 
    Given simulation data, return the eigendecomposition of the 
    linearization at each point around the limit cycle.
    Returns data as dictionary with the following entries:
    eig["l_neg"] negative root eigenvalues at each of the N data points
    eig["l_pos"] positive root eigenvalues
    eig["evec_neg"] eigenvector associated with l_neg
    eig["evec_pos"] eigenvector associated with l_pos
    '''
    #loop through data
    xs = data.y[0,:]
    ys = data.y[1,:]
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

plot_single_limit_cycle = False


mus = np.asarray([4])
## Run Simulation
# sweep over various values of mu
#mus = np.asarray([0, 1, 4, 8])
for mu in mus:
    sim = VanderPolSim(mu = mu)
    data = sim.run_sim()
    print('running sim with mu  = %.2f' %mu)
    
    # extract from struct
    ts = data.t
    xs = data.y[0,:]
    ys = data.y[1,:]

    lc_x, lc_y, lc_t = get_limit_cycle(data) # get exactly 1 limit cycle
    


if (plot_single_limit_cycle):
    ## Plotting
    plt.figure(0)
    plt.plot(xs, ys)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Simulation Trajectory mu = %.2f" %mu)
    
    plt.figure(1)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(lc_x, lc_y,label='mu = %.1f' % mu)
    plt.legend()
    plt.title("Limit Cycle Phase Portrait")
    
    ed = eigen_decomp(data, mu)
     
    plt.figure(2)
    plt.plot(ts, ed["l_neg"],label= '$\lambda_{-}$')
    plt.plot(ts, ed["l_pos"],label= '$\lambda_{+}$')
    plt.xlabel('t')
    plt.ylabel('$Re (\lambda) $')
    plt.legend()
    
    plt.figure(3)
    xs, ys, _, idxs = get_limit_cycle(data, indices=True)
    
    plt.quiver(xs, ys, ed["evec_neg"][0,idxs], ed["evec_neg"][1,idxs], ed["l_neg"][idxs], scale = 8, headwidth = 6)
    plt.title("Linear Stability (Negative Root Eigenvalue")
    plt.plot(lc_x, lc_y,label='mu = %.1f' % mu, alpha = .4,c = 'red')
    plt.colorbar()
    
    plt.figure(4)
    plt.quiver(xs, ys, ed["evec_pos"][0,idxs], ed["evec_pos"][1,idxs], ed["l_pos"][idxs], scale = 8, headwidth = 6)
    plt.plot(lc_x, lc_y,label='mu = %.1f' % mu, alpha = .4,c = 'red')
    plt.title("Linear Stability (Positive Root Eigenvalue")
    plt.colorbar()
    
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
    
    plt.figure(5)
    plt.quiver(xs, ys, net_x[0,:], net_x[1,:], np.log(np.linalg.norm(net_x,axis=0)/delta), scale = 1, headwidth = 6)
    plt.plot(lc_x, lc_y,label='mu = %.1f' % mu, alpha = .4,c = 'red')
    plt.title("X - Perturbation Net Direction & Strength")
    cbar = plt.colorbar()
    cbar.set_label("$log(||A\delta|| / ||\delta||)$")
    
    
    plt.figure(6)
    plt.quiver(xs, ys, net_y[0,:], net_y[1,:], np.log(np.linalg.norm(net_y,axis=0)/delta), scale = 1, headwidth = 6)
    plt.plot(lc_x, lc_y,label='mu = %.1f' % mu, alpha = .4,c = 'red')
    plt.title("Y - Perturbation Net Direction & Strength")
    cbar = plt.colorbar()
    cbar.set_label("$log(||A\delta|| / ||\delta||)$")
    
    plt.show()


# for each point along the limit cycle, perturb in x


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
    
    conv_idx = 0
    plt.figure()
    plt.plot(x_traj,y_traj)
    plt.show()
    #while not converged(sim state, limit cycle)
        # compute next sim step & update
        
     
    # once converged, note time and location.
    
    # given time and point x,y of intersection, compute the  
    
    
    
# run simulation to determine when min(||state traj(x,y) - limit_cycle(a,b, t)||) < delta 
# given this time, compute the latent phase:
    # theta( w * t + phi_lat ) =theta( limit_cycle(a,b, t) )
    
      
get_latent_phase(lc_x, lc_y, lc_t, 3, 0)



