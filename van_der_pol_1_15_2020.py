# Numerical Simulations for Exploring Van der Pol Oscillator

# Chris Fritz 1/15/2020

##import statements
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import argrelextrema
import cmath


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
    
        return np.asarray([x_dot(y), y_dot(x, y, mu)]) 
    
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


def get_limit_cycle(data):
    ''' 
    Given a sim data struct, compute & return the limit cycle as a
    (3, N) array where N is the number of data points in 1 period.
    (2 state variables & time variable data)
    This code assumes the 2nd half of data is exclusive limit cycle data
    i.e. the system is in steady state oscillation with no transients.
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
    return (xs[idxs], ys[idxs], ts[idxs])
    


def l_pos(x, y, mu):
    ''' eigenvalue 0 of linear approximation'''
    num = mu * (1 - x**2) + cmath.sqrt(mu**2 * (1 - x**2)**2 + 4 * (2*mu*x*y - 1) )
    return np.real(num / 2)

def l_neg(x, y, mu):
    ''' eigenvalue 0 of linear approximation'''
    num = mu * (1 - x**2) - cmath.sqrt(mu**2 * (1 - x**2)**2 + 4 * (2*mu*x*y - 1) )
    return np.real(num / 2)

## Generate Data
# sweep over various values of mu
#mus = np.asarray([0, 1, 4, 8])
mus = np.asarray([4])
for mu in mus:
    sim = VanderPolSim(mu = mu)
    data = sim.run_sim()
    print('running sim with mu  = %.2f' %mu)
    
    # extract from struct
    ts = data.t
    xs = data.y[0,:]
    ys = data.y[1,:]

    lc_x, lc_y, lc_t = get_limit_cycle(data) # get exactly 1 limit cycle
    
    ## Plot Phase Space
    plt.figure(1)
    plt.plot(lc_x, lc_y,label='mu = %.1f' % mu)

plt.figure(1)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
 
 
 
 
 
# plt.figure(2)
# plt.xlabel('t')
# plt.ylabel('L_neg')
# plt.legend()
# 
# plt.show()
# 
#     # compute eigenvalues
#     ls_p = [l_pos(xs[j],ys[j],mu) for j in np.arange(len(xs))]
#     ls_n = [l_neg(xs[j],ys[j],mu) for j in np.arange(len(xs))]
# 
# 
#     # x-component of neg eigvec
#     eigvec_xs = [-(mu - mu*xs[j]**2 + cmath.sqrt(-4+mu**2 - 2*mu**2*xs[j]**2 + mu**2*xs[j]**4 + 8*mu*xs[j]*ys[j]))
#                  / (2*(-1 + 2*mu*xs[j]*ys[j]))
#                  for j in np.arange(len(xs))]
# 

