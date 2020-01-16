# Numerical Simulations for Exploring Van der Pol Oscillator

# Chris Fritz 1/15/2020

##import statements
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import cmath

## van der pol eqs
# dx/dt = y
# dy/dt = mu * (1-x**2) * y  - x

def y_dot(x, y, mu):
    return mu * (1-x*x) * y - x

def x_dot(y):
    return y

def deriv(t, X):
    ''' derivative function for use in integrator. Assumes mu defined globally'''
    x = X[0]
    y = X[1]
    num = np.asarray([x_dot(y), y_dot(x, y, mu)])
    return num 

def l_pos(x, y, mu):
    ''' eigenvalue 0 of linear approximation'''
    num = mu * (1 - x**2) + cmath.sqrt(mu**2 * (1 - x**2)**2 + 4 * (2*mu*x*y - 1) )
    return np.real(num / 2)

def l_neg(x, y, mu):
    ''' eigenvalue 0 of linear approximation'''
    num = mu * (1 - x**2) - cmath.sqrt(mu**2 * (1 - x**2)**2 + 4 * (2*mu*x*y - 1) )
    return np.real(num / 2)


## consider simulation OO framework: fast simulatable object that you provide difeqs for
plt.figure(1)
plt.figure(2)

## sweep over various values of mu

mus = np.asarray([0, 1, 4, 8])
## parameters
T = 50     # simulation time
dt = .01  # time step
mu = 3     # damping
x0 = .5    # init x
y0 = .5    # init y


    ## run simulation
for mu in mus:
    data = solve_ivp(deriv, (0, T), np.asarray([x0, y0]), t_eval = np.arange(0, T, dt))

    ## data processing

    # extract from struct
    ts = data.t
    xs = data.y[0,:]
    ys = data.y[1,:]

    # compute eigenvalues
    ls_p = [l_pos(xs[j],ys[j],mu) for j in np.arange(len(xs))]
    ls_n = [l_neg(xs[j],ys[j],mu) for j in np.arange(len(xs))]


    # x-component of neg eigvec
    eigvec_xs = [-(mu - mu*xs[j]**2 + cmath.sqrt(-4+mu**2 - 2*mu**2*xs[j]**2 + mu**2*xs[j]**4 + 8*mu*xs[j]*ys[j]))
                 / (2*(-1 + 2*mu*xs[j]*ys[j]))
                 for j in np.arange(len(xs))]

    ## Plots

    # Phase space
    plt.figure(1)
    plt.plot(xs, ys,label='mu = %.1f' % mu)

    plt.figure(2)
    plt.plot(ts, ls_n, label='mu = %.1f' %mu)
    

plt.figure(1)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

plt.figure(2)
plt.xlabel('t')
plt.ylabel('L_neg')
plt.legend()

plt.show()


