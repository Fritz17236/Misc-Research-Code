
# exploration & plotting of analytic results form andronov-hopf oscillator


import numpy as np
import matplotlib.pyplot as plt




def ttc(eps):
    ''' time to convergence as function of perturbation along radial axis'''
    return 1 / (2 + 3*eps + eps**2)


def err_x(eps, t, phi_0, w):
    ''' Error between approximate and actual trajectory for radial perturb eval along x axis'''
    term1 = eps - (2*eps + 3*eps**2 + eps**3)*t
    term2 = np.cos(w*t + phi_0)
    
    return np.abs(term1 )#* term2)



## sweep over phi
eps = .1
ts = np.linspace(0,ttc(eps),num=10000)
phis = np.linspace(0, 2*np.pi, num = 5)
w = 2 * np.pi

plt.figure()
for phi in phis:
    errs_x = err_x(eps, ts, phi, w)
    plt.plot(ts, errs_x,label='$\phi = %.2f$'%phi)
    plt.title("Sweeping $\phi$")
     
plt.legend() 


## now sweep over epsilon
eps = np.linspace(0, 10, num = 10)


w = 2 * np.pi

plt.figure()
for e in eps:
    ts = np.linspace(0,ttc(e),num=1000)
    errs_x = err_x(e, ts, 0, w)
    #plt.plot(ts, errs_x,label='$\epsilon = %f$'%e)
    plt.scatter(e, np.linalg.norm(errs_x),label='$\epsilon = %f$'%e)
     
plt.title("Sweeping $\epsilon$")
plt.xlabel("time ")
plt.ylabel("error in approximation along radial axis")
plt.legend() 
plt.show()


