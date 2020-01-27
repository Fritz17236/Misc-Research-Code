
# exploration & plotting of analytic results form andronov-hopf oscillator
# todo: manual integraton of r(t) 
# verify results via numerical simulation 
# generalize vanderpol analysis functions into python package 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


plt.rcParams['figure.figsize'] = [6, 4.5]
plt.rcParams['figure.dpi'] = 200
plt.close()



def ttc(eps, delta):
    ''' time to convergence as function of perturbation along radial axis'''
    num = -(1+2*delta+delta**2)
    denom = 2*delta+delta**2
    coeff = 1/(1+eps)**2 - 1
    
    return 1/2 * np.log(num / denom * coeff)

def r_t(eps, t):
    '''
    Radius as function of time for radial
    perturbation epsilon
    '''
    return np.asarray([
        np.exp(t) / np.sqrt( 1/(1+eps)**2 + np.exp(2*t)-1)
          for t in ts])
    
def err_x(eps, t, phi_0, w):
    ''' Error between approximate and actual trajectory for radial perturb eval along x axis'''
    term1 = r_t(eps, t) - 1
    term2 = np.cos(w*t + phi_0)
    
    return np.abs(term1 * term2)
 


## Time to convergence 

def rmse(eps, phi_0, w):
    ''' Compute convergence rmse'''
    t = ttc(eps,delta)
    a = eps**2
    b = 4*eps**2 + 12*eps**3 + 13*eps**4 + 6*eps**5 + eps**6
    c = 4*eps**2 + 6*eps**3 + 2*eps**4
    term1 = (1/w**3) * np.sin(2*w*t + 2*phi_0) * (6*w**2*(a-c*t) + b*(6*w**2*t**2-3))
    term2 = 2*w**3*t*(6*a + t*(2*b*t-3*c))
    term3 = 3*w*(c-2*b*t)*np.cos(2*w*t+2*phi_0)
    term4 = np.sin(2*phi_0)/w**3 * (6*a*w**2 - 3*b)
    term5 = 2*w*c*np.cos(2*phi_0)
    
    return np.sqrt(
        term1 + term2 - term3 - term4 + term5
        )

eps = np.linspace(0.001,10,num=1000)
deltas = eps * .001
ttcs = [ttc(eps[i], deltas[i]) for i in np.arange(len(eps))]
plt.figure("ttc_delta_frac_eps")
plt.plot(eps, ttcs)
plt.xlabel('$\epsilon$')
plt.ylabel('Time to convergence')
plt.title("Time for radius to converge within $\delta = \epsilon / 1000$")
plt.savefig("ttc_delta_frac_eps.png",bbox_inches='tight')

delta = .001
ttcs = [ttc(e, delta) for e in eps]
plt.figure("ttc_delta_const")
plt.plot(eps, ttcs)
plt.xlabel('$\epsilon$')
plt.ylabel('Time to convergence')
plt.title("Time for radius to converge within $\delta = %f$"%delta)
plt.savefig("ttc_delta_const.png",bbox_inches='tight')

## sweep over phi & plot error trajectory
eps = .1
delta = eps / 1000
ts = np.linspace(0,ttc(eps, delta),num=1000)
phis = np.linspace(0, 2*np.pi, num = 100)
show_trajs_phi = np.linspace(0, len(phis)-1, num = 6, dtype = int)
w = 2 * np.pi
colors = cm.cool(phis/np.max(phis))
err_areas_phi = []
 
for i,phi in enumerate(phis):
    plt.figure("phi_err_traj")
    errs_x = err_x(eps, ts, phi, w)
    if i in show_trajs_phi:
        plt.plot(ts, errs_x,color=cm.cool(phi/np.max(phis)),label='$\phi_0 = %.2f$'%phi)
    err_areas_phi.append(np.linalg.norm(errs_x))
 
plt.xlabel("Time")
plt.ylabel("Approximation Error")
plt.title("Approximation Error vs Time for Various $\phi_0$")
lgd = plt.legend()
plt.savefig("voltage_error_phi_sweep_traj_delt_frac.png", bbox_extra_artists=(lgd,), bbox_inches='tight') 
 
# 
plt.figure("phi_sweep_rmse")
plt.scatter(phis, err_areas_phi)
plt.title("Voltage RMSE vs $\phi_0$, $\epsilon = %.2f$"%eps)
plt.ylabel("Voltage RMSE")
plt.xlabel("$\phi_0$")
plt.savefig("voltage_rmse_vs_phi_delt_frac.png",bbox_inches='tight') 
 
## now sweep over epsilon
eps = np.linspace(0.001, 10, num = 100)
deltas = eps/1000
deltas = np.ones(eps.shape) * .001
show_trajs_eps = np.linspace(0, len(eps)-1, num = 100, dtype = int)
err_areas_eps = []
for i, e in enumerate(eps):
    ts = np.linspace(0,ttc(e,deltas[i]),num=1000)
    errs_x = err_x(e, ts, 0, w)
    plt.figure("eps_err_traj")
    if i in show_trajs_eps:
        plt.plot(ts, errs_x, color=cm.cool(e/np.max(eps)))#, label='$\epsilon = %f$'%e)
    err_areas_eps.append(np.linalg.norm(errs_x))
     
      
plt.title("Voltage Approximation Error for various $\epsilon$, $\phi_0 = 0$")
plt.xlabel("Time")
plt.ylabel("Error Along Voltage (x) axis")
#plt.legend()
plt.savefig("voltage_error_epsilon_sweep_traj_delt_frac.png",bbox_inches='tight') 
# 
# 
plt.figure("eps_err_area")
plt.title("Approximation RMSE versus Perturbation Strength, $\phi_0 = %.2f$"%0)
plt.xlabel("$\epsilon$")
plt.ylabel("Voltage RMSE")
plt.scatter(eps, err_areas_eps)
 
plt.savefig("voltage_error_epsilon_sweep_area_delt_frac.png",bbox_inches='tight') 
 
# 
# plt.figure("ttc_vs_eps")
# plt.plot(eps, ttc(eps,delta))
# plt.title("Convergence time Versues Perturbation Strength")
# plt.xlabel("$\epsilon$")
# plt.ylabel("Time to Convergence vs $\epsilon$")
# plt.savefig("ttc_epsilon_sweep_area.png",bbox_inches='tight') 


plt.show()





