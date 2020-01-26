
# exploration & plotting of analytic results form andronov-hopf oscillator


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


plt.rcParams['figure.figsize'] = [6, 4.5]
plt.rcParams['figure.dpi'] = 200
plt.close()



def ttc(eps):
    ''' time to convergence as function of perturbation along radial axis'''
    return 1 / (2 + 3*eps + eps**2)


def err_x(eps, t, phi_0, w):
    ''' Error between approximate and actual trajectory for radial perturb eval along x axis'''
    term1 = eps - (2*eps + 3*eps**2 + eps**3)*t
    term2 = np.cos(w*t + phi_0)
    
    return np.abs(term1 * term2)



## sweep over phi
eps = .1
ts = np.linspace(0,ttc(eps),num=1000)
phis = np.linspace(0, 2*np.pi, num = 1000)
show_trajs_phi = np.linspace(0, len(phis)-1, num = 6, dtype = int)
w = 2 * np.pi

err_areas_phi = []
colors = cm.cool(phis/np.max(phis))

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
plt.savefig("voltage_error_phi_sweep_traj.png", bbox_extra_artists=(lgd,), bbox_inches='tight') 


plt.figure("phi_err_area")
plt.scatter(phis, err_areas_phi)
plt.title("Voltage Approximation Error (Trajectory Norm)")
plt.ylabel("Voltage Approximation Error")
plt.xlabel("Phase of Perturbation")
plt.savefig("voltage_error_phi_sweep_area.png",bbox_inches='tight') 

## now sweep over epsilon
eps = np.linspace(0, 10, num = 1000)
show_trajs_eps = np.linspace(0, len(eps)-1, num = 15, dtype = int)
err_areas_eps = []
for i, e in enumerate(eps):
    ts = np.linspace(0,ttc(e),num=100)
    errs_x = err_x(e, ts, 0, w)
    plt.figure("eps_err_traj")
    if i in show_trajs_eps:
        plt.plot(ts, errs_x, color=cm.cool(e/np.max(eps)), label='$\epsilon = %f$'%e)
    err_areas_eps.append(np.linalg.norm(errs_x))
    
     
plt.title("Voltage Approximation Error for various $\epsilon$")
plt.xlabel("Time")
plt.ylabel("Error Along Voltage (x) axis")
plt.legend()
plt.savefig("voltage_error_epsilon_sweep_traj.png",bbox_inches='tight') 


plt.figure("eps_err_area")
plt.title("Approximation Error Area versus Perturbation Strength")
plt.xlabel("$\epsilon$")
plt.ylabel("Error Along Voltage (x) axis (Trajectory Norm)")
plt.plot(eps, err_areas_eps)

plt.savefig("voltage_error_epsilon_sweep_area.png",bbox_inches='tight') 


plt.figure("ttc_vs_eps")
plt.plot(eps, ttc(eps))
plt.xlabel("$\epsilon$")
plt.ylabel("Time to Convergence vs $\epsilon$")
plt.savefig("ttc_epsilon_sweep_area.png",bbox_inches='tight') 


plt.show()


