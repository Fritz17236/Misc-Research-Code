
# exploration & plotting of analytic results form andronov-hopf oscillator
 
### TODO:
## Simulate & compute voltage approximation error vs phi/ epsilon 


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import Simulation_Analysis_Toolset as sat

class AndronovHopfSim(sat.DynamicalSystemSim):    
    ''' 
    AndronovHopfSim is meant to simulate the normal form limit cycle oscillator,
    Andronov-Hopf oscillator, which undergoes a Hopf bifurcation.
    Starting coordinates should always be given in cartesian coordinates,
    although internally it computes derivative via polar coordinates 
    '''
    def __init__(self, X0, T, dt, t0 = 0, w = 1):
        
        super().__init__(X0, T, dt, t0)    
        self.w = w
        

    def deriv(self, t, X):
        '''
        Andronov-Hopf Oscillator
        dr/dt = r - r^3
        dphi/dt = 1 
        '''
        return np.asarray([X[0]-X[0]**3, self.w])

    
    def copy_sim(self):
        return AndronovHopfSim(
            X0 = self.X0,
            T = self.T,
            t0 = self.t0,
            dt = self.dt,
            w = self.w,
            )

        
    def run_sim(self):
        #convert cart to polar
        r = self.X0[0]
        phi =  self.X0[1]
        x,y = cart_to_polar(r, phi)
        self.X0 = np.asarray([x,y])
    
        #run sim
        data =  super().run_sim()
        data["w"] = self.w
        data['r'] = data['X'][0,:]
        data['phi'] = data['X'][1,:]
        self.X0 = np.asarray([r, phi])
        
        # convert polar to cart
        xs, ys = polar_to_cart(data['X'][0,:], data['X'][1,:])
        data['X'][0,:] = xs
        data['X'][1,:] = ys
  
        return data

    
class AndronovHopfAnalyzer(sat.PlanarLimitCycleAnalyzer):
    '''
    Van der Pol is a 2D system so extend PlanarLimitCycleAnalyzer class. 
    '''
    
    def lc_eigen_decomp(self, sim_data, limit_cycle):
        pass


        
def ttc(eps, delta):
    ''' time to convergence as function of perturbation along radial axis'''
    num = -(1+2*delta+np.square(delta))
    denom = 2*delta+np.square(delta)
    coeff = 1/np.square(1+eps) - 1 
    return 1/2 * np.log(np.divide(num * coeff , denom ))


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


def polar_to_cart(rs, thetas):
    ''' convert polar coords to cartesian coords'''
    
    xs = rs * np.cos(thetas)
    ys = rs * np.sin(thetas)
    return (xs, ys)


def cart_to_polar(xs, ys):
    ''' convert cartesian coords to polar '''
    rs = np.sqrt(np.square(xs) + np.square(ys))
    thetas = np.arctan2(ys, xs)
    return (rs, thetas)

    
{}
## Figure Configuration
plt.rcParams['figure.figsize'] = [8, 4.5]
plt.rcParams['figure.dpi'] = 200
    
    
## Simulation Paramters
epsilon = .5  # perturbation strength
r0   = 1 + epsilon  #radial perturbation by epsilon
phi0 = 0
period = 1
x, y = polar_to_cart(r0, phi0)
T = 25
dt = .0001
delta = .001

## Run Simulation
sim = AndronovHopfSim(X0 = np.asarray([x, y]), T = T, dt = dt, w = 2*np.pi / period)
aha = AndronovHopfAnalyzer()
data = sim.run_sim()
    
xs = data['X'][0,:]
ys = data['X'][1,:]
ts = data['t']

r_exacts = r_t(epsilon, ts)
phi_exacts = data['w']*ts 

x_exacts = r_exacts * np.cos(phi_exacts)
y_exacts = r_exacts * np.sin(phi_exacts) 

plt.close()


################### Data Analysis & Plotting Configuration 
plot_trajectory            = 0   # Plot the (x,y) and (t,x), (t,y) trajectories of the simulation including nullclines

plot_limit_cycle           = 0   # Assuming at least 2 full periods of oscillation, compute & plot the limit cycle trajectory

plot_traj_perturbations    = 0   # Numerically simulate a given perturbation along given points of a limit cycle

plot_conv_analysis         = 0   # Simulate given perturbation for chosen indices & compute their distance to limit cycle vs phase/phi






print("Running Simulation...")

if (plot_trajectory):    
    print('Plotting trajectory ... ')
    
    plt.figure("phase_space")
    plt.plot(xs, ys, label='Simulated')
    plt.plot(x_exacts, y_exacts,'--',label='Analytic')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Phase Space")
    plt.legend()

     
    plt.figure("voltage trace")
    plt.plot(ts,xs,label='Simulated')
    plt.plot(ts,x_exacts,'--',label='Analytic')
    plt.xlabel("t")
    plt.ylabel("X-axis (Voltage) value")
    plt.title("X-axis (Voltage) Trace")
    plt.legend()
    
    plt.figure("y trace")
    plt.plot(ts,ys,label='Simulated')
    plt.plot(ts,y_exacts,'--',label='Analytic')
    plt.xlabel("t")
    plt.ylabel("Y-axis value")
    plt.title("Y-axis Trace")
    plt.legend()
    




 

## Time to convergence 



# eps = np.linspace(0.001,10,num=1000)
# deltas = eps * .001
# ttcs = [ttc(eps[i], deltas[i]) for i in np.arange(len(eps))]
# plt.figure("ttc_delta_frac_eps")
# plt.plot(eps, ttcs)
# plt.xlabel('$\epsilon$')
# plt.ylabel('Time to convergence')
# plt.title("Time for radius to converge within $\delta = \epsilon / 1000$")
# plt.savefig("ttc_delta_frac_eps.png",bbox_inches='tight')
# 
# delta = .001
# ttcs = [ttc(e, delta) for e in eps]
# plt.figure("ttc_delta_const")
# plt.plot(eps, ttcs)
# plt.xlabel('$\epsilon$')
# plt.ylabel('Time to convergence')
# plt.title("Time for radius to converge within $\delta = %f$"%delta)
# plt.savefig("ttc_delta_const.png",bbox_inches='tight')
# 
# ## sweep over phi & plot error trajectory
# eps = .1
# delta = eps / 1000
# ts = np.linspace(0,ttc(eps, delta),num=1000)
# phis = np.linspace(0, 2*np.pi, num = 100)
# show_trajs_phi = np.linspace(0, len(phis)-1, num = 6, dtype = int)
# w = 2 * np.pi
# colors = cm.cool(phis/np.max(phis))
# err_areas_phi = []
#  
# for i,phi in enumerate(phis):
#     plt.figure("phi_err_traj")
#     errs_x = err_x(eps, ts, phi, w)
#     if i in show_trajs_phi:
#         plt.plot(ts, errs_x,color=cm.cool(phi/np.max(phis)),label='$\phi_0 = %.2f$'%phi)
#     err_areas_phi.append(np.linalg.norm(errs_x))
#  
# plt.xlabel("Time")
# plt.ylabel("Approximation Error")
# plt.title("Approximation Error vs Time for Various $\phi_0$")
# lgd = plt.legend()
# plt.savefig("voltage_error_phi_sweep_traj_delt_frac.png", bbox_extra_artists=(lgd,), bbox_inches='tight') 
#  
# # 
# plt.figure("phi_sweep_rmse")
# plt.scatter(phis, err_areas_phi)
# plt.title("Voltage RMSE vs $\phi_0$, $\epsilon = %.2f$"%eps)
# plt.ylabel("Voltage RMSE")
# plt.xlabel("$\phi_0$")
# plt.savefig("voltage_rmse_vs_phi_delt_frac.png",bbox_inches='tight') 
#  
# ## now sweep over epsilon
# eps = np.linspace(0.001, 10, num = 100)
# deltas = eps/1000
# deltas = np.ones(eps.shape) * .001
# show_trajs_eps = np.linspace(0, len(eps)-1, num = 100, dtype = int)
# err_areas_eps = []
# for i, e in enumerate(eps):
#     ts = np.linspace(0,ttc(e,deltas[i]),num=1000)
#     errs_x = err_x(e, ts, 0, w)
#     plt.figure("eps_err_traj")
#     if i in show_trajs_eps:
#         plt.plot(ts, errs_x, color=cm.cool(e/np.max(eps)))#, label='$\epsilon = %f$'%e)
#     err_areas_eps.append(np.linalg.norm(errs_x))
#      
#       
# plt.title("Voltage Approximation Error for various $\epsilon$, $\phi_0 = 0$")
# plt.xlabel("Time")
# plt.ylabel("Error Along Voltage (x) axis")
# #plt.legend()
# plt.savefig("voltage_error_epsilon_sweep_traj_delt_frac.png",bbox_inches='tight') 
# # 
# # 
# plt.figure("eps_err_area")
# plt.title("Approximation RMSE versus Perturbation Strength, $\phi_0 = %.2f$"%0)
# plt.xlabel("$\epsilon$")
# plt.ylabel("Voltage RMSE")
# plt.scatter(eps, err_areas_eps)
#  
# plt.savefig("voltage_error_epsilon_sweep_area_delt_frac.png",bbox_inches='tight') 
#  
# # 
# # plt.figure("ttc_vs_eps")
# # plt.plot(eps, ttc(eps,delta))
# # plt.title("Convergence time Versues Perturbation Strength")
# # plt.xlabel("$\epsilon$")
# # plt.ylabel("Time to Convergence vs $\epsilon$")
# # plt.savefig("ttc_epsilon_sweep_area.png",bbox_inches='tight') 
# 

if (plot_limit_cycle):
    print('Plotting limit cycle ... ')

    limit_cycle = aha.get_limit_cycle(data)
    plt.figure()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(limit_cycle['X'][0,:], limit_cycle['X'][1,:],label='Simulated')
    phis = np.linspace(0,2*np.pi,1000)
    plt.plot(np.cos(phis),np.sin(phis),'--',label='Analytic')
    plt.legend()
    plt.title("Limit Cycle Phase Portrait")

if (plot_traj_perturbations):
    print('Plotting perturbation trajectories...')

    if not plot_limit_cycle:
        limit_cycle = aha.get_limit_cycle(data)
    
    
    # given indices, perturb them along an eigenvector and plot the convergence results
    idxs = np.linspace(0, len(limit_cycle['t'])-1,num = 100, dtype = int )
    # add an (x,y) component for each perturbation
    # for each point, compute (x,y), scale by 1 + epsilon that u
    us = (1 + epsilon) * limit_cycle['X']
    pert_sims = aha.perturb_limit_cycle(sim, limit_cycle, us, idxs)
    
    plt.figure("perturbation_trajectories")
    for i in idxs:
        pert_xs = pert_sims['%i'%i]['X'][0,:]
        pert_ys = pert_sims['%i'%i]['X'][1,:]
        plt.plot(pert_xs, pert_ys, label='sim %i'%i)
    
    plt.xlabel('X (Voltage)')
    plt.ylabel('Y')
    plt.title('Perturbations Along Radial Axis')
       
if (plot_conv_analysis):
    print("Plotting convergence analysis (sweeping phi)...")
    
    if not plot_limit_cycle:
        limit_cycle = aha.get_limit_cycle(data)
    # perturb along radial axis, compute ttc, sweep phase and phi
    
    # Sweep along phi and perturb radial axis
    # given indices, perturb them along an eigenvector and plot the convergence results
    idxs = np.linspace(0, len(limit_cycle['t'])-1,num = 10, dtype = int )
    # add an (x,y) component for each perturbation
    # for each point, compute (x,y), scale by 1 + epsilon that u
    us = (1 + epsilon) * limit_cycle['X']
    pert_sims = aha.perturb_limit_cycle(sim, limit_cycle, us, idxs)
    
    phases = aha.lc_times_to_phases(limit_cycle, False)
    ttcs = []
    
    print('Computing Convergences...')
    for i, idx in enumerate(idxs):
        print('%i/%i'%(i+1, len(idxs)))
        # compute ttc of trajectory
        conv_idx = aha.idx_of_lc_convergence(limit_cycle, pert_sims[str(idx)]['X'], delta)
        ttcs.append(pert_sims[str(idx)]['t'][conv_idx])
        
        
        
    plt.figure("ttc_vs_phi")
    plt.plot(phases[idxs], ttcs/aha.lc_period(limit_cycle), label='Simulated')
    plt.axhline(ttc(epsilon, delta)/aha.lc_period(limit_cycle),label='Analytic')
    
    plt.xlabel("$\phi_{0}$")
    plt.ylabel("Time to convergence")
    plt.title("Time to convergenc e versus (radial) perturbation phase. $\epsilon = %.2f$, $\delta$ = $ \epsilon/1000$"%epsilon)
    plt.legend()
    
    
    

    print("Plotting convergence analysis (sweeping epsilon)...")
    
    if not plot_limit_cycle:
        limit_cycle = aha.get_limit_cycle(data)
    # perturb along radial axis, compute ttc, sweep phase and phi
    
    epsilons = np.linspace(0.1, 10, num = 25)
    deltas = epsilons / 1000
    #deltas = np.ones(len(epsilons)) * delta
    ttcs_numeric = []
    ttcs_exact = []
    
    # for epsilons, simulate trajectory starting at limit_cycle[0] + epsilon (in xdir)
    for i, eps in enumerate(epsilons):
        print("%i/%i"%(i+1, len(epsilons)))
        pert_sim = sim.copy_sim()
        pert_sim.X0 = np.asarray([1 + eps, 0]) # perturb radially at phi = 0
        pert_data = pert_sim.run_sim()
        ttcs_numeric.append(pert_data['t'][aha.idx_of_lc_convergence(limit_cycle, pert_data['X'], deltas[i])])
        ttcs_exact.append(ttc(eps, deltas[i]))
    
    #plot and compare to analytic ttc
    plt.figure("ttcs_vs_epsilon")
    plt.plot(epsilons, ttcs_numeric / aha.lc_period(limit_cycle), label= 'Simulated')
    plt.plot(epsilons, np.asarray(ttcs_exact) / period ,label= 'Analytic')
    plt.xlabel("$\epsilon$")
    plt.ylabel("Time to convergence (fractions of period)")
    plt.title("Time to convergence versus (radial) perturbation strength. $\phi_0 = 0$, $\delta$ = $ \epsilon/1000$")
    plt.legend()
    
    #plt.savefig('ttc_vs_perturbation_strength_delta_frac.png',bbox_inches = 'tight')
        
print("Simulation Complete.")
plt.show()





