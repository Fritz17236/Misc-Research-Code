
# exploration & plotting of analytic results form andronov-hopf oscillator
 
### TODO:
## use events to maximize limit_cycle and converged approximations 
#compute phase approximation error for two x perturbations spaced out by tau
## done in theory ^^ next do in simulation!


 
import numpy as  np 
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import Simulation_Analysis_Toolset as sat
from scipy.integrate import solve_ivp
from math import isclose
from numpy import vectorize
from mpl_toolkits.mplot3d import Axes3D
from basic_units import radians
 
class AndronovHopfSim(sat.DynamicalSystemSim):    
    ''' 
    AndronovHopfSim is meant to simulate the normal form limit cycle oscillator,
    Andronov-Hopf oscillator, which undergoes a Hopf bifurcation.
    Starting coordinates should always be given in cartesian coordinates,
    although internally it computes derivative via polar coordinates 
    '''
    def __init__(self, X0, T, dt, delta=.001, t0=0, w=1):
        
        super().__init__(X0, T, dt, t0)    
        self.w = w
        self.delta = delta
        

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
            delta = self.delta
            )

        
    def run_sim(self):
        def is_converged(t, X):
            '''
            Check if state X is converged to limit cycle.
            '''

            
            if isclose(np.abs(X[0] - 1),0, rel_tol=0, abs_tol = self.delta):
                return 0
            else:
                return 1
            
        #convert cart to polar, assuming 
        x0 = self.X0[0]
        y0 =  self.X0[1]
        r,phi = cart_to_polar(x0, y0)
        self.X0 = np.asarray([r,phi])
        
        is_converged.terminal = True
        
        #run sim
        sim_data =  solve_ivp(
            self.deriv,   # Derivative function
            (self.t0, self.T),       # Total time interval
            self.X0,  # Initial State
            t_eval = np.arange(self.t0, self.T, self.dt),  # Returned evaluation time points
            method='LSODA',  #Radau solver for stiff systems
            dense_output=True,
            events = is_converged,
            rtol = 10*np.finfo(float).eps,
            atol = np.finfo(float).eps,
            )
        
        data = {}
        
        for param in self.__dict__.keys():
            data[str(param)] = self.__dict__[param]
        
        
        data['X'] = sim_data.y
        data['t'] = sim_data.t
        data["w"] = self.w
        self.X0 = np.asarray([x0, y0])
        
        

        
        # convert polar to cart
        xs, ys = polar_to_cart(data['X'][0,:], data['X'][1,:])
        data['X'][0,:] = xs
        data['X'][1,:] = ys
        
        data['conv_time'] = sim_data.t_events[0][0]
        data['conv_phase'] = cart_to_polar(data['X'][0,-1], data['X'][1,-1])[1]

        return data

    
class AndronovHopfAnalyzer(sat.PlanarLimitCycleAnalyzer):
    '''
    Van der Pol is a 2D system so extend PlanarLimitCycleAnalyzer class. 
    '''
    
    def lc_eigen_decomp(self, sim_data, limit_cycle): 
        pass
    
    
    def helper_latent_phase(self, sim, X, limit_cycle, delta, rads = True): 
        '''
        Helper function does most of the work for get_latent_phase
        and get_latent_error_trajectory
        '''
        
        # start simulation at initial condition & run simulation for T
        pert_sim = sim.same_sim_at_point(X)
        pert_sim_data = pert_sim.run_sim()
        
        # get time to convergence, 
        traj_conv_idx = self.traj_idx_of_lc_convergence(limit_cycle, pert_sim_data['X'], delta)
        
        if traj_conv_idx == -1:
            print("Could not compute convergence phase")
            return  (np.nan, np.nan, np.nan, np.nan, np.nan)
        
        pert_ttc = pert_sim_data['t'][traj_conv_idx]
        
                
        #get phase of convergence
        # trajectory state at conv_idx
        _, lc_conv_pt, lc_conv_idx = self.nearest_lc_point(limit_cycle, pert_sim_data['X'][:,traj_conv_idx])
        lc_phases = self.lc_times_to_phases(limit_cycle, rads = True)
        lc_convergence_phase = lc_phases[lc_conv_idx]
        
        # compute latent phase as (wt + latent_phase = phase_of_convergence)
        if rads:
            w = 2 * np.pi  / self.lc_period(limit_cycle)
            latent_phase = np.mod(lc_convergence_phase - w * pert_ttc, 2 * np.pi)            
    
        else:
            lc_convergence_phase = lc_convergence_phase / (2 * np.pi)
            f = 1 / self.lc_period(limit_cycle) 
            latent_phase = np.mod(lc_convergence_phase - f * pert_ttc , 1)
            
        return latent_phase, traj_conv_idx, lc_conv_idx, pert_ttc, pert_sim
    
    
    def get_latent_phase(self, sim, X, rads = False): 
        '''  
        Given a point X in the phase plane, compute its latent phase according to the
        simulation sim.
        The latent phase is defined as the phase of the limit cycle corresponding 
        to the intersection of the trajectory starting at X with the limit cycle
        Also returns time to convergence.
        With naive delta, we use the same value of delta (from the original sim) regardless
        of where we are on the phase plane. 
        '''
        
        pert_data = sim.same_sim_at_point(X).run_sim()
        conv_time = pert_data['conv_time']
        
        #another limit cycle oscillator starting at phi_0 return
        w = pert_data['w']
        
        #convergence phase is initial phase plus omega t
        
        _, conv_phases = cart_to_polar(pert_data['X'][0,:], pert_data['X'][1,:])
        conv_phases = np.unwrap(conv_phases)
        conv_phase = conv_phases[-1]
        latent_phase = np.mod(conv_phase - w * conv_time, 2 * np.pi)
        
        if not rads:
            latent_phase = latent_phase / (2 * np.pi)
  
        #w * ttc + phi_lat = phi_conv
        return (latent_phase, conv_time) 
    
        
    def get_latent_error_trajectory(self, sim, limit_cycle, X, errfunc): 
        '''
        Compute the difference at each point of a sim trajectory starting 
        at X and its latent phase approximation according to the provided 
        errfunc. 
        '''   
        (
            latent_phase,
            traj_conv_idx,
            lc_conv_idx,
            pert_ttc,
            pert_sim
        ) = self.helper_latent_phase(sim, X, limit_cycle, delta)
        
        pert_data = pert_sim.run_sim()
        
        # find limit cycle point nearest to trajectory at convergence         
        approx_sim = sim.same_sim_at_point(limit_cycle['X'][:,lc_conv_idx])
        approx_data = approx_sim.run_sim()
        return (
            pert_data['X'],
            approx_data['X'],
            self.trajectory_difference(pert_data['X'], approx_data['X'], errfunc),
            lc_conv_idx
            ) 
     
        
    def spaced_radial_perturbations(self, limit_cycle,  sim, tau, epsilon, delta, phi_0 = 0):
        '''
        Perturb a limit cycle by epsilon radially, simulate its trajectory for 
        tau time units, then perturb it again radially by epsilon.
        Concatenate & the trajectories 
        '''
        
        #initial radial perturbation by epsilon
        first_pert_sim = sim.copy_sim()
        x0, y0 = polar_to_cart(1 + epsilon, phi_0)
        first_pert_sim.X0 = np.asarray([x0, y0])
        first_pert_sim.t0 = 0
        first_pert_sim.T = tau
        first_pert_data = first_pert_sim.run_sim()
        
        # compute beginning of 2nd trajectory by perturbing along radial axis
        last_first_pert_state = first_pert_data['X'][:,-1]
        
        r, theta = cart_to_polar(first_pert_data['X'][0,-1], first_pert_data['X'][1,-1])
        r += epsilon
        x0, y0 = polar_to_cart(r, theta)
        sec_X0 = np.asarray([x0, y0])
        
        
        #2nd perturbation after time tau has occured
        sec_pert_sim = sim.copy_sim()
        sec_pert_sim.X0 = sec_X0
        sec_pert_sim.t0 = tau
        sec_pert_sim.T += tau 
        sec_pert_data = sec_pert_sim.run_sim()
        
        twice_pert_data, clip_idx = aha.concat_data_traj(first_pert_data, sec_pert_data, get_clip_idx = True)
        
        #compute the actual phase of convergence
        t_conv = twice_pert_data['t'][aha.traj_idx_of_lc_convergence(limit_cycle, twice_pert_data['X'], delta)]  

        return twice_pert_data, clip_idx, t_conv
        
        
def ttc(eps, delta):
    ''' time to convergence as function of perturbation along radial axis'''
    num = (1+2*delta+np.square(delta))
    denom = 2*delta+np.square(delta)
    coeff = 1 - 1/np.square(1+eps) 
    return 1/2 * np.log(np.divide(num * coeff , denom ))


def r_t(eps, ts):
    '''
    Radius as function of time for radial
    perturbation epsilon
    '''
    
    # first make sure ts is iterable
    
    try:
        _  =  [t for t in ts]
    except TypeError:
        ts = [ts]
        
    
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
    
    xs = np.multiply(rs, np.cos(thetas))
    ys = np.multiply(rs, np.sin(thetas))
    return (xs, ys)


def cart_to_polar(xs, ys):
    ''' convert cartesian coords to polar '''
    rs = np.sqrt(np.square(xs) + np.square(ys))
    thetas = np.arctan2(ys, xs)
    return (rs, thetas)


# def phase_approx_err(w, tau, epsilon, phi_0, delta):
#     ''' 
#     Compute the approximation error in phase for two epsilon pertrubations
#     in the x direction spaced apart by time tau with the first occurring in 
#     phase phi_0. 
#     '''
#     term_approx = epsilon 
#     term_err = np.abs(1 - r_t(epsilon, tau))
#      
#     err_time = np.abs(ttc(term_approx, delta) - ttc(term_err + term_approx, delta))
#     
#     
#     return w * err_time
#      
#     
#     
#   


def phase_approx_err(eps, w, tau):
    '''
    
    '''
    r_tau = r_t(eps, tau)
    return np.abs(eps * (1 - r_tau**-1))
    


# # Figure Configuration 


plt.rcParams['figure.figsize'] = [8, 4.5]
plt.rcParams['figure.dpi'] = 200


print("Machine epsilon for the following simulations is ", np.finfo(float).eps)

epsilon = .1  # perturbation strength

#epsilons = np.logspace(-2, 1, num = 100)
epsilons = np.linspace(.001, 10, num = 100)
r0   = 1 + epsilon  #radial perturbation by epsilon
phi0 = 1
w = 2 * np.pi 
period = 2 * np. pi / w
x0, y0 = polar_to_cart(r0, phi0)
T = 20
dt = .001
delta = epsilon * 10**-3
deltas = epsilons * 10**-3


# Data Analysis & Plotting Configuration 

run_test_suite             = 0   # Run the various test functions

plot_trajectory            = 0   # Plot the (x,y) and (t,x), (t,y) trajectories of the simulation including nullclines

plot_limit_cycle           = 0   # Assuming at least 2 full periods of oscillation, compute & plot the limit cycle trajectory

plot_traj_perturbations    = 0   # Numerically simulate a given perturbation along given points of a limit cycle

plot_conv_analysis         = 0   # Simulate given perturbation for chosen indices & compute their distance to limit cycle vs phase/phi

plot_isochron_phase_space  = 0   # Tile the phase space and compute the latent phase for each point, plot

plot_approx_err_voltage    = 0   # Given a point, compute its latent-approximated & actual trajectories and plot error along voltage axis

plot_spaced_rad_perts      = 0   # Perturb radially along the limit cycle by epsilon twice space by tau time units, compute phase approximation error

plot_approx_err_phase      = 0   # comment here  
 
plot_approx_err_phase_tau  = 0   # comment here





#endregion
 


sim = AndronovHopfSim(X0 = np.asarray([x0, y0]), delta = delta,  T = T, dt = dt, w = 2*np.pi / period)
data = sim.run_sim()
aha = AndronovHopfAnalyzer(data)
limit_cycle = aha.get_limit_cycle(data)
    
xs = data['X'][0,:]
ys = data['X'][1,:]
ts = data['t']

r_exacts = r_t(epsilon, ts)
phi_exacts = data['w']*ts 

x_exacts = r_exacts * np.cos(phi_exacts + phi0) 
y_exacts = r_exacts * np.sin(phi_exacts + phi0) 




print("Running Simulation...")

if (plot_trajectory):    

    aha.plot_phase_space()
    
    aha.plot(
        [x_exacts, y_exacts],
        mode = 'plot', 
        fig_name = 'trajectory',
        label = 'Analytic',
        ls = '--'
        )  
    
    aha.plot(
        [data['t'], x_exacts],
        fig_name = 'x_trace',
        label = 'Analytic',
        ls = '--')    
    
    
    aha.plot(
        [ts, y_exacts],
        'plot',
        'y_trace',
        label='Analytic',
        ls = '--')


if (plot_limit_cycle):
    print('Plotting limit cycle ... ')

    aha.plot_limit_cycle()    
    phis = np.linspace(0,2*np.pi,1000)
    aha.plot([np.cos(phis), np.sin(phis)],
             mode='plot',
             fig_name = 'limit cycle',
             label='Analytic',
             ls = '--')
    


if (plot_traj_perturbations):
    print('Plotting perturbation trajectories...')


    # given indices, perturb them along an eigenvector and plot the convergence results
    idxs = np.linspace(0, len(limit_cycle['t'])-1,num = 3, dtype = int )
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
    print("Plotting convergence analysis (sweeping delta)...")
    
    idxs = np.linspace(0, len(limit_cycle['t'])-1,num = 10, dtype = int )
    d_sweep = np.logspace(-9,np.log(np.max(epsilon)),num = 10)
    ttcs_numeric = []
    ttcs_exact = []
    
    for i,d in enumerate(d_sweep):
        print("%i/%i"%(i+1, len(d_sweep)))
        pert_sim = sim.copy_sim()
        pert_sim.X0 = np.asarray([1 + epsilon, 0]) # perturb radially at phi = 0
        pert_sim.delta = d
        pert_data = pert_sim.run_sim()
        ttcs_numeric.append(pert_data['conv_time'])
        ttcs_exact.append(ttc(epsilon, d))
    
    plt.figure("delta sweep")
    plt.semilogx((d_sweep)/epsilon, np.asarray(ttcs_numeric) / period, label= 'Simulated')
    plt.semilogx((d_sweep)/epsilon, np.asarray(ttcs_exact) / period,label= 'Analytic')
    plt.xlabel(r"$\frac{\delta}{\epsilon}$")
    plt.ylabel(r"$\frac{t^*}{T}$")
    plt.title("Time to convergence versus convergence limit, $\epsilon = %.2f$"%epsilon)
    plt.legend()

    print("Plotting convergence analysis (sweeping epsilon)...")
    plt.savefig('ttc_vs_perturbation_strength_delta_frac.png',bbox_inches = 'tight')

    ttcs_numeric = []
    ttcs_exact = []
    cvs_numeric = []
    # for epsilons, simulate trajectory starting at limit_cycle[0] + epsilon (in xdir)
    for i, eps in enumerate(epsilons):
        print("%i/%i"%(i+1, len(epsilons)))
        pert_sim = sim.copy_sim()
        pert_sim.X0 = np.asarray([1 + eps, 0]) # perturb radially at phi = 0
        pert_sim.delta = deltas[i]
        pert_data = pert_sim.run_sim()
        ttcs_numeric.append(pert_data['conv_time'])
        ttcs_exact.append(ttc(eps, deltas[i]))
        cvs_numeric.append(pert_data['conv_phase']/(2 * np.pi))
    
    #plot and compare to analytic ttc
    plt.figure("ttcs_vs_epsilon")
    plt.plot((epsilons), ttcs_numeric / aha.lc_period(limit_cycle), label= 'Simulated')
    plt.plot((epsilons), np.asarray(ttcs_exact) / period ,label= 'Analytic')
    plt.xlabel("$\epsilon$")
    plt.ylabel("$t^*$")
    plt.title("Time to convergence versus (radial) perturbation strength. $\delta$ = $ \epsilon/1000$")
    plt.legend()
    
    d_eps = epsilons[5] - epsilons[4]
    plt.figure("cvs vs epsilon")
    plt.plot(epsilons, np.mod(cvs_numeric, 1))
    plt.xlabel("$\epsilon$")
    plt.ylabel("Phase of Convergence")
    #plt.savefig('ttc_vs_perturbation_strength_delta_frac.png',bbox_inches = 'tight')
     
        
#endregion 


#region Workspace  Here: 

 
 
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



#endregion



if (plot_isochron_phase_space):
    
    res = 100
    xc = np.linspace(-10, 10, num = res)
    yc = np.linspace(-10, 10, num = res)

    phi_lats = np.zeros((res, res))
    count = 1
    ttcs = []
    for i, x in enumerate(xc):
        
        for j,y in enumerate(yc):
            print('%i/%i'%(count, res**2))
            count += 1
            pert_sim = sim.same_sim_at_point([x,y])
            r_pert, phi_pert = cart_to_polar(x, y)
            eps_pert = np.abs(1 - r_pert)
            delta_pert = eps_pert * 10**-6
            pert_sim.delta = delta_pert
            pert_data = pert_sim.run_sim()
            X = np.asarray([x,y])
            latent_phase = aha.get_latent_phase(pert_sim, X, rads = True)[0] 
            phi_lats[j,i] = latent_phase
         
    
    xs,ys = np.meshgrid(xc, yc)
#    plt.scatter(xs,ys,c=phi_lats,cmap='inferno')

    
    
# 
    fig = plt.figure("lat_phase_sweep")
    plt.scatter(xs,ys, c=phi_lats/(2*np.pi), cmap='magma',s = 5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Numerically Evaluated Latent Phase')
    cbar = plt.colorbar(label='Latent Phase (Normalized)')
    plt.clim(0, 1)
    plt.plot(limit_cycle['X'][0,:],limit_cycle['X'][1,:],label='Limit Cycle')
    plt.savefig('isochron_hopf_numerical.png',bbox_inches='tight')



if (plot_approx_err_voltage):
    
    
    def voltage_err(X, Y):
        ''' error along voltage axis'''
        return np.abs(X[0] - Y[0])
    
    
    (
        true_traj,
        approx_traj,
        err_traj,
        conv_idx
    ) = aha.get_latent_error_trajectory(
        sim,
        limit_cycle,
        np.asarray([x0, y0]),
        voltage_err,
        )
    
    ts = sim.dt * np.arange(len(err_traj))
 
#     plt.figure("approx err phase space")
#     plt.scatter(true_traj[0,:], true_traj[1,:],marker = 'x', label = 'True Simulated Trajectory', c = ts)
#     plt.scatter(approx_traj[0,:], approx_traj[1,:], marker = '.', label = 'Limit Cycle Approximation', c = 'green')
#     plt.colorbar()
#     plt.xlabel('X (voltage)')
#     plt.ylabel('Y')
#     plt.scatter(approx_traj[0,conv_idx], approx_traj[1,conv_idx],label='Point of Convergence', c = 'red')
#     plt.title('Approximating Perturbed Trajectory by Limit Cycle with Latent Phase')
#     plt.legend()
     
    
    plt.figure("approx error voltage ")
    plt.plot(ts, true_traj[0,:], label = 'True Simulated Trajectory')
    plt.plot(ts, approx_traj[0,:], label = 'Limit Cycle Approximation')
    plt.plot(ts, err_traj, label='Approximation Error')
    plt.xlabel('Time')
    plt.ylabel("Voltage")
    plt.legend()

 

if (plot_spaced_rad_perts):
#     taus = np.linspace(0,1,num=1000) 
#     
#     
#     for i,eps in enumerate(epsilons):
#         phase_errs = []
#         for tau in taus:
#             phase_errs.append(phase_approx_err(w, tau, eps, phi0, deltas[i]))
#         
#         plt.figure("phase approximation error")
#         plt.plot((taus/T), phase_errs,label='$\epsilon = %.3f$'%eps)
#     
#     plt.xlabel(r"$\frac{\tau}{T}$")
#     plt.ylabel('Phase Approximation Error')
#     plt.title(r'Phase Approximation Error for two pulses spaced by$ \tau. $')
#     plt.legend()

    
    plt.figure("two perts")
    taus = np.linspace(0.01, 5, num = 50)
    conv_time_actual = []
    conv_time_approx= []
    for tau in taus:
        # first run the simulation with two perturbations spaced by tau and measure time to convergence
        tp, clip_idx, act = aha.spaced_radial_perturbations(limit_cycle, sim, tau, epsilon, delta)
        conv_time_actual.append(act)
         
        # now run sim with one perturbation at second point (tau) and compute time to convergence
        x0, y0 = polar_to_cart(1 + epsilon, sim.w * tau)
        _, pct = aha.get_latent_phase(sim, np.asarray([x0, y0]), delta, limit_cycle)
        conv_time_approx.append(pct)
         
    plt.plot(taus, np.abs(np.asarray(conv_time_actual) - np.asarray(conv_time_approx)), label='numerically simulated latent phases')
    plt.xlabel(r'$\tau$')
    plt.ylabel('$\phi_{lat}$')
      


def test_get_latent_phase(sim): 
    
    aha = AndronovHopfAnalyzer()
    data = sim.run_sim()
    limit_cycle = aha.get_limit_cycle(data)
 
    lat_phases = []
    pert_phis = np.linspace(0.00,2*np.pi, num = 100) 
    for phi in pert_phis:
        x0, y0 = polar_to_cart(r0, phi)
        lat_phases.append(aha.get_latent_phase(sim, np.asarray([x0, y0]), rads=True)[0])
        
        
    # plot pert phase,
    # plot latent phase, should be straight line
    plt.figure("test latent phase")
    plt.title('Latent Phase test: should be straight line')
    plt.xlabel('$\phi_0$')
    plt.ylabel('Latent Phase')
    plt.scatter(pert_phis, lat_phases ,s = 3,label='test result')
    plt.plot(pert_phis,pert_phis,'--',c = 'red', label='correct result')
    plt.legend()
    

    
if plot_approx_err_phase_tau:
    res = 100
    eps = np.logspace(-5, 0, num = res)
    taus = np.logspace(-3,0,num=res)
    
    w_approx_err = (2 * np.pi)*(2)
    
    count = 1
    
    errs = np.zeros((res,res))
    for i, e in enumerate(eps):
        
            print('%i/%i'%(count, res**2))
            count += res
            errs[:,i] = phase_approx_err(e, w, taus)
            
    plt.figure("eps vs tau")
    xs, ys = np.meshgrid(eps, taus)
    #ax = plt.gca()
    ax = plt.axes(projection='3d')
    
    rel_err = errs / (2 * np.pi)
    my_col = cm.get_cmap('viridis', res**2)
    #inferno(rel_err/np.amax(rel_rr))
    ax.plot_surface(xs, ys,  rel_err, cmap= my_col,
                  #   norm=mpl.colors.LogNorm(vmin=10**-8,vmax=.1)
                     )
    #ax.set_yscale('log')
    #ax.set_xscale('log')
    plt.xlabel('$\epsilon$')
    plt.ylabel(r'$\tau$')
    ax.set_zlabel(r'$\psi_{err}$')
    ax.view_init(azim=140 )
    plt.title('Phase Approximation Error for 2 Perturbations')
    plt.savefig('2_pert.png')
  #  plt.colorbar()
  #  plt.clim(vmin = 10**-6, vmax = 3)



if plot_approx_err_phase:
    res = 300
    eps = np.linspace(0, 1, num = res)
    err_rs = np.linspace(0,1,num=res)
    
    count = 1
    
    errs = np.zeros((res,res))
    for i, e in enumerate(eps):
        
            print('%i/%i'%(count, res**2))
            count += res
            errs[:,i] = np.arctan2(eps*err_rs[i],1)
            
    plt.figure("eps vs tau")
    xs, ys = np.meshgrid(eps, err_rs)
    ax = plt.gca()
    plt.scatter(xs, ys, c=errs/(2*np.pi),
   #             norm=mpl.colors.LogNorm(vmin=10**-6),
                 s = 5)
    #ax.set_yscale('log')
    #ax.set_xscale('log')
    plt.xlabel('$\epsilon$')
    plt.ylabel(r'$r$')

    plt.colorbar()   
 
  
#periodic perturbations

def two_pert_pulse(sim, X, t0,  t_pulse, epsilon, tau):
    ''' 
    Starting at a point X, apply radial perturbation 
    epsilon, then wait tau units and apply an angular pulse eps/r(t). Then simulate
    until t_pulse time has elapsed. 
    '''
    #given an initital point/time (X, t0), perturb by r an amouint epsilon, simulate to tau
    X = data['X'][:-1]
    X[0] += epsilon #perturb radially by epsilon
    first_pert_sim = sim.same_sim_at_point(X)
    first_pert_sim.t0 = data['t'][-1]
    first_pert_sim.T = tau+t0
    fp_data = first_pert_sim.run_sim()
    
    
    #create a new simulation with initial state at t=tau and X[tau]. perturb angular by eps/r,
    # run to t_pulse+t0
    sec_pert_sim = sim.same_sim_at_point(fp_data['X'][:,-1])
    sec_pert_sim.X0[1] += ( epsilon / sec_pert_sim.X0[0])
    sec_pert_sim.t0 = tau
    sec_pert_sim.T = t0 + t_pulse
    sp_data = sec_pert_sim.run_sim()
    
    return aha.concat_data_traj(fp_data, sp_data)
    
t_pulse = 5
epsilon = .5
tau = .5
    
tp_data = two_pert_pulse(sim, data, t_pulse, epsilon, tau)

aha.plot(tp_data['X'], 'plot', 'two perturbations')

def two_pert_sequence(sim, t_pulse, epsilon, tau, num_pulses):
    '''
    Apply a 2 pert pulse sequence for num_pulses, combining
    trajectories after each pulse occurs. 
    '''
    pass
#start at limit cycle
# perturb r by epsilon, 
# perturb phi by epsilon/r
# wait until T_pulse has passed 
# repeat for k pulses


 
 
    
    
if (run_test_suite):
    test_get_latent_phase(sim)
 
print("Simulation Complete.")
plt.show()





