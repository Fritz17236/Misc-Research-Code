'''
Van der Pol Oscillator Simulation & Analysis
'''
import Simulation_Analysis_Toolset as sat
import numpy as np
import matplotlib.pyplot as plt
import cmath


## TODO:
# latent_phase computation & implementation

class VanderPolSim(sat.DynamicalSystemSim):
    '''Extend DynamicalSystemSim class for Van der Pol Oscillator'''
    def __init__(self, T = 50, dt = .01, mu = 3, x0 = .5, y0 = .5):
    
        X0 = np.asarray([x0, y0])
        super().__init__(X0, T, dt)
        self.mu = mu   # damping coefficient 
        

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

    
    def copy_sim(self):
        return VanderPolSim(
            T = self.T,
            dt = self.dt,
            mu = self.mu,
            x0 = self.X0[0],
            y0 = self.X0[1]
            )

class VanderPolAnalyzer(sat.PlanarLimitCycleAnalyzer):
    '''
    Van der Pol is a 2D system so extend PlanarLimitCycleAnalyzer class. 
    '''
    
    def lc_eigen_decomp(self, sim_data, limit_cycle):
        
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
        
        xs = limit_cycle['X'][0,:]
        ys = limit_cycle['X'][1,:]
        mu = sim_data['mu']
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
            "evec_pos" : evec_poss,
            }  
        
        return eig_dec



{}
##################### Simulation Parameters 
mu = 3
T = 50
dt = .001

## Analysis Parameters
epsilon = .5         # perturbation strength
u =  epsilon * np.asarray([1, 0])  # Perturb in x direction 
delta = epsilon*10**-3  # threshold for convergence (converged if dist <= delta)

##################### Run Simulation
sim = VanderPolSim(mu = mu, T = T, dt = dt)
data = sim.run_sim()
vpa = VanderPolAnalyzer()

xs = data['X'][0,:]
ys = data['X'][1,:]
ts = data['t']

################### Data Analysis & Plotting Configuration 
plot_trajectory            = 0   # Plot the (x,y) and (t,x), (t,y) trajectories of the simulation including nullclines

plot_limit_cycle           = 0   # Assuming at least 2 full periods of oscillation, compute & plot the limit cycle trajectory

plot_eigen_decomp          = 0   # Compute the eigenvalues/eigenvectors along the limit cycle & display them

plot_perturbation_analysis = 0   # Perturb along an eigenvector & compute its linearized growth for each point on the limit cycle

plot_traj_perturbations    = 1   # Numerically simulate a given perturbation along given points of a limit cycle

plot_convergence_analysis  = 0   # Simulate given perturbation for chosen indices & compute their distance to limit cycle vs phase

ttc_vs_phase_vs_mu         = 0   # Similar to plot convergence analysis, except do so for various values of mu


print("Running Simulation")

if (plot_trajectory):    
    print('Plotting trajectory ... ')
    plt.figure()
    plt.plot(xs, ys, label='Oscillator Trajectory')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Phase Space mu = %.2f" %mu)
     
    #plot nullclines
    x_nullcline = np.linspace(-2*np.max(np.abs(xs)), 2*np.max(np.abs(xs)), num = 1000)
    y_nullcline = [  x / (mu * (1 - x**2))  for x in x_nullcline]
    plt.plot(x_nullcline, y_nullcline, '--',color='black', label = 'Y nullcline') 
    plt.axhline(y = 0, label= 'X-nullcline', color='green')
    plt.ylim([1.5 * np.min(ys), 1.5*np.max(ys)])
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

    limit_cycle = vpa.get_limit_cycle(data)
    plt.figure()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(limit_cycle['X'][0,:], limit_cycle['X'][1,:],label='mu = %.1f' % mu)
    plt.legend()
    plt.title("Limit Cycle Phase Portrait")
    
if (plot_eigen_decomp):
    print('Plotting eigendecomposition...')
    
    if not (plot_limit_cycle):
        limit_cycle = vpa.get_limit_cycle(data)

    ed = vpa.lc_eigen_decomp(data, limit_cycle)
    
    lc_t = limit_cycle['t']
    lc_x = limit_cycle['X'][0,:]
    lc_y = limit_cycle['X'][1,:]
    lc_p = vpa.lc_period(limit_cycle)
     
    plt.figure()
    plt.plot((lc_t-lc_t[0]) / lc_p, ed["l_neg"],label= '$\lambda_{-}$')
    plt.plot((lc_t-lc_t[0]) / lc_p, ed["l_pos"],label= '$\lambda_{+}$')
    plt.xlabel('Fraction of period time')
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
    
    if not plot_eigen_decomp:
        if not plot_limit_cycle:
            limit_cycle = vpa.get_limit_cycle(data)
            lc_x = limit_cycle['X'][0,:]
            lc_y = limit_cycle['X'][1,:]
            lc_t = limit_cycle['t']
        
        ed = vpa.lc_eigen_decomp(data, limit_cycle)
    
    # Project a horizontal perturbation x onto the eigvecs scaled by eigvals & quiver plot 
    pert_x = epsilon * np.asarray([1, 0])
    pert_y = epsilon * np.asarray([0, 1])
    
    N = len(lc_t)
    
    net_x = np.zeros((2, N))
    net_y = np.zeros((2, N))
    
    plot_idxs = np.linspace(0, N-1, num = N, dtype = int)
    
    for i in np.arange(N):
        net_x[:,i] = (
            np.exp(ed["l_pos"][i]) * pert_x@ed["evec_pos"][:,i] # project onto E-vecs & scale
         + np.exp(ed["l_neg"][i]) * pert_x@ed["evec_neg"][:,i]
         )
         
        net_y[:,i] = (
            np.exp(ed["l_pos"][i]) * pert_y@ed["evec_pos"][:,i] # project onto E-vecs & scale
         + np.exp(ed["l_neg"][i]) * pert_y@ed["evec_neg"][:,i]
         ) 
    
    plt.figure("x_pert_net")
    plt.quiver(lc_x[plot_idxs], lc_y[plot_idxs], 
               net_x[0,plot_idxs],
                net_x[1,plot_idxs],
                 np.log(np.linalg.norm(net_x,axis=0)/epsilon), scale = 50, headwidth = 10
                 )    
    plt.title("X - Perturbation Net Direction & Strength")
    cbar = plt.colorbar()
    cbar.set_label("log(||$A \epsilon$|| / ||$\epsilon$||)")       
    plt.plot(lc_x, lc_y,label='mu = %.1f' % mu, alpha = .4,c = 'red')
    
    
    plt.figure("y_pert_net")
    plt.quiver(lc_x[plot_idxs], lc_y[plot_idxs], 
           net_y[0,plot_idxs],
            net_y[1,plot_idxs],
             np.log(np.linalg.norm(net_x,axis=0)/epsilon), scale = 50, headwidth = 10
             )
    plt.plot(lc_x, lc_y,label='mu = %.1f' % mu, alpha = .4,c = 'red')
    plt.title("Y - Perturbation Net Direction & Strength")
    cbar = plt.colorbar()
    cbar.set_label("$log(|A\epsilon| / |\epsilon|)$")
    
if (plot_traj_perturbations):
    print('Plotting perturbation trajectories...')

    if not plot_limit_cycle:
        limit_cycle = vpa.get_limit_cycle(data)
    
    if not plot_eigen_decomp:
        ed = vpa.lc_eigen_decomp(data, limit_cycle)
    
    # given indices, perturb them along an eigenvector and plot the convergence results
    idxs = np.linspace(0, len(limit_cycle['t'])-1,num = 100, dtype = int )
    us = limit_cycle['X'] + epsilon * ed['evec_neg']
    pert_sims = vpa.perturb_limit_cycle(sim, limit_cycle, us, idxs)
    
    plt.figure("perturbation_trajectories")
    for i in idxs:
        pert_xs = pert_sims['%i'%i]['X'][0,:]
        pert_ys = pert_sims['%i'%i]['X'][1,:]
        plt.plot(pert_xs, pert_ys, label='sim %i'%i)
    
    plt.xlabel('X (Voltage)')
    plt.ylabel('Y')
    plt.title('Perturbations Along Radial Axis$')
       


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

print("Simulation Complete")

plt.show()