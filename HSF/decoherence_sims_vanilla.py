'''
Created on Mar 12, 2020

@author: fritz
'''
from HSF.HSFNets import * #@unusedwildimport squelch
import matplotlib.pyplot as plt

def ortho_gaussian_noise(net):
    ''' Return gaussian noise but projected onto the decohered basis '''
    sigma = ortho_gaussian_noise.sigma
    dec = ortho_gaussian_noise.decoh
    mag = ortho_gaussian_noise.magnitude
    np.random.seed(0)

    noise = np.random.normal(loc=0, scale=sigma, size = (net.N,1))
    
    ortho_noise = dec @ noise    
    ortho_noise = ortho_noise / np.linalg.norm(ortho_noise)
    return mag * ortho_noise


def gaussian_noise(net):
    ''' Return gaussian noise'''
    sigma = gaussian_noise.sigma
    mag = gaussian_noise.magnitude
    np.random.seed(0)

    noise = np.random.normal(loc=0, scale=sigma, size = (net.N,1))
    noise = noise / np.linalg.norm(noise)

    return mag * noise

#Sim config


def avg_error(data):
    ''' Compute the average error for each dimension of state (x) ''' 
    
    d = data['A'].shape[0]
    avg_errs = np.zeros((d,1))
    T = data['t'][-1] - data['t'][0]
        
    for i in np.arange(d):
        len_hat = len(data['x_hat'][0,:])
        len_true = len(data['x_true'][0,:])
         
        assert( len_hat == len_true), "x_hat does not have same length as x_true (%i, %i)"%(len_hat, len_true)
        z = data['x_hat'][i,:] - data['x_true'][i,:]
        avg_errs[i] =   data['dt'] * np.sum(z) / T 
        
    return avg_errs

run_sweep = False    
        
A =  np.zeros((2,2))
A[0,1] = 1
A[1,0] = -1
A =   2 *np.pi *A
B = 0*np.eye(2)
u0 = np.zeros((A.shape[0]))
x0 = np.asarray([0, 1])
T = 10
sim_dt = 1e-3
lds_dt = 1e-3
lds = sat.LinearDynamicalSystem(x0, A, u0, B, u = lambda t: 1*np.ones((B.shape[1],1)), T = T, dt = lds_dt)
N = 256
lam =  1
mode = '2d cosine'
D = gen_decoder(A.shape[0], N)
decoh = (np.linalg.pinv(D) @ D  - np.eye(N))

ortho_gaussian_noise.decoh = decoh
ortho_gaussian_noise.magnitude = 1
gaussian_noise.magnitude = 1

if run_sweep:
    num_trials = 1
    num_sigs = 1
    noise_sigs = np.linspace(0,1,num=num_sigs)
    
    avg_errs = np.zeros((A.shape[0], num_sigs))
    avg_errs_dec = np.zeros((A.shape[0], num_sigs))
    
    for l in np.arange(num_trials):
        print('Trial %i/%i'%(l+1, num_trials))
        for j in np.arange(num_sigs):
            print('Magnitude %i/%i'%(j+1, num_sigs))
            
            
            ortho_gaussian_noise.sigma = noise_sigs[j]
            gaussian_noise.sigma = noise_sigs[j]        
            
            net_dec = GapJunctionDeneveNet(T, sim_dt, N, D, lds, lam, t0=0)
            net_dec.suppress_console_output = True
            
            net = GapJunctionDeneveNet(T, sim_dt, N, D, lds, lam, t0=0)
            net.suppress_console_output = True
            
            net_dec.inject_noise = (True, ortho_gaussian_noise)
            net.inject_noise = (True, gaussian_noise)
            
            data_dec = net_dec.run_sim()
            data = net.run_sim()
            
            avg_errs[:,j:] += avg_error(data)/num_trials
            avg_errs_dec[:,j:] += avg_error(data_dec)/num_trials
    


    plt.plot(noise_sigs, avg_errs[0,:], label='$x_1$ Raw Noise')
    #plt.plot(noise_sigs, avg_errs[1,:], label='$x_2$ Raw Noise')
    plt.plot(noise_sigs, avg_errs_dec[0,:], label='$x_1$ Decohered Noise')
    #plt.plot(noise_sigs, avg_errs_dec[1,:], label='$x_2$ Decohered Noise')
    plt.xlabel('Noise Magnitude')
    plt.ylabel('Average Error')
    plt.title("Noise versus Average Error")
    plt.legend()
    plt.show()
        
# run sim
# inject ortho noise
# compute avg error
# compute isi stats
# increase noise strength
 

# look at error trajectory over time
# run sim, one with gausisan noise regular
# run sim, one with ortho noise

ortho_gaussian_noise.decoh = decoh
ortho_gaussian_noise.magnitude = 1e-4
ortho_gaussian_noise.sigma = 1

gaussian_noise.magnitude = 1e-4
gaussian_noise.sigma = 1

net_raw = GapJunctionDeneveNet(T, sim_dt, N, D, lds, lam, t0=0, thresh='not full')
net_ortho = GapJunctionDeneveNet(T, sim_dt, N, D, lds, lam, t0=0, thresh='not full')
net_clean = GapJunctionDeneveNet(T, sim_dt, N, D, lds, lam, t0=0, thresh='not full')

net_raw.inject_noise = (True, gaussian_noise)
net_ortho.inject_noise = (True, ortho_gaussian_noise)

data_ortho = net_ortho.run_sim()
data_raw = net_raw.run_sim()
data_clean = net_clean.run_sim()

plt.plot(data_raw['t'], data_ortho['x_hat'][0,:],'--',label='ortho')
plt.plot(data_raw['t'], data_raw['x_hat'][0,:],'-',label='raw')
plt.plot(data_raw['t'], data_ortho['x_true'][0,:],label='true solution')
plt.plot(data_raw['t'], data_clean['x_hat'][0,:],'-.',label='no noise')
plt.legend()

ts = data_raw['t']
# plot error trajectories
# plt.figure()
# plt.plot(ts, data_raw['error'][0,:],label = 'raw')
# plt.plot(ts, data_ortho['error'][0,:],label = 'ortho')
# plt.plot(ts, data_clean['error'][0,:],label = 'clean')
# plt.legend()

# plt.figure()
# ortho_diff = np.cumsum(data_ortho['x_hat'][0,:] - data_clean['x_hat'][0,:])
# raw_diff =  np.cumsum(data_raw['x_hat'][0,:] - data_clean['x_hat'][0,:])
# ts = data_raw['t']
# plt.plot(ts, ortho_diff, label = 'ortho noise dist from clean')
# plt.plot(ts, raw_diff, label = 'raw noise dist from clean')
#plt.legend()
    
plt.show()
