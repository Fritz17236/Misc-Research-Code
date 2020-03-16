'''
Created on Mar 12, 2020

@author: fritz
'''
from HSFNets import *


def ortho_gaussian_noise(net):
    ''' Return gaussian noise but projected onto the decohered basis '''
    sigma = ortho_gaussian_noise.sigma
    dec = ortho_gaussian_noise.decoh
    mag = ortho_gaussian_noise.magnitude
    noise = np.random.normal(loc=0, scale=sigma, size = (net.N,1))
    
    ortho_noise = noise - dec @ noise
    #ortho_noise = ortho_noise / np.linalg.norm(ortho_noise) 
    
    return mag * ortho_noise


def gaussian_noise(net):
    ''' Return gaussian noise'''
    sigma = gaussian_noise.sigma
    mag = gaussian_noise.magnitude
    noise = np.random.normal(loc=0, scale=sigma, size = (net.N,1))
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
        avg_errs[i] =   np.sum(z) / T 
        
    return avg_errs
        
    

A =  np.zeros((2,2))
A[0,1] = 1
A[1,0] = -1
A =  A
B = 0*np.eye(2)
u0 = np.zeros((A.shape[0]))
x0 = np.asarray([0, 1])
T = 1
sim_dt = 1e-3
lds_dt = .001
lds = sat.LinearDynamicalSystem(x0, A, u0, B, u = lambda t: 1*np.ones((B.shape[1],1)), T = T, dt = lds_dt)
N = 50
lam =  1
mode = '2d cosine'
D = gen_decoder(A.shape[0], N, mode)
decoh = (np.linalg.pinv(D) @ D )



    
ortho_gaussian_noise.decoh = decoh
ortho_gaussian_noise.magnitude = .1
gaussian_noise.magnitude = .1

num_trials = 10
num_sigs = 5
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







# diff = data['x_hat'] - data['x_true']
# plt.figure()
# plt.plot(data['t'],data['x_hat'][0,:], label = r'$\hat{x}$')
# plt.plot(data['t_true'],data['x_true'][0,:],label = '$x_{true}$')
# 
# plt.title('First Decoded Component')
# plt.xlabel('t')
# plt.ylabel('$x_1$')
# plt.legend()
# 
# plt.figure()
# plt.plot(data['t'],data['x_hat'][1,:], label = r'$\hat{x}$')
# plt.plot(data['t_true'],data['x_true'][1,:],label = '$x_{true}$')
# plt.xlabel('t')
# plt.ylabel('$x_2$')
# plt.title('Second Decoded Component')

# plt.figure()
# for i in np.arange(len(data['t']), step = 1000):
#     plt.plot(data['dec'][:,i],label='t = %f'%data['t'][i])
# 
# plt.legend()    
plt.show()



