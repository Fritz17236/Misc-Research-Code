'''
Created on Mar 12, 2020

@author: fritz
'''
from HSFNets import *


A =  np.zeros((2,2))
A[0,1] = 1
A[1,0] = -1
A =  A
B = np.eye(2)
u0 = np.zeros((A.shape[0]))
x0 = np.asarray([0, 1])
T = 10
sim_dt = 1e-3
lds_dt = .001
lds = sat.LinearDynamicalSystem(x0, A, u0, B, u = lambda t: 1*np.ones((B.shape[1],1)), T = T, dt = lds_dt)
N = 50
lam =  1
mode = '2d cosine'
D = gen_decoder(A.shape[0], N, mode)
net = GapJunctionDeneveNet(T, sim_dt, N, D, lds, lam, 0)

data = net.run_sim()

diff = data['x_hat'] - data['x_true']
plt.figure()
plt.plot(data['t'],data['x_hat'][0,:], label = r'$\hat{x}$')
plt.plot(data['t_true'],data['x_true'][0,:],label = '$x_{true}$')
plt.plot(data['t'],np.sum(data['dec'],axis = 0), label = 'sum decoherence (t)')
plt.plot(data['t'],diff[0,:],'--', label = r'$\hat{x} - x$')
plt.legend()

plt.figure()
plt.plot(data['t'],data['x_hat'][1,:], label = r'$\hat{x}$')
plt.plot(data['t_true'],data['x_true'][1,:],label = '$x_{true}$')
plt.plot(data['t'],np.sum(data['dec'],axis = 0), label = 'sum decoherence (t)')
plt.plot(data['t'],diff[1,:],'--', label = r'$\hat{x} - x$')

# plt.figure()
# for i in np.arange(len(data['t']), step = 1000):
#     plt.plot(data['dec'][:,i],label='t = %f'%data['t'][i])
# 
# plt.legend()    
plt.show()



