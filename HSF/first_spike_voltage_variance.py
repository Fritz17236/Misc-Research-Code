'''
Created on Mar 10, 2020

@author: fritz
'''
from HSFNets import *

run_sim = False
A =  np.zeros((2,2))
A[0,1] = 1
A[1,0] = -1
A =  A
B = np.eye(2)
u0 = np.zeros((A.shape[0]))
x0 = np.asarray([0, 1])
T = 5
sim_dt = 1e-3
lds_dt = .001
p = .5
lds = sat.LinearDynamicalSystem(x0, A, u0, B, u = lambda t: 1*np.ones((B.shape[1],1)), T = T, dt = lds_dt)
N = 50
lam =  1
mode = '2d cosine'
D = gen_decoder(A.shape[0], N, mode)
net = SpikeDropDeneveNet(T=T, dt=sim_dt, N=N, D=D, lds=lds,p=p, lam=lam, t0 = 0)

data = net.run_sim()



# for i in num trials
    # run sim for short time
    # get first spike time
    # get voltage immediately before, assert equal to known
    # get voltage immediately after spike drop and record 


# do pca of voltage data and compare to predicted eigval/eigvecs

fig, ax = plt.subplots()
ax2 = ax.twinx()
ax.plot(data['t'],data['x_hat'][0,:],label = r'$\hat{x}$')
plt.plot(data['t_true'],data['x_true'][0,:],label = r'$x$')
plt.legend()
plt.show()
