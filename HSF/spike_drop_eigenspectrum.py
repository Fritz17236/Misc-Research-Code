'''
Created on Mar 16, 2020

@author: fritz
'''
from HSF.HSFNets import * #@unusedwildimport squelch

A =  np.zeros((2,2))
A[0,1] = 1
A[1,0] = -1
A = 10 * A
N = 128
lam =  5
mode = '2d cosine'
p = .99

    
D = gen_decoder(A.shape[0], N, mode=mode)
B = np.eye(2)
u0 = .001*D[:,0]
x0 = np.asarray([1, 1])
T = 1
sim_dt = 1e-4
lds_dt = 1e-3
lds = sat.LinearDynamicalSystem(x0, A, u0, B, u = lambda t: u0 , T = T, dt = lds_dt)


def eigen_spec_partition(data):
    '''
    Given a simulation, compute the power between ortho basis and error basis at each time point
    '''
    dec = data['dec']
    V = data['V']
    t = data['t']
    N = len(V[:,0])
    num_pts = len(t)
    dec_pow = np.zeros((num_pts,))
    err_pow = np.zeros((num_pts,))
    d_dag_d = np.linalg.pinv(data['D']) @ data['D']
    for i in np.arange(num_pts):
        dec_pow[i] = np.linalg.norm((d_dag_d - np.eye(N)) @ V[:,i])
        err_pow[i] = np.linalg.norm(V[:,i]) - dec_pow[i]
    
    return err_pow, dec_pow


def eigen_spec_full(data, D):
    '''
    Given a decoder matrix and data, factor the matrix into its right eigenbasis
    (in N-space), project the voltage at each time point onto each eigenbasis and return the 
    projection of each vector over time
    '''
    u,s,vt = np.linalg.svd(D)
    N = len(data['V'][:,0])
    num_pts = len(data['V'][0,:])
    eig_spec = np.zeros((N, num_pts))
    for i in np.arange(num_pts):
        eig_spec[:,i] = vt @ data['V'][:,i]
    return eig_spec, u, s, vt 
    
    
net = SpikeDropDeneveNet(T, sim_dt, N, D, lds, lam, p, t0=0, thresh = 'not full')


data = net.run_sim()


plt.figure()
plt.plot(data['t'], data['x_hat'][0,:], label = r'$\hat{x}$')
plt.plot(data['t_true'], data['x_true'][0,:],label=r'$x_{true}$')
plt.legend()


err_pow, dec_pow = eigen_spec_partition(data)
plt.figure()
plt.plot(data['t'], err_pow, label='Error Power')
plt.plot(data['t'], dec_pow, label='Decoherence Power')
plt.legend()


plt.figure()
es = eigen_spec_full(data, D)[0]
plt.imshow(np.log(np.abs(es)+1))
plt.axis('auto')
plt.xlabel('Time')
plt.ylabel('Right Eigenvector')
cbar = plt.colorbar()
cbar.set_label('voltage projection')


plt.show()
