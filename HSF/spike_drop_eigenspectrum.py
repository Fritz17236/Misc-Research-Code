'''
Created on Mar 16, 2020

@author: fritz
'''
from HSF.HSFNets import * #@unusedwildimport squelch 
import matplotlib.colors as colors
import timeit
#Figure Params
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['figure.figsize'] = [16, 9]
plt.rcParams['font.size'] = 20


#Sim Params
A =  np.zeros((2,2))
A[0,1] = 1
A[1,0] = -1
A = 10 * A
N = 64
lam = 1
mode = '2d cosine'
p = 1
ps = [0.01, .1, .3, .5, .7, .9, .99, 1]
#ps = [1]

#Generate Data    

B = np.eye(2)
u0 = np.asarray([0, 0])
x0 = np.asarray([0, 1])
T = 5
dt = 1e-3
lds = sat.LinearDynamicalSystem(x0, A, B, u = lambda t: u0 , T = T, dt = dt)
D = gen_decoder(A.shape[0], N, mode=mode)


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
    
#net = GapJunctionDeneveNet(T=T, dt=dt, N=N, D=D, lds=lds, lam=lam, t0=0, thresh = 'not full')    

#for p in ps:

for p in ps:
    D = D (1 - 1-p)
    net = SpikeDropDeneveNet(T=T, dt=dt, N=N, D=D, lds=lds, lam=lam, p=p, t0=0, thresh = 'not full')
    data = net.run_sim()
    
    plt.figure(1)
    plt.clf()
    plt.xlabel('Time')
    plt.ylabel('$\hat{x}$')
    ax = plt.gca()
    ax2 = ax.twinx() 
    ax2.set_ylabel('$x_{true}$')
    ax.plot(data['t'], data['x_hat'][0,:],c = 'red', label = r'$\hat{x}$')
    ax2.plot(data['t_true'], data['x_true'][0,:],label=r'$x_{true}$')
    plt.title('State Space Estimation & Actual p = %.2f'%p)
    plt.gcf().legend()
    plt.savefig('./Figures/f1p=%.2f.png'%p)
    
    err_pow, dec_pow = eigen_spec_partition(data)
    fig = plt.figure(2)
    plt.ylabel('Relative Power of Subspace')
    plt.xlabel("Time")

    plt.clf()
    ax = plt.gca()
    ax2 = ax.twinx()
    tot_pow = err_pow + dec_pow
    tow_pow = [1 for j in tot_pow if not np.nonzero(j)]
    rel_err = np.divide(err_pow, tot_pow)
    rel_dec = np.divide(dec_pow, tot_pow)
    ax.plot(data['t'], rel_err , label='Error Power')
    ax.plot(data['t'], rel_dec, label='Relative Decoherence Power')
    ax2.plot(data['t'],tot_pow,c='green',label='Total Power')
    ax2.set_ylabel('Total Power')
    plt.title('Fraction of Total Energy Decohered p=%.2f'%p)
    fig.legend(borderaxespad= 10)
    plt.savefig('./Figures/f2p=%.2f.png'%p)


    plt.figure(3)
    plt.clf()
    es = eigen_spec_full(data, D)[0]
    plt.imshow(es,
                cmap = 'seismic',
                norm=colors.SymLogNorm(linthresh=.01, vmin=np.min(es), vmax=np.max(es))
                )
    plt.axis('auto')
    plt.xlabel('Time')
    plt.ylabel('Right Eigenvector j')
    plt.title('Time course of Voltage Eigenspectrum p=%.2f'%p)
    cbar = plt.colorbar()
    cbar.set_label('Projection $c_j$')
    
    num_ticks = 10
    idxs = np.arange(0, len(data['t']), step = int(len(data['t'])/num_ticks))
    
    x_label_list = np.round(data['t'][idxs],2).astype(np.unicode)
    ax = plt.gca()
    ax.set_xticks(idxs)
    ax.set_xticklabels(x_label_list)
    plt.savefig('./Figures/f3p=%.2f.png'%p)
plt.show()
