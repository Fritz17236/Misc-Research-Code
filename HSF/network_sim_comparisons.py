'''
Created on Jun 30, 2020

@author: fritz
'''
import sys
sys.path.append('..')
import scipy
from HSFNets import *
from utils import *

# Simulate Classic Deneve Net 

plt.close()
def run_sim(lam, N):

    A =  -np.eye(2)

    #A =  2 * np. pi * A
    #N = 32
    #lam =  1
    mode = '2d cosine'
    p = .9

    D = gen_decoder(A.shape[0], N, mode=mode)
    #D = gen_decoder(A.shape[0], N)
    B = np.eye(2)
    u0 = D[:,0]
    x0 = np.asarray([.5, .5])
    T = 10
    dt = 1e-3

    lam_v = 1


    sin_func = lambda t :   np.asarray([np.sin(2*np.pi*t), np.cos(2*np.pi*t)])

    lds = sat.LinearDynamicalSystem(x0, A, B, u = sin_func , T = T, dt = dt)




    #gj_net = GapJunctionDeneveNet(T=T, dt=dt, N=N, D=D, lds=lds, lam=lam, t0=0)
    #classic_net = ClassicDeneveNet(T=T, dt=dt, N=N, D=D, lds=lds, lam=lam, lam_v=lam_v, t0=0)
    sc_net = SelfCoupledNet(T=T, dt=dt, N=N, D=D, lds=lds, lam=lam, t0=0)


    #gj_data = gj_net.run_sim()
    #classic_data = classic_net.run_sim()
    sc_data = sc_net.run_sim()

    return sc_data
    #return gj_data, classic_data, sc_data

#gj_data, classic_data, sc_data = run_sim(10, 4)
sc_data = run_sim(10, 4)


plt.close('all')
plt.ion()
plt.figure(figsize=(16,9))
#plt.figure()
for i in range(4):
    plt.plot(sc_data['t'],sc_data['V'][i,:], label='Neuron %i Voltage'%i)
#plt.plot(sc_data['t'],sc_data['V'][4,:], label= 'Neuron > %i Voltage'%i)
plt.legend()
plt.title("Neuron Membrane Potentials")
plt.xlabel("Simulation Time")
plt.ylabel('Membrane Potential')
plt.savefig('membrane_potential_plot.png',bbox_inches='tight')


plt.figure(figsize=(16,9))
#plt.figure()
plt.imshow(sc_data['V'],extent=[0,sc_data['t'][-1], 0,3],vmax=np.max(sc_data['V']), vmin=np.min(sc_data['V']))
plt.xlabel("Time")
plt.axis('auto')
plt.colorbar()
plt.title('Neuron Membrane Potentials )')
plt.ylabel('Neuron #')
plt.savefig('membrane_potential_image.png',bbox_inches='tight')


 

plot_step = 10

plt.ion()
for i in range(sc_data['x_hat'].shape[0]):
    plt.figure(figsize=(16,9))
    #plt.figure()
    plt.xlabel('Time')
    plt.ylabel('Dimension %i '%i)
    #plt.plot(gj_data['t'][0:-1:plot_step], gj_data['x_hat'][i,0:-1:plot_step],c='g',label='Gap Junction Network')
    #plt.plot(classic_data['t'][0:-1:plot_step], classic_data['x_hat'][i,0:-1:plot_step],c='c',label='Classic Deneve Network')
    plt.plot(sc_data['t'][0:-1:plot_step], sc_data['x_hat'][i,0:-1:plot_step],c='r',label='Self Coupled Network')
    plt.plot(sc_data['t'][0:-1:plot_step], sc_data['x_true'][i,0:-1:plot_step],c='k',label='True Dynamical System')
    plt.title('Network Decode Dimension %i '%i)
    plt.legend()
    #plt.ylim([-1.1, 1.1])
    plt.savefig('network_decode_dim_%i.png'%i,bbox_inches='tight')
    
    plt.figure(figsize=(16,9))
    #plt.figure()
    plt.xlabel('Time')
    plt.ylabel('Dimension %i '%i)
    
    #gj_err = rmse(gj_data['x_hat'][i,0:-1:plot_step], gj_data['x_true'][i,0:-1:plot_step])
    #classic_err = rmse(classic_data['x_hat'][i,0:-1:plot_step], classic_data['x_true'][i,0:-1:plot_step])
    sc_err = rmse(sc_data['x_hat'][i,0:-1:plot_step], sc_data['x_true'][i,0:-1:plot_step])
    
    #plt.plot(gj_data['t'][0:-1:plot_step], gj_data['x_hat'][i,0:-1:plot_step] - gj_data['x_true'][i,0:-1:plot_step],c='g',label='Gap Junction Network, RMSE=%.3f'%gj_err)
    #plt.plot(classic_data['t'][0:-1:plot_step], classic_data['x_hat'][i,0:-1:plot_step] - classic_data['x_true'][i,0:-1:plot_step],c='c',label='Classic Deneve Network, RMSE=%.3f'%classic_err)
    plt.plot(sc_data['t'][0:-1:plot_step], sc_data['x_hat'][i,0:-1:plot_step] - sc_data['x_true'][i,0:-1:plot_step],c='r',label='Self Coupled Network, RMSE=%.3f'%sc_err)
    plt.title('Decode Error Dimension %i '%i)
    plt.legend()
    #plt.ylim([-1.1, 1.1])
    plt.savefig('decode_error_dim_%i.png'%i,bbox_inches='tight')

print(sc_data['vth'])

plt.show(block=True)




