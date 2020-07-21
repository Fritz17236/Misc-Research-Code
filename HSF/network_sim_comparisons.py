'''
Created on Jun 30, 2020

@author: fritz
'''
import sys
sys.path.append('..')
import scipy
from HSFNets import *
from utils import *


plt.rcParams['font.size'] = 14
plt.rcParams['font.weight'] = 'bold'

# Simulate Classic Deneve Net 
plt.close()
def run_sim(N, p=1):

    A =  - np.eye(2)

    #A =  2 * np. pi * A
    #N = 32
    lam =  1
    mode = '2d cosine'

    D = gen_decoder(A.shape[0], N, mode=mode)
    B = np.eye(2)
    u0 = D[:,0]
    x0 = np.asarray([.5, .5])
    T = 40
    dt = 1e-5

    lam_v = 1


    sin_func = lambda t :  10 * np.asarray([np.cos( (1/4) * np.pi*t), np.sin( (1/4) * np.pi*t)])

    lds = sat.LinearDynamicalSystem(x0, A, B, u = sin_func , T = T, dt = dt)

    #gj_net = GapJunctionDeneveNet(T=T, dt=dt, N=N, D=D, lds=lds, lam=lam, t0=0)
    #classic_net = ClassicDeneveNet(T=T, dt=dt, N=N, D=D, lds=lds, lam=lam, lam_v=lam_v, t0=0)
    sc_net = SelfCoupledNet(T=T, dt=dt, N=N, D=D, lds=lds, lam=lam, t0=0, spike_trans_prob = p, dimensionless=True)
    #nd_net = SelfCoupledNet(T=T, dt=dt, N=N, D=D, lds=lds, lam=lam, t0=0, spike_trans_prob=p, dimensionless=False)

    #gj_data = gj_net.run_sim()
    #classic_data = classic_net.run_sim()
    sc_data = sc_net.run_sim()
    #nd_data = nd_net.run_sim()

    return sc_data
   # return sc_data, nd_data
    #return gj_data, classic_data, sc_data

#gj_data, classic_data, sc_data = run_sim(10, 4)
lam = 1
N = 4
sc_data = run_sim(N, 1)
# num_trials = 10
# ps = np.linspace(0, 1,num=10, endpoint=True)
# rmses = np.zeros((num_trials, len(ps)))
# data_samples = { }


# for i in range(num_trials):
#     for j in range(len(ps)):
#         sc_data = run_sim(N, ps[j])
#         rmses[i,j] = rmse(sc_data['x_hat'], sc_data['x_true'])
#         if i==0:
#             data_samples[(j, ps[j])] = sc_data
#          
# 
# 
# plot_step = 10
# for i,p in data_samples.keys():
#         plt.figure(figsize=(16,9))
#         sc_data = data_samples[(i, p)]
#         plt.plot(sc_data['t'][0:-1:plot_step], sc_data['x_hat'][0,0:-1:plot_step],c='r',label='Self Coupled Network (Dimensionless)' )
#         plt.plot(sc_data['t_true'][0:-1:plot_step], sc_data['x_true'][0,0:-1:plot_step],c='k',label='True Dynamical System')
# 
#         plt.xlabel('Time (dimensionless)')
#         plt.ylabel('Network Decode Dimension 0')
#         plt.title('Network Decode Dimension 0, trans. prob. p={0}'.format(p))
#     
# plt.figure(figsize=(16,9))
# plt.plot(ps, np.mean(rmses, axis=0))
# plt.errorbar(ps, np.mean(rmses,axis=0), yerr=np.std(rmses,axis=0))
# plt.title("RMSE vs Synaptic Transmission Probabilty  for N = {0} Neuron Network,  {1} Trials per probability".format(N, num_trials))
# plt.xlabel('Synaptic Transmission Probability')
# plt.ylabel('Signal RMSE')
# 
# plt.show()


# plt.close('all')
# plt.ion()
# plt.figure(figsize=(16,9))
# #plt.figure()
# for i in range(4):
#     plt.plot(sc_data['t'],sc_data['V'][i,:], label='Neuron %i Voltage'%i)
# #plt.plot(sc_data['t'],sc_data['V'][4,:], label= 'Neuron > %i Voltage'%i)
# plt.legend()
# plt.title("Neuron Membrane Potentials")
# plt.xlabel(r"Simulation Time (Dimensionless Units of $\tau$")
# plt.ylabel('Membrane Potential')
# plt.savefig('membrane_potential_plot.png',bbox_inches='tight')
#  
#  
plt.figure(figsize=(16,9))

cbar_ticks = np.round(np.linspace( start = np.min(sc_data['V']), stop = .5,  num = 8, endpoint = True), decimals=1)

plt.imshow(sc_data['V'],extent=[0,sc_data['t'][-1], 0,3],vmax=.5, vmin=np.min(sc_data['V']))
plt.xlabel(r"Dimensionless Units of $\tau$")
plt.axis('auto')
cbar = plt.colorbar(ticks=cbar_ticks)
cbar.set_label('$v_j$')

plt.title('Neuron Membrane Potentials')
plt.ylabel('Neuron #')
plt.yticks([.4,1.15,1.85,2.6], labels=[1, 2, 3, 4])
plt.savefig('membrane_potential_image.png',bbox_inches='tight')
 
 
  
 
plot_step = 10




#PLOT NETWORK DECODE, 2 DIMENSIONS ON ONE PLOT
plt.figure(figsize=(16,9))
plt.plot(sc_data['t'][0:-1:plot_step], sc_data['x_hat'][0,0:-1:plot_step],c='r',label='Decoded Network Estimate (Dimension 0)' )
plt.plot(sc_data['t'][0:-1:plot_step], sc_data['x_hat'][1,0:-1:plot_step],c='g',label='Decoded Network Estimate (Dimension 1)' )

plt.plot(sc_data['t_true'][0:-1:plot_step], sc_data['x_true'][0,0:-1:plot_step],c='k')
plt.plot(sc_data['t_true'][0:-1:plot_step], sc_data['x_true'][1,0:-1:plot_step],c='k',label='True Dynamical System')

plt.title('Network Decode')
plt.legend()
plt.ylim([-8, 8])
plt.xlabel(r'Dimensionless Time $\tau_s$')
plt.ylabel('Decoded State')
plt.savefig('network_decode.png',bbox_inches='tight')

#PLOT DECODE ERROR 2 DIMENSIOENS ON ONE PLOT
plt.figure(figsize=(16,9))
plt.plot(sc_data['t'][0:-1:plot_step], sc_data['x_hat'][0,0:-1:plot_step] - sc_data['x_true'][0,0:-1:plot_step],c='r',label='Estimation Error (Dimension 0)' )
plt.plot(sc_data['t'][0:-1:plot_step], sc_data['x_hat'][1,0:-1:plot_step] - sc_data['x_true'][1,0:-1:plot_step],c='g',label='Estimation Error (Dimension 1)' )


plt.title('Decode Error')
plt.legend()
plt.ylim([-8, 8])
plt.xlabel(r'Dimensionless Time $\tau_s$')
plt.ylabel('Decode Error')
plt.savefig('decode_error.png',bbox_inches='tight')

#PLOT MEMBRANE POTENTIAL IMAGE OF 4 NEURONS W/ FIXED LABELS

# plt.ion()
# for i in range(sc_data['x_hat'].shape[0]):
#     plt.figure(figsize=(16,9))
#     #plt.figure()
#     plt.xlabel('Time')
#     plt.ylabel('Dimension %i '%i)
#     #plt.plot(gj_data['t'][0:-1:plot_step], gj_data['x_hat'][i,0:-1:plot_step],c='g',label='Gap Junction Network')
#     #plt.plot(classic_data['t'][0:-1:plot_step], classic_data['x_hat'][i,0:-1:plot_step],c='c',label='Classic Deneve Network')
#     plt.plot(sc_data['t'][0:-1:plot_step], sc_data['x_hat'][i,0:-1:plot_step],c='r',label='Decoded Network Estimate (Dimension 0)' )
#     #plt.plot(nd_data['t'][0:-1:plot_step]*lam, nd_data['x_hat'][i,0:-1:plot_step],'--',c='g',label='Self Coupled Network')
#  
#     plt.plot(sc_data['t_true'][0:-1:plot_step] * lam, sc_data['x_true'][i,0:-1:plot_step],c='k',label='True Dynamical System')
#     plt.title('Network Decode Dimension %i '%i)
#     plt.legend()
#     plt.ylim([-8, 8])
#     plt.savefig('network_decode_dim_%i.png'%i,bbox_inches='tight')
#      
#     plt.figure(figsize=(16,9))
#     #plt.figure()
#     plt.xlabel(r'Dimensionless Units of $\tau$')
#     plt.ylabel('Dimension %i '%i)
#      
#     #gj_err = rmse(gj_data['x_hat'][i,0:-1:plot_step], gj_data['x_true'][i,0:-1:plot_step])
#     #classic_err = rmse(classic_data['x_hat'][i,0:-1:plot_step], classic_data['x_true'][i,0:-1:plot_step])
#     sc_err = rmse(sc_data['x_hat'][i,0:-1:plot_step], sc_data['x_true'][i,0:-1:plot_step])
#     #nd_err = rmse(nd_data['x_hat'][i,0:-1:plot_step], nd_data['x_true'][i,0:-1:plot_step])
#     #plt.plot(gj_data['t'][0:-1:plot_step], gj_data['x_hat'][i,0:-1:plot_step] - gj_data['x_true'][i,0:-1:plot_step],c='g',label='Gap Junction Network, RMSE=%.3f'%gj_err)
#     #plt.plot(classic_data['t'][0:-1:plot_step], classic_data['x_hat'][i,0:-1:plot_step] - classic_data['x_true'][i,0:-1:plot_step],c='c',label='Classic Deneve Network, RMSE=%.3f'%classic_err)
#     plt.plot(sc_data['t'][0:-1:plot_step], sc_data['x_hat'][i,0:-1:plot_step]-sc_data['x_true'][i,0:-1:plot_step],c='r',label='Self Coupled Network (Dimensionless), RMSE=%.3f'%sc_err)
#     #plt.plot(nd_data['t'][0:-1:plot_step]*lam, nd_data['x_hat'][i,0:-1:plot_step]-nd_data['x_true'][i,0:-1:plot_step],'--' ,c='g',label='Self Coupled Network, RMSE=%.3f'%sc_err)
#  
#     #plt.plot(sc_data['t'][0:-1:plot_step], -sc_data['x_true'][i,0:-1:plot_step],c='r')
#     #plt.plot(sc_data['t'][0:-1:plot_step], sc_data['x_hat'][i,0:-1:plot_step],c='r')
#     plt.title('Decode Error Dimension %i '%i)
#     plt.legend()
#     plt.ylim([-8, 8])
# 
#     plt.savefig('decode_error_dim_%i.png'%i,bbox_inches='tight')

# print(sc_data['vth'])
# print(nd_data['vth'])


plt.show(block=True)




