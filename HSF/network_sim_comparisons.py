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
  

T = 10
dt = 1e-3
p = 1

A =  - np.eye(2)
B = np.eye(2)
x0 = np.asarray([.5, 0])
sin_func = lambda t :  np.asarray([np.cos( (1/4) * np.pi*t), np.sin( (1/4) * np.pi*t)])
lds = sat.LinearDynamicalSystem(x0, A, B, u = sin_func , T = T, dt = dt)

lam = 1
N = 4
D_scale = 1
D = D_scale * np.eye(N)[0:A.shape[0],:]

sc_net = SelfCoupledNet(T=T, dt=dt, N=N, D=D, lds=lds, lam=lam, t0=0, spike_trans_prob = p, dimensionless=True)
pcf_net = 
sc_data = sc_net.run_sim() 




#sc_data = run_sim(N, 1)
#plot_rmse_vs_phi_const_time_window()
plt.show(block=True)




