'''
Created on Mar 10, 2020

@author: fritz
'''
from HSFNets import *
from scipy.special import lambertw


# compute first 3 spike times of a regular gap-junction deneve net

#setup decoder uniformly distributed along unit circle
N = 10
lam = 1
d = 2
mode = '2d cosine'
D = gen_decoder(2, N, mode)

#set lds to start at origin and follow first col of decoder
A =  np.zeros((d,d))
B = np.eye(d)
u0 = D[:,0]
x0 = np.asarray([0, 0])
T = 30
sim_dt = 1e-3
lds_dt = .001
lds = sat.LinearDynamicalSystem(x0, A, u0, B, u = lambda t: u0, T = T, dt = lds_dt)

net = GapJunctionDeneveNet(T=T, dt=sim_dt, N=N, D=D, lds=lds, lam=lam, t0=0, thresh = 'zero-spike')

data = net.run_sim()



plt.figure()
for i in np.arange(N-1):
    plt.plot(data['t'],data['V'][i,:],label=i)
plt.legend()
plt.axhline(0)
omega = np.real(lambertw(lam))
sec_isi = omega / lam

print('first spike should be %f'%1)
print('second spike should be %f'%(1 +sec_isi) )


plt.figure()
for i in np.arange(N):
    plt.plot(1 / np.diff(data['O'][str(i)]),label=i)
plt.legend()
   
# plt.figure()
# plt.plot(data['t'], data['x_hat'][0,:])
# plt.plot(data['t_true'],data['x_true'][0,:])
plt.show()
