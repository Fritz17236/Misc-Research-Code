'''
Created on Feb 5, 2020

@author: chris
'''
import numpy as np
import matplotlib.pyplot as plt




plt.rcParams['figure.figsize'] = [8, 4.5]
plt.rcParams['figure.dpi'] = 200


# euler step solver


r0 = 1
phi0 = 1
epsilon = .01
t0 = 0
T = 2
dt = 10**-5
w = 2 * np.pi
t = 0

rs = [r0]
phis = [phi0]
ts = [t0]
noise_freq = 1  #maximum frequency of noise in Hz

# euler simulation, except at each timestep draw from gaussian distribution with mean 0 and variance epsilon 

rand_rs = np.random.normal(loc=0, scale = epsilon, size = int(T/dt) + 10) 
rand_rs = [k for k in rand_rs]

count = 0
num_steps_until_noise = (1/noise_freq)

while t < T:
    r_dot = rs[-1] - rs[-1]**3  
    phi_dot = w
    
    rs.append(rs[-1] + r_dot * dt )
    phis.append(phis[-1] + phi_dot * dt + epsilon * rand_rs.pop(0))
    
    ts.append(t)
    t += dt
    
    
    
rs = np.asarray(rs)
phis = np.asarray(phis)

plt.figure()
plt.plot(rs * np.cos(phis), rs * np.sin(phis), label='Noisy')
plt.plot(np.cos(phis), np.sin(phis), label= 'No Noise')
plt.legend()
plt.show()
    







#region OLD CODE
# rand_rs  = np.random.choice([-eps_r, eps_r], size = int( 2 *T/dt)).tolist()
# rand_phis  = np.random.choice([-1, 1], size = int( 2 *T/dt)).tolist()
# 
# count = 0
# noise_cnt = 1000
# while t < T:
#     r_dot = rs[-1] - rs[-1]**3 
#     phi_dot = w
#     
#     if count == noise_cnt:
#         pert = rand_rs.pop()
#         phi_sign = rand_phis.pop()
#         pert_phi_actual = eps_phi / rs[-1] * phi_sign
#         pert_phi_approx = eps_phi * phi_sign
#         count = 0
#     else:
#         pert = 0 
#         pert_phi_actual = 0
#         pert_phi_approx = 0
#         count += 1
#         
#     rs.append(rs[-1] + r_dot * dt + pert)
#     phis_2d.append(phis_2d[-1] + phi_dot * dt + pert_phi_actual)
#     phis_approx.append(phis_approx[-1] + phi_dot * dt + pert_phi_approx )
#     
#     ts.append(t)
#     t += dt
#     
#     
#     
# rs = np.asarray(rs)
# phis_2d = np.asarray(phis_2d)
# phis_approx = np.asarray(phis_approx)
# plt.figure()
# plt.plot(rs * np.cos(phis_2d), rs * np.sin(phis_2d), label='2d system')
# plt.plot(np.cos(phis_approx), np.sin(phis_approx), label= '1d approx')
# 
# 
# plt.figure()
# plt.plot(ts, np.sin(phis_2d), label='2d actual')
# plt.plot(ts,np.sin(phis_approx),label='1d approx')
# plt.legend()
# 
# plt.show()
#     
#endregion









