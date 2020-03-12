'''
Created on Mar 10, 2020

@author: fritz
'''
from HSFNets import *
import math


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
N = 512
lam =  1
mode = '2d cosine'
D = gen_decoder(A.shape[0], N, mode)
