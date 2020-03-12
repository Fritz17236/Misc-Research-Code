'''
Created on Mar 12, 2020

@author: chris
'''
'''
Created on Mar 10, 2020
Eigendecomposition of Decoder basis in R2 and RN
@author: fritz
'''
from HSFNets import *
import math





N = 512
d = 2
mode = '2d cosine'
D = gen_decoder(d, N, mode)

dtd = D.T@D

def fft_mtx(N):
    ''' Create an NxN FFT matrix '''
    F = np.zeros((N,N), dtype = np.complex)
    F_norm = np.zeros(F.shape, dtype = np .complex)
    for l in np.arange(N):
            F[l,:] = [np.exp(1j *  l * k * 2 * np.pi / N) for k in np.arange(N)]
            
            
            # normalize real and complex components separately
            real_norm = np.linalg.norm(np.real(F[:,l]))
            imag_norm = np.linalg.norm(np.imag(F[:,l]))
            
            if  math.isclose(real_norm, 0):
                real_part = np.zeros(F_norm[l,:].shape)
            else:
                real_part = np.real(F[:,l]) / real_norm
                
            if math.isclose(imag_norm, 0):
                imag_part = np.zeros(F_norm[l,:].shape)
            else:
                assert(imag_norm is not np.isnan(imag_norm))
                imag_part = np.imag(F[:,l]) / imag_norm
                
            F_norm[:,l] = real_part   + 1j * imag_part   
            
    return  F, F_norm

s, v = np.linalg.eigh(dtd)
sort_idxs = np.argsort(-s)
s = s[sort_idxs]
v = v[:,sort_idxs]



F, F_norm = fft_mtx(N)
eigdft = np.diag(F.T @ dtd @ F)
sort_idxs = np.argsort(-eigdft)
eigdft = eigdft[sort_idxs]
F_norm = F_norm[:,sort_idxs]
F = F[:,sort_idxs]

  
plt.figure()
plt.plot(np.real(eigdft),label = 'Analytic')
plt.plot(np.real(s),label='Numeric')
plt.xscale('log')
plt.legend()
plt.title("Eigenvalues of $D^{T}D$")
plt.xlabel('i')
plt.ylabel(r'|$\lambda_{i}$|')


res = dtd @ F_norm[:,0]
plt.figure()
plt.xlabel(r'Neuron j')
plt.ylabel(r'$V_j$')
plt.title("Eigenvectors of $D^{T}D$")
plt.plot( v[:,0], label =  r'$Re\left\{v\right\}$ Numeric')
plt.plot( v[:,1], label =  r'$Img\left\{v\right\}$ Numeric')
plt.plot( np.real(F_norm[:,0]),'--', label =  r'$Re\left\{v\right\}$ Analytic')
plt.plot( np.imag(F_norm[:,0]),'--', label =  r'$Img\left\{v\right\}$ Analytic')
plt.legend()


plt.show()