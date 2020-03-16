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




plt.rcParams['lines.linewidth'] = 2
N = 128

d = 2
mode = '2d cosine'
D = gen_decoder(d, N, mode)

dtd = D.T@D
dt_dag = np.linalg.pinv(D.T)


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


def fourier_coeffs(v, k):
    '''
    return the kth coefficients for a fourier basis in sine and cosine of a vector v
    '''
    ak = 0
    bk = 0
    N = len(v)
    for i in np.arange(N):
        ak += v[i] * np.cos(2 * np.pi * i * k / N) * 2 / N
        bk += v[i] * np.sin(2 * np.pi * i * k / N) * 2 / N
    
    
    return (ak, bk)
    
s, v = np.linalg.eigh(dtd)
s = np.sort(s)
v = v[:,np.argsort(s)]





v_cos = np.asarray([np.cos(2 * np.pi * i * 1 / N) for i in np.arange(N)])
v_cos = v_cos / np.linalg.norm(v_cos)
v_sin = np.asarray([np.sin(2 * np.pi * i * 1 / N) for i in np.arange(N)]) 
v_sin = -v_sin / np.linalg.norm(v_sin)


 

plt.figure("eigdecomp")
plt.xlabel(r'Neuron j')
plt.ylabel(r'$V_j$')
plt.title("Eigenvectors of $D^{T}D$, N = %i"%N)
plt.plot( v[:,-1], label =  r'$v_1$ Numeric, $\lambda_1 = %.3f$'%s[-1])
plt.plot( v[:,-2], label =  r'$v_2$ Numeric, $\lambda_2 = %.3f$'%s[-2])
plt.plot( v_cos,'--', label =  r'$v_{cos}$ Analytic, a1 = %.3f'%((N+1)/2))
plt.plot( v_sin,'--', label =  r'$v_{sin}$ Analytic, b1 = %.3f'%((N-1)/2))
plt.legend()
plt.show()




## assert that if we perturb the voltage by error-orthogonalized noise, the error is unchanged
# init voltage
v_init = 1*np.ones((N,))
## init error
e_init = dt_dag @ v_init
 

##add noise
noise = 10 * np.random.normal(scale = 1, size=(N,))

def ortho_proj(v, k):
    '''
    Given a vector V, project each element j of the vector onto the kth complex basis 
    exp(-1i 2pi  j k /N )
    '''
    
    N = len(v)
    #proj onto sine cosine bases
    cos_vec = []
    sin_vec = []
    ak, bk = fourier_coeffs(v, k)
    for i in np.arange(N):
        cos_vec.append(np.cos(2 * np.pi * i * k / N))
        sin_vec.append(np.sin(2 * np.pi * i * k / N))
    
    return  ( ak * np.asarray(cos_vec) + bk * np.asarray(sin_vec) )

noise_ortho =  noise - ortho_proj(noise, 1) 
ortho_norm = np.linalg.norm(noise_ortho)
noise = ortho_norm * noise / np.linalg.norm(noise)


v_p = v_init + noise
e_p = dt_dag @ v_p

v_new = v_init + noise_ortho
e_new = dt_dag @ v_new


plt.figure()
plt.plot(noise, alpha = .5, label = 'Raw Noise, Norm = %.5f'%np.linalg.norm(noise))
plt.plot(noise_ortho,'--',alpha= .8,  label= 'Orthogonalized Noise, Norm = %.5f'%np.linalg.norm(noise_ortho))
plt.plot(noise - np.linalg.pinv(D)@D@noise, alpha = .3, label='Numerical')
plt.title('Noise Added to Voltages')
plt.legend()
plt.show()

plt.figure()
plt.plot(np.arange(N), v_init,alpha = .5, label='Initial Voltage')
plt.plot(np.arange(N), v_init + noise, alpha = .5, label = 'Raw-Perturbed Voltage')
plt.plot(np.arange(N), v_new, alpha = .5,  label= 'Ortho-Perturbed Voltage')
plt.title('Voltages')
plt.legend()
plt.show()  
  
min_ex = 1.5 * np.min(e_p[0])
max_ex = 1.5 * np.max(e_p[0])
min_x = np.min([min_ex, -1])
max_x = np.max([max_ex, 1])
xlims = [min_x, max_x]

min_ey = 1.5 * np.min(e_p[1])
max_ey = 1.5 * np.max(e_p[1])
min_y = np.min([min_ey, -1])
max_y = np.max([max_ey, 1])
ylims = [min_y, max_y]

plt.figure()
plt.scatter(e_init[0], e_init[1],s= 100, marker='x', label= 'original error')
plt.scatter(e_new[0], e_new[1],s = 100, marker='*',alpha=.5, label= 'ortho noise perturbed error')
plt.scatter(e_p[0], e_p[1],s = 100, marker='.',alpha=.5, label= 'raw noise-perturbed error')
plt.xlim(xlims)
plt.ylim(ylims)
plt.title('Error of Voltages')
print(min_ey, max_ey)
print(ylims)
plt.legend()



plt.show()