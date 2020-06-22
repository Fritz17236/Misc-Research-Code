'''
Created on Jun 5, 2020

@author: fritz
'''



import numpy as np 
from numpy import pi
import matplotlib.pyplot as plt
from scipy.optimize import nnls
import random
from matplotlib.widgets import Slider
import matplotlib.cm as cm
from matplotlib.colors import SymLogNorm, LogNorm
from lib2to3.tests.data.infinite_recursion import dsa_method

from utils import *

{}
### Function Definitions







def curr_a_b(a, b, lam):
    '''
    a and b are tuples representing locations of 
    two somas.  compute the current at soma b due to 
    injecting unit-current I0 at soma a
    (use manhattan distance)
    '''
    dist_ab = abs(a[0] - b[0]) + abs(a[1] - b[1])
    return np.exp(-lam * dist_ab)
    
def fill_node(ua_somas, ua_encoders, encoders, somas, n, curr_node=None):
    '''
    Recursive helper for filling a given node based on its dot-distance to parent.
    Given a parent node with assigned encoding vector e_parent, find the encoding
    vector e_j with the minimum angle to e_parent and assign it to 
    curr_node. If the nearest node has an angle greater than 45 deg to the parent_node, assign it to a new cluster
    '''
    
    
    if curr_node == None or curr_node not in ua_somas:
        return
    else:
        assert(curr_node in ua_somas), ("Current node %i is not in list of somas available for assignment"%(curr_node),  ua_somas)
        neighbors = neighborhood(curr_node, n)
        
        neighbor_vecs = [encoders[n] for n in neighbors if n in somas.values()]
        neighborhood_average = sum(neighbor_vecs)/len(neighbor_vecs)
        
        ua_encoders.sort(key=lambda ej: angle(neighborhood_average, np.asarray(ej)))        
        nearest_encoder = ua_encoders.pop(0)
        somas[tuple(nearest_encoder)] = curr_node
        encoders[curr_node] = nearest_encoder
        ua_somas.remove(curr_node)  
        
        return

def neighborhood(node, n):
    '''
    given a soma number (node), return a list of all adjacent nodes.
    adjacent nodes are either left, right, up, or down in a 2d grid, which is vectorized 
    with column major order (first index changes fastest) 
    '''
    assert(node < n**2 and node >= 0), "Node should be between %i and %i but was %i"%(0, n**2-1, node)
    
    row, col = idx_to_grid(node, n, 'F') 
    
    neighbors = []
    
    if row > 0:
        neighbors.append(node - n)
    if row < n-1:
        neighbors.append(node + n)
    if col > 0:
        neighbors.append(node - 1)
    if col < n-1:
        neighbors.append(node + 1)
    
    if row > 0 and col > 0:
        neighbors.append(node - n - 1)
    if row > 0 and col < n-1:
        neighbors.append(node - n + 1)
    if row < n-1 and col > 0:
        neighbors.append(node + n - 1)
    if row < n-1 and col < n-1:
        neighbors.append(node + n + 1)
    
    return neighbors    

def get_uniform_unit_vectors(N, dim):
    '''
    Return a dim x N matrix, containing N unit vectors sampled from a uniform distribution
    about the the dim-dimensional hypersphere.
    This is done via choosing all vector elements from a normal distibution & normalizing their length 
    '''
    samples = np.zeros((dim, N)) 
    
    for i in range(N):
        samples[:,i] = np.random.normal(loc = 0, scale = 1, size =((dim)))
        samples[:,i] = samples[:,i] / (samples[:,i].T @ samples[:,i])**.5
    
    return samples


# get nearest ua soma
# fill soma with average of its neighborhood

def get_soma_encoder_assignments(n, d, randomize=False):
    
    E = get_uniform_unit_vectors(n**2, d)
    
    ua_somas = [i for i in range(n**2)]
    ua_encoders = [E[:,i] for i in range(n**2)]
    encoders = {}
    somas = {} 
   
    if randomize:
        while len(ua_somas) > 0:
            
            soma = ua_somas.pop()
            random.shuffle(ua_encoders)
            rand_enc = tuple(ua_encoders.pop())
            encoders[soma] = rand_enc
            somas[rand_enc] = soma
                
        return somas, encoders
   
   
   
    
    
    middle_node = (n**2)//2
    ua_somas.remove(middle_node)
    enc_vec = ua_encoders.pop(0)
    somas[tuple(enc_vec)] = middle_node
    encoders[middle_node] = enc_vec
    prev_node = middle_node
    mid_row, mid_col = idx_to_grid(middle_node, n, order='F')
    #sort somas by distance from origin
    ua_somas.sort(key=lambda node: ((idx_to_grid(node, n, order='F')[0]-mid_row)**2 + (idx_to_grid(node, n, order='F')[1]-mid_col)**2)**.5)
    while len(ua_somas) >  0:
        
        curr_node = ua_somas.pop(0) 
        
        neighbors = neighborhood(curr_node, n)
        
        neighbor_vecs = [encoders[n] for n in neighbors if n in somas.values()]
        if len(neighbor_vecs) > 0:
            neighborhood_average = sum(neighbor_vecs)/len(neighbor_vecs)
        else:
            neighborhood_average = encoders[prev_node]
        
        ua_encoders.sort(key=lambda ej: angle(neighborhood_average, np.asarray(ej)))        
        nearest_encoder = ua_encoders.pop(0)
        somas[tuple(nearest_encoder)] = curr_node
        encoders[curr_node] = nearest_encoder
        prev_node = curr_node
    
    assert(len(ua_somas) == 0), "Attempted to Return without Assigning all encoders/somas"
    return somas, encoders


        
# def get_soma_encoder_assignments(n, d, randomize=False):
#     E = get_uniform_unit_vectors(n**2, d)
#     
#     ua_somas = [i for i in range(n**2)]
#     ua_encoders = [E[:,i] for i in range(n**2)]
#     encoders = {}
#     somas = {} 
#    
#     if randomize:
#         while len(ua_somas) > 0:
#             
#             soma = ua_somas.pop()
#             random.shuffle(ua_encoders)
#             rand_enc = tuple(ua_encoders.pop())
#             encoders[soma] = rand_enc
#             somas[rand_enc] = soma
#                 
#         return somas, encoders
#    
#    
#    
#     first_node = ua_somas.pop(0)
#     enc_vec = ua_encoders.pop(0)
#     somas[tuple(enc_vec)] = first_node
#     encoders[first_node] = enc_vec
#  
#     for node in neighborhood(first_node, n):   
#         fill_node(ua_somas, ua_encoders, encoders, somas, n, curr_node=node, parent_node=first_node)
#     
#     assert(len(ua_somas) == 0), "Attempted to Return without Assigning all encoders/somas"
#     return somas, encoders

def project_assigned_encoders_onto_dims(somas, encs, n, d):
    '''
    given a soma, enc set of assignments,
    display the projections o these assignments 
    onto each dimension d of the encoded vector space
    ''' 
    
    for i in range(d):
        test_grid = np.zeros((n,n))
        e_hat_i = np.eye(d)[:,i]
    
        for soma in [i for i in somas.values()]:
            row, col = idx_to_grid(soma, n, 'F')
            e_b = encs[soma]
            test_grid[row, col] = dot_product(e_hat_i, e_b) 
        
        plt.figure()
        plt.imshow(test_grid)
        plt.colorbar()
        plt.title('Projection of Soma Assignment Against dim %i'%(i + 1))

def LIF_rate(current):
    ''' Apply LIF rate function to current'''
    TAU_M = .005
    TAU_REF = .001

    def lif(j):
        denom = (TAU_REF- TAU_M * np.log(1-1./(j+1)))
        num = 1
        if math.isnan(denom) or math.isinf(denom):
            return 0
        else:
            return num/denom
    #lif = lambda j : 1./(TAU_REF- TAU_M * np.log(1-1./(j+1)))
    return [lif(curr) for curr in list_check(current)] 
      
def get_diffuser_response_matrix(soma_locs, loc_current_func):
    '''
    given physical soma locations on grid, and a function that
    computes diffusion currents between two locations,
    compute the diffuser response matrix: N x N matrix R
    such that R[i,j] = current diffused to soma i if unit current injected into soma j
    loc_current_func(a, b) takes locations as tuples a & b and returns the current
    flowing to b if unit current injected at a. 
    '''
    N = len(soma_locs)
    gamma = np.zeros((N,N))
    for i in range(N): 
        gamma[i,:] = np.asarray([loc_current_func(soma_locs[j], soma_locs[i]) for j in range(N)])
    
    return gamma
        
def encode_stimulus_direct(somas, encoders, xs, alphas, betas, rate_func):
    ''' 
    given soma/encoder assignments and a stimulus x,
    encode stimulus into the somas directly for each soma by
    computing current.  return the rates generated
    from rate_func(current). 
    '''
    assert(len(list_check(xs)[0]) == len(encoders[0])), "Desired Stimulus has dimension %i but encoders have dimension %i"%(len(list_check(xs)[0]), len(encoders[0]))
    rates = np.zeros((  len(list_check(xs)) , len(somas)  ))
    currents = np.zeros(rates.shape)
    for i, x in enumerate(list_check(xs)):
        for soma in somas.values():
            e_dot_x = dot_product(encoders[soma], x)
            if e_dot_x > 0:
                currents[i,soma] = e_dot_x * alphas[soma] + betas[soma]
                rates[i,soma] = rate_func(currents[i,soma])[0]
    
    return rates, currents

# to do: 
    # visualize rates encoded vs x
    # convert an n**2 vector to an nxn grid
    # this is 1 frame of an animation 

def get_K_taps(opt_currs, K):
    n = opt_currs.shape[1]
    
    norm_currs = np.zeros(opt_currs.shape)
    for i in range(opt_currs.shape[0]):
        norm_currs[i,:] = opt_currs[i,:] / np.linalg.norm(opt_currs[i,:])
    mean_curr = np.mean(norm_currs, axis=0)
    tap_thresh = np.sort(mean_curr)[-K]
    return [tap for tap in range(len(mean_curr)) if mean_curr[tap] >= tap_thresh]

def diffuse_current(curr, diffuser_response, taps):
    '''
    Given a current n**2 vector and selected taps, diffuse current
    at only the tap points. Return a n**2 vector of diffused currents
    '''
    # mask opt_curr to only tap points nonzero
    masked_curr = np.zeros(curr.shape)
    masked_curr[taps] = curr[taps]    
    return diffuser_response @ masked_curr

def encode_stimulus_diffuser(opt_curr, diffuser_response, gains, offsets, taps, rate_func):
    '''
    Inject stimulus having optimal currents [opt_curr] at tap points [taps] using [diffuser_response]
    matrix to compute current at each soma. scale each current by gain,
    add offset and return the rate as determined by rate_func(current) 
    '''
    diffused_curr = diffuse_current(opt_curr, diffuser_response, taps)    
    diffused_curr = np.asarray([diffused_curr[i] *gains[i] + offsets[i] for i in range(len(gains))])
    diffused_curr[diffused_curr < 0] = .0000000001 
    diffused_rates = rate_func(diffused_curr)
    
    return np.asarray(diffused_rates), diffused_curr
      
def get_decoder(firing_rates, xtrain): 
    '''
     generate an optimal decoder from given firing rates and training data.
     firing rates is an |xtrain| x N numpy matrix of N neurons response to |xtrain| data points
     xtrain is a |xtrain| length list of d-vectors corresponding to the sample data points  
     '''
    data_shape = np.asarray(list_check(xtrain)[0]).shape
    assert(len(data_shape)==1),("xtrain should be a d-vector but had shape: ",data_shape)
    N = firing_rates.shape[1]
    assert(firing_rates.shape[0] == len(xtrain)),(" Dimension mismatch between firing rates: %i x %i and training data: %i"%(firing_rates.shape[0], firing_rates.shape[1], len(xtrain)))
    
    B = np.zeros((d, len(xtrain)))
    for i in range(len(xtrain)):
        B[:,i] = np.asarray(xtrain[i])
    return np.linalg.pinv(firing_rates) @ B.T   
    
def get_offsets(opt_currs, taps, R):
    '''
    Given a  |num_train_points| x n**2 matrix of (optimal) target currents,
    a list of possible tap points, and n**2 x n**2 diffuser response matrix R,
    compute the optimal offsets to adjust diffused current as close as possible
    to the optimal current. Currenly optimal offsets are averaged over the |num_train_points|
    '''
    diffused_currs = np.zeros(opt_currs.shape)
    for i in range(diffused_currs.shape[0]):
        diffused_currs[i,:] = diffuse_current(opt_currs[i,:], R, taps)
    return np.mean(opt_currs- diffused_currs,axis=0)
    
def get_gains(opt_currs, taps, R):
    '''
    Given a  |num_train_points| x n**2 matrix of (optimal) target currents,
    a list of possible tap points, and n**2 x n**2 diffuser response matrix R,
    and offsets given by get_offsets(),   
    compute the optimal gains to adjust diffused current as close as possible
    to the optimal current. Currenly optimal gains are averaged over the |num_train_points|
    '''
    diffused_currs = np.zeros(opt_currs.shape)
    for i in range(diffused_currs.shape[0]):
        diffused_currs[i,:] = diffuse_current(opt_currs[i,:], R, taps)
        
    thresh = .1
    
    gains = np.ones(diffused_currs.shape)
    for i in range(opt_currs.shape[0]):
        for j in range(opt_currs.shape[1]):
            if diffused_currs[i,j] >  thresh and opt_currs[i,j] > thresh and opt_currs[i,j] > 0:
                gains[i,j] = opt_currs[i,j] / diffused_currs[i,j]
            
    return np.mean(gains,axis=0)

    #return np.mean(np.divide(opt_currs, diffused_currs),axis=0)
    
def plot_encode_decode_interactible(d, n, enc_rates, diff_rates, D, xs, normalize=False):     
    fig, _ = plt.subplots(1,2, figsize = (16, 9))
    plt.subplot(131)
    
    plt.suptitle('%i-dimensional Encode/Decode in %i x %i Soma Grid'%(d, n, n))
    
    plt.title('Optimal Current Injection')
    opt_img = plt.imshow(np.reshape(enc_rates[25,:], (n,n), order='F'),aspect='equal')
    cbar = plt.colorbar()
    cbar.set_label('Firing Rate (Hz)')
    ax_x = plt.axes([.2, .05, .65, .03], facecolor='lightgoldenrodyellow')
    plt.subplots_adjust(bottom = .3)
    
    
    plt.subplot(132)
    plt.title('Current Injection via K=%i tap points'%len(taps))
    diff_img = plt.imshow(np.reshape(diff_rates, (n,n), order='F'), aspect='equal')
    cbar = plt.colorbar()
    cbar.set_label('Firing Rate (Hz)')
    
    x_plot = plt.subplot(133)
    x_actual = plt.scatter(xs_test[25][0], xs_test[25][1], label='Encoded Stimulus')
    x_hat = enc_rates[25,:] @ D
    diff_x_hat = diff_rates @ D  
    #x_decoded = plt.scatter(x_hat[0], x_hat[1],marker='x', label='Optimal Decode')
    diff_x_decoded = plt.scatter(diff_x_hat[0], diff_x_hat[1],marker='x', label='Diffused-Taps Decoded Estimate')
    
    plt.legend()
    x_plot.set_aspect('equal')
    plt.xlim([-1.2, 1.2])
    plt.ylim([-1.2, 1.2])
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    
    slider_x = Slider(ax_x, "Stimulus Number", valmin=int(0), valmax=int(len(xs)-1), valinit=int(0), valstep = 1.0)
    
    def update(val):
        opt_img.set_data(np.reshape(enc_rates[int(val),:], (n,n), order='F'))
        
        
        diff_rates, _ = encode_stimulus_diffuser(opt_currs[int(val),:], R, gains, offsets, taps, LIF_rate)
        diff_img.set_data(np.reshape(diff_rates, (n,n), order='F'))
        
        x_hat = enc_rates[int(val),:] @ D  
        #x_decoded.set_offsets([x_hat[0], x_hat[1]])
        
        
        diff_x_hat = diff_rates @ D
        if normalize:
            diff_x_hat /= np.linalg.norm(diff_x_hat)  
        diff_x_decoded.set_offsets([diff_x_hat[0], diff_x_hat[1]])
        x_actual.set_offsets([xs[int(val)][0], xs[int(val)][1]])
        fig.canvas.draw_idle()
    
    slider_x.on_changed(update)
    return fig, slider_x

def plot_optimal_current_injections(d, n, enc_rates, diff_rates, D, xs, normalize=False): 
    fig, _ = plt.subplots(1,2, figsize = (16, 9))
    plt.subplot(121)
    plt.suptitle('%i-dimensional Encode/Decode in %i x %i Soma Grid'%(d, n, n))
    
    plt.title('Optimal Current Injection')
    opt_img = plt.imshow(np.reshape(enc_rates[25,:], (n,n), order='F'), norm=LogNorm(), aspect='equal',cmap='jet')
    cbar = plt.colorbar()
    cbar.set_label('Firing Rate (Hz)')
    ax_x = plt.axes([.2, .05, .65, .03], facecolor='lightgoldenrodyellow')
    plt.subplots_adjust(bottom = .3)
    current_cmap = cm.get_cmap()
    current_cmap.set_bad(color=current_cmap(0))
    
    
    
    plt.subplot(122)
    x_actual = plt.scatter(xs_test[25][0], xs_test[25][1], label='Encoded Stimulus')
    x_hat = enc_rates[25,:] @ D  
    x_decoded = plt.scatter(x_hat[0], x_hat[1],marker='x', label='Optimal Decode')
    
    
    plt.legend()
    
    plt.xlim([-1.2, 1.2])
    plt.ylim([-1.2, 1.2])
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    
    slider_x = Slider(ax_x, "Stimulus Number", valmin=int(0), valmax=int(len(xs)-1), valinit=int(0), valstep = 1.0)
    
    def update(val):
        opt_img.set_data(np.reshape(enc_rates[int(val),:], (n,n), order='F'))
        
        x_hat = enc_rates[int(val),:] @ D  
        x_decoded.set_offsets([x_hat[0], x_hat[1]])
        x_actual.set_offsets([xs[int(val)][0], xs[int(val)][1]])
        fig.canvas.draw_idle()
    
    slider_x.on_changed(update)
    return fig, slider_x

def encoder_mean(xs):
    ys = list_check(xs)
    if len(ys) == 1 or len(ys) == 0:
            return xs
    else:
        tot = np.asarray(ys[0])
        for j in range(1, len(ys)):
            tot += ys[j]
        return tuple(np.asarray(tot) / len(ys)) 

def cluster_somas_by_angle(encs, somas, K):
    '''
     given a dictionary containing encs = {soma : encoder} for soma in range(N), 
     assign each soma to a cluster via k-means, where the distance measure is the angle between vectors  
     '''
   
    
    num_clusters = K  #number of clusters is 2 * num dimensions   

    def dist_func(x,y):
            return(angle(x,y))

    return k_means_clustering(num_clusters, [tuple(e) for e in encs.values()], data_tags = somas, distance_measure = dist_func, data_mean = encoder_mean)
    
def get_grid_centroids(clusters, somas, soma_locs):
    '''
    Given clustered somas physically located at soma-locs on the grid,
    return the (row,col) corresponding to the physical centroid of each cluster
    ''' 
    
    centroids  = {}   
    for c in clusters: #for each cluster 
        cent_x = 0
        cent_y = 0
        c_size = 0
        
        for e in clusters[c][0]: 
            soma = somas[e] # compute the physical centroid of that cluster
            row,col = idx_to_grid(soma, n, order='F')
            cent_x += row
            cent_y += col
            c_size += 1
        centroids[c] = ( cent_x//c_size, cent_y//c_size)

    return centroids
    
            
            
            
            
            
            # get soma whos soma loc is nearest physical centroid
    
def rmse_matrix(X, Y):
    '''
    Given two |num_data_points| x M matrices,
    compute the average rmse accross |num_train_points| 
    '''
    rmse = lambda x, y: np.sqrt((x-y).T @ (x-y))
    return np.mean([rmse(X[i,:],Y[i,:]) for i in range(X.shape[0])])         

def cluster_binary(encs):
    '''
     cluster vectors based on their binary sign (pos=0 or neg=1)
     a vector's cluster number is then its binary to decimal converstion.
     e.g. vec = [-1, 0, 1] = [neg, pos, pos] = [1, 0, 0] = 4 + 0 + 0 = cluster 4
     '''
    clusters = {}
            
    for e in encs.values():
        c = get_quadrant(e)
        
        if c not in clusters.keys():
            clusters[c] = ([], None) 
        
        clusters[c][0].append(tuple(e))
    
    for c in clusters.keys():
        clusters[c] = (clusters[c][0], encoder_mean(clusters[c][0]))
         
    return clusters

def plot_soma_clusters(n, somas, encs, num_clusters, binary_clustering=False):
    
    def get_quadrant(vec):
        ''' 
        given a d-vectors, use the sign of each component to return a quadrant number:
        quadrant are determiend by a binary code 0 is positive, 1 is negative so that 
        quadrant[0] = [0, 0, ..., 0],
        quadrant[1] = [0, 0, ..., 0, 1],
        ....
        quadrant[2**d] = [1,1,...,1]
        '''
        
        signs = np.sign(list_check(vec))
        signs[signs < 0] = 0
        return sum([2**i for i in range(len(signs)) if signs[i] > 0])
        
    clusters= cluster_binary(encs) if binary_clustering else cluster_somas_by_angle(encs, num_clusters) 
       
    cl_vals = np.zeros((n,n))

    plt.subplots(1,2)
    ax = plt.subplot(1,2,2)
    for c in clusters:
        
        if len(encs[0])==3:
            ax.scatter(x=clusters[c][1][0],y=clusters[c][1][1],s=50, c=[cm.get_cmap('jet')(c/(len(clusters)-1))])
            ex = [clusters[c][0][i][0] for i in range(len(clusters[c][0]))]
            ey = [clusters[c][0][i][1] for i in range(len(clusters[c][0]))]
            #ez = [clusters[c][0][i][2] for i in range(len(clusters[c][0]))]
            ax.scatter(x=ex,y=ey,s=20, c=[cm.get_cmap('jet')(c/(len(clusters)-1))])
                
        else:
            plt.scatter(clusters[c][1][0],clusters[c][1][1],c=[cm.get_cmap('jet')(c/(len(clusters)-1))])
            plt.title('Cluster Centroids, 1st Two Dimensions')
            plt.xlabel('Dim 1')
            plt.ylabel('Dim 2')
           
        es = clusters[c][0]
        for e in es: 
            soma = somas[e]
            row,col = idx_to_grid(soma, n, order='C')
            cl_vals[row,col] = c
             
    plt.subplot(1,2,1)
    plt.imshow(cl_vals, cmap='jet')
    plt.title('Soma Grid Cluster Assignments')
    plt.colorbar()

plt.rcParams['font.weight'] = 'bold'
plt.rcParams['figure.figsize'] = (16,16)
    
{}
### Parameters
n = 64 # number of somas in neuron pool ( n^2)
N = n**2
d = 3  # number of input dimensions 
E = np.zeros((d, N))
lam = 1 # diffusion decay rate (units of 1/L where L is the distance between two adjacent somas) 
soma_locs =[(i * lam, j * lam) for i in range(1, n+1) for j in range(1, n+1)]  # the physical location of each soma on the grid (vectorized column-major order) 
K = 2*d # number of desired tap points
num_clusters = 2 ** d
num_sample_pts = 50

   
#get correlation for random assignments 
# corrs = []
# for i in range(100):
#     somas, encs = get_soma_encoder_assignments(n, d, randomize=True)
#     A = []
#     D = []
#     for si in somas.values():
#         for sj in somas.values():
#             if si != sj:
#                 A.append(angle(encs[si], encs[sj]))
#                 rowi, coli = idx_to_grid(si, n, order='F')
#                 rowj, colj = idx_to_grid(sj, n, order='F')
#                 D.append( ((rowi-rowj)**2 + (coli-colj)**2)**.5)
#        
#     corrs.append(np.corrcoef(A,D)[0,1])
#        
# rand_std, rand_mean = (np.std(corrs), np.mean(corrs))
# print(rand_std, rand_mean)
#   
# 
# ##plot greedy assignments for d = 2, 3, 4, 8
#   
# ds = [2, 3, 4, 8]
# corr_coeffs = []
# fig, axs = plt.subplots(3,2, figsize=(16,16))
# plt.suptitle('Preferred Directions for Greedy Soma Assignments')
# for i,d in enumerate(ds):
#     somas, encs = get_soma_encoder_assignments(n, d)
#        
#     plt.subplot(3,2,i+1)
#     plt.title('d = %i'%d)
#        
#     angs = np.zeros((n,n))
#     for soma in somas.values():
#         row,col = idx_to_grid(soma, n, order='F')
#         angs[row,col] = angle(encs[n**2 // 2], encs[soma])
#     
#     plt.imshow(angs,cmap='jet',vmax=pi, vmin=0)
#     #plt.axis('equal')
#        
# plt.subplots_adjust( wspace=.05)
# plt.subplot(3,2,6)
# plt.axis('off')
# plt.subplot(3,2,5)
# plt.axis('off')
# cbar = plt.colorbar(ax=axs, ticks=[pi, 3/4 * pi, pi/2, pi/4, 0], orientation='Horizontal')
# cbar.ax.set_xticklabels(['$\pi$', r'$\frac{3}{4}\pi$',r'$\frac{\pi}{2}$', r'$\frac{\pi}{4}$', '0'])
# cbar.set_label('Angle Relative to First Soma')
# plt.savefig('fig1_greedy_pref_dirs.png',bbox_inches='tight')
#  
# #now compute the anlge between each pari of somas, and compute hte distance between each pair of osmas
# n_trials = 10
#   
# ds = range(2,10)
# corrs = np.zeros((n_trials, len(ds)))
# for i in range(n_trials):
#     for j,d in enumerate(ds):
#         if i != j:
#             somas, encs = get_soma_encoder_assignments(n, d)
#       
#             A = []
#             D = []
#             for si in somas.values():
#                 for sj in somas.values():
#                     A.append(angle(encs[si], encs[sj]))
#                     rowi, coli = idx_to_grid(si, n, order='F')
#                     rowj, colj = idx_to_grid(sj, n, order='F')
#                     D.append( ((rowi-rowj)**2 + (coli-colj)**2)**.5)
#               
#         corrs[i,j] = np.corrcoef(A,D)[0,1]
#   
# plt.figure()
# plt.bar(x=ds, height=np.mean(corrs, axis = 0), yerr=np.std(corrs, axis=0))
# fill_bounds = [d for d in ds]
# fill_bounds[0] = 0
# fill_bounds[-1]+= 1.5
# plt.fill_between(fill_bounds, rand_mean-rand_std, rand_mean+rand_std,color='k', linestyle='-',alpha = .3, label='Random Encoder-Soma Assignment for 100 Trials',zorder=2)
# #plt.axhline(y=rand_mean,xmin=ds[-1], xmax=ds[0], alpha=1, color='k',linestyle='-')
# plt.xlabel('Dimension')
# plt.legend()
# plt.ylabel(r'$\rho$ (corr. coeff.)')
# plt.title('Pairwise Angle - Distance Correlation in %i x %i Grid Assignment, num_trials = %i'%(n,n,n_trials))
#  
# #ds = [d for d in ds]
# plt.xlim([min(ds)-.5, max(ds)+.5])


# n_trials = 1
# rmses_opts = []
# rmses_ones = []
# for i in range(n_trials):
#     print('Trial %i/%i'%(i+1, n_trials))
#     
#     ## TRAINING DECODER
#     # Optimal Encoding & Soma Assignments
# somas, encs = get_soma_encoder_assignments(n, d)
#     
#     quad1 = [1, 1]
#     quad3 = [1,-1]
#     quad2 = [-1, 1]
#     quad4 = [-1,-1]
#     quads = np.zeros((n,n))
#     # go through each soma
#     
#     for soma in range(N):
#         
#         row,col = idx_to_grid(soma, n, order='F')        
#         if np.sign(encs[soma][0]) < 0 and encs[soma][1] < 0 and encs[soma][2] > 0:
#             quads[row,col] = 1
#         if np.sign(encs[soma][0]) > 0 and encs[soma][1] < 0 and encs[soma][2] > 0:
#             quads[row,col] = 2
#         if np.sign(encs[soma][0]) > 0 and encs[soma][1] > 0 and encs[soma][2] > 0:
#             quads[row,col] = 3
#         if np.sign(encs[soma][0]) < 0 and encs[soma][1] > 0 and encs[soma][2] > 0:
#             quads[row,col] = 4
#             
#         if np.sign(encs[soma][0]) < 0 and encs[soma][1] < 0 and encs[soma][2] < 0:
#             quads[row,col] = 5
#         if np.sign(encs[soma][0]) > 0 and encs[soma][1] < 0 and encs[soma][2] < 0:
#             quads[row,col] = 6
#         if np.sign(encs[soma][0]) > 0 and encs[soma][1] > 0 and encs[soma][2] < 0:
#             quads[row,col] = 7
#         if np.sign(encs[soma][0]) < 0 and encs[soma][1] > 0 and encs[soma][2] < 0:
#             quads[row,col] = 8    
#             
#             
#         # 
#     plt.imshow(quads,cmap='jet')
#     plt.colorbar()
#     plt.show()
#     
#     # clusters = cluster_somas_by_angle(encs, num_clusters)
#     # 

# xs = np.random.normal(loc=0, scale=1, size=(num_sample_pts, d))
# xs_test = np.sort(np.random.normal(loc=0, scale=1, size=(num_sample_pts, d)),)
# #     
# for i in range(num_sample_pts):
#     xs[i,:] /= np.linalg.norm(xs[i,:])
#     xs_test[i,:] /= np.linalg.norm(xs_test[i,:])
#  
# xslist = [xs[i,:] for i in range(num_sample_pts)]
# x0 = xslist[0]
# xs_testlist = [xs_test[i,:] for i in range(num_sample_pts)]
#  
# xslist.sort(key=lambda vec: angle(vec, xs[0,:]))
# xs_testlist.sort(key=lambda vec: angle(vec, xs_test[0,:]))
#  
# #xslist = [[np.cos(2*np.pi * i/num_sample_pts), np.sin(2*np.pi*i/num_sample_pts)] for i in range(num_sample_pts)]
#  
# for i in range(num_sample_pts):
#     xs[i,:] = np.asarray(xslist[i])
#     xs_test[i,:] = np.asarray(xs_testlist[i])
# 
#     
# alphas = [1 for soma in somas.values()]
# betas = [0 for soma in somas.values()]
# enc_rates, opt_currs  = encode_stimulus_direct(somas, encs, xs, alphas, betas, LIF_rate)
 

# Diffusive Encoding Through Tap Points
# taps = random.choices(list(somas.values()), k=K) #alternatively pick K random indices out of [1,...N] (works better - encoders are uniformly distributed)
# taps = [row * n  + col for row,col in get_grid_centroids(clusters, somas, soma_locs).values()]
# R = get_diffuser_response_matrix(soma_locs, lambda a,b : curr_a_b(a, b, lam))

#gains = get_gains(opt_currs, taps, R)
 
# amplified_curr = np.zeros(opt_currs.shape)
# for j in range(opt_currs.shape[1]):
#     amplified_curr[:,j] *= gains[j]
 
#offsets = get_offsets(amplified_curr, taps, R)
#ones = np.ones(offsets.shape)
 
 
#diff_rates_opt, diff_currs_opt = encode_stimulus_diffuser(opt_currs[25,:], R, gains, offsets, taps, LIF_rate)
#diff_rates, diff_currs = encode_stimulus_diffuser(opt_currs[25,:], R, ones, 0*ones, taps, LIF_rate)
 
#mDiff_rates_opt = np.zeros(opt_currs.shape)
#mDiff_rates = np.zeros(opt_currs.shape)
 
# for i in range(num_sample_pts):
#     mDiff_rates_opt[i,:] =  encode_stimulus_diffuser(opt_currs[i,:], R, gains, offsets, taps, LIF_rate)[0]
#     mDiff_rates[i,:] =  encode_stimulus_diffuser(opt_currs[i,:], R, ones, 0*ones, taps, LIF_rate)[0]
#  
# D_perf = get_decoder(enc_rates, xs)
# D_opts = get_decoder(mDiff_rates_opt, xs)
# D_ones = get_decoder(mDiff_rates, xs)
#  
 
#GIVEN DECODERS & Offsets/Gains, NOW TEST
 
#enc_rates, opt_currs  = encode_stimulus_direct(somas, encs, xs_test, alphas, betas, LIF_rate)
#diff_rates_opt, diff_currs_opt = encode_stimulus_diffuser(opt_currs[25,:], R, gains, offsets, taps, LIF_rate)
#diff_rates, diff_currs = encode_stimulus_diffuser(opt_currs[25,:], R, ones, 0*ones, taps, LIF_rate)
 
#fig, _ = plot_encode_decode_interactible(d, n, enc_rates, diff_rates_opt, D_opts, xs_test)
#enc_rates, opt_currs  = encode_stimulus_direct(somas, encs, xs, alphas, betas, LIF_rate)
#fig, _ = plot_optimal_current_injections(d, n, enc_rates, enc_rates, D_perf, xs)
# somas, encs = get_soma_encoder_assignments(n, d)
# plot_soma_clusters(n, somas, encs, num_clusters, binary_clustering=False)
# plot_soma_clusters(n, somas, encs, num_clusters, binary_clustering=True)
# 
# plt.show()

# numpy_xs = np.zeros((num_sample_pts, d))
# X_hat_opts = np.zeros(numpy_xs.shape)
# X_hat_ones = np.zeros(numpy_xs.shape)
# for j in range(numpy_xs.shape[0]):
#     numpy_xs[j,:] = xs_test[j]
#     X_hat_opts[j,:] =  diff_rates_opt @ D_opts
#     X_hat_ones[j,:] =  diff_rates @ D_ones
#  
# rmses_opts.append(rmse_matrix(X_hat_opts, numpy_xs))
# rmses_ones.append(rmse_matrix(X_hat_ones, numpy_xs))

# print('Average RMSE over %i trials for Optimized Gain/Offset: '%n_trials, np.mean(rmses_opts))
# print('Average RMSE over %i trials for Unit Gain Zero Offset: '%n_trials, np.mean(rmses_ones))
#      
# # what have we done so far? 
#     # Generated N = n**2 encoding vectors that uniformly tile the d-hypersphere
#     # Assigned these encoding vectors to somas on our diffusive grid, clustering vectors based on their angle to each other
#     # Computed the smallest injection of current into each soma that best encodes an arbitrary d-vector into the grid of somas
#     # TODO:
#     
#     # Fix Recursive Greedy Algo: keeps exceeding recursion depth 
#     
#     # Then discuss tap points
#     
#     # STILL NEED TO Figure Out Tap Point Selection: Choose based on clusters?
#     # double check rmse metrics
#     # given a tap point selection, computed the "best" (prove this?) gains/offsets for each soma
#     # explore relationship between RMSE, diffusion lambda, K num tap points, n num neurons, and d dimensionality: vary ratios/ K:n, d:n, lambda, 
# 
#   
# c
# 
# 
# 
# plt.show()

