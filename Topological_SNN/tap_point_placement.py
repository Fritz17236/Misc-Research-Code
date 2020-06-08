'''
Created on Jun 5, 2020

@author: fritz
'''



import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import nnls
import random
from matplotlib.widgets import Slider
import matplotlib.gridspec as gridspec
 
{}
### Function Definitions
def iterable_dot(a, b):
    ''' Dot product of n-tuples'''
    n = len(a)
    return sum([a[i] * b[i] for i in range(n)])

def dot_product_distance(ei, ej):
    '''
     given two unit vectors ei, ej, return the distance between them as defined 
     by D(ei, ej) = 1 - dot(ei, ej) 
     '''
    
    return  1 - np.asarray(ei).T @ np.asarray(ej)

def curr_a_b(a, b, lam):
    '''
    a and b are tuples representing locations of 
    two somas.  compute the current at soma b due to 
    injecting unit-current I0 at soma a
    (use manhattan distance)
    '''
    dist_ab = abs(a[0] - b[0]) + abs(a[1] - b[1])
    return np.exp(-lam * dist_ab)
    
def fill_node(ua_somas, ua_encoders, encoders, somas, n, curr_node=None, parent_node=None):
    '''
    Recursive helper for filling a given node based on its dot-distance to parent.
    Given a parent node with assigned encoding vector e_parent, find the encoding
    vector e_j with the minimum dot_product_distance to e_parent and assign it to 
    curr_node   
    '''
    if curr_node == None or curr_node not in ua_somas:
        return
    else:
        assert(curr_node in ua_somas), ("Current node %i is not in list of somas available for assignment"%(curr_node),  ua_somas)
        ua_encoders.sort(key=lambda ej: dot_product_distance(encoders[parent_node], np.asarray(ej)))        
        nearest_encoder = ua_encoders.pop(0)
        somas[tuple(nearest_encoder)] = curr_node
        encoders[curr_node] = nearest_encoder
        ua_somas.remove(curr_node) 
        neighbors = neighborhood(curr_node, n)
        random.shuffle(neighbors)
        for node in neighbors:
            if node in ua_somas:
                fill_node(ua_somas, ua_encoders, encoders, somas, n, curr_node=node, parent_node=curr_node) 
        
        return

def neighborhood(node, n):
    '''
    given a soma number (node), return a list of all adjacent nodes.
    adjacent nodes are either left, right, up, or down in a 2d grid, which is vectorized 
    with column major order (first index changes fastest) 
    '''
    assert(node < n**2 and node >= 0), "Node should be between %i and %i but was %i"%(0, n**2-1, node)
    
    row, col = idxToGrid(node, n, 'F') 
    
    neighbors = []
    
    if row > 0:
        neighbors.append(node - n)
    if row < n-1:
        neighbors.append(node + n)
    if col > 0:
        neighbors.append(node - 1)
    if col < n-1:
        neighbors.append(node + 1)
    
    return neighbors    

def idxToGrid(idx, n, order):
    '''
    given an index i which is mapped to a vectorized nxn grid, 
    return row, col such that grid[row][col] = vectorized_grid[idx].
    If order = F (fortran ordering) it is vectorized in column major order, 
    otherwise if order is C (C language) it is vectorized in row major order. 
    '''
    if order.upper() == 'F': 
        col = idx % n
        row = idx // n
    elif order.upper() == 'C':
        col = idx // n
        row = idx % n
    else:
        assert(False), " Order must be either F (column major), or C (row major), but was %s"%order 
    
    return row, col 
    
def get_uniform_unit_vectors(N, dim):
    '''
    Return a dim x N matrix, containing N unit vectors evenly sampled across the dim-dimensional hypersphere.
    This is done via choosing all vector elements from a normal distibution & normalizing their length 
    '''
    samples = np.zeros((dim, N)) 
    for i in range(N):
        
        samples[:,i] = np.random.normal(loc = 0, scale = 1, size =((d,)))
        samples[:,i] = samples[:,i] / (samples[:,i].T @ samples[:,i])**.5
    
    return samples
        
def get_soma_encoder_assignments(n, d):
    E = get_uniform_unit_vectors(n**2, d)
    
    ua_somas = [i for i in range(n**2)]
    ua_encoders = [E[:,i] for i in range(n**2)]
    encoders = {} # encoders[encoder]
    somas = {} # somas[encoder]
    
    first_node = ua_somas.pop(0)
    enc_vec = ua_encoders.pop(0)
    somas[tuple(enc_vec)] = first_node
    encoders[first_node] = enc_vec  
    for node in neighborhood(first_node, n):   
        fill_node(ua_somas, ua_encoders, encoders, somas,n,  curr_node=node, parent_node=first_node)
    
    assert(len(ua_somas) == 0), "Attempted to Return without Assigning all encoders/somas"
    return somas, encoders

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
            row, col = idxToGrid(soma, n, 'F')
            e_b = encs[soma]
            test_grid[row, col] = iterable_dot(e_hat_i, e_b) 
        
        plt.figure()
        plt.imshow(test_grid)
        plt.colorbar()
        plt.title('Projection of Soma Assignment Against dim %i'%(i + 1))
        
def list_check(x):
    ''' make object iterable, even if integer'''
    try:
        _ = [y for y in x]
    except:
        x = [x]
    return x

def LIF_rate(current):
    ''' Apply LIF rate function to current'''
    TAU_M = .005
    TAU_REF = .001

    lif = lambda j : 1./(TAU_REF- TAU_M * np.log(1-1./(j+1)))
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
    
    rates = np.zeros((  len(list_check(xs)) , len(somas)  ))
    currents = np.zeros(rates.shape)
    for i, x in enumerate(list_check(xs)):
        for soma in somas.values():
            e_dot_x = iterable_dot(encoders[soma], x)
            if e_dot_x > 0:
                currents[i,soma] = e_dot_x * alphas[soma] + betas[soma]
                rates[i,soma] = rate_func(currents[i,soma])[0]
    
    return rates, currents

# to do: 
    # visualize rates encoded vs x
    # convert an n**2 vector to an nxn grid
    # this is 1 frame of an animation 

def inject_target_current(diffuser_response, target_current, gains, offsets, K):
    '''
    Inject current at K tap points to achieve target_current through diffuser response,
    along with adjusting gains and offsets of each soma. return the diffused current.
    Target current is assumed to be a vector   
    '''
    
    # compute regularized current subject to diffuser constraints
    opt_taps = nnls(diffuser_response, target_current)[0]    
    
    # inject top K currents
    tap_thresh = np.sort(opt_taps)[-K]
    
    
    injections = np.zeros(opt_taps.shape)
    injections[opt_taps >= tap_thresh] = opt_taps[opt_taps >= tap_thresh]  #TODO: make sure no repeats of currents (unlikely with continuous values)
        
    diffused_current = diffuser_response @ injections    
    
    return np.asarray([diffused_current[i]*gains[i] + offsets[i] for i in range(len(diffused_current))])
    
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
    
{}
### Parameters

n = 16  # number of somas in neuron pool ( n^2)
N = n**2
d = 2   # number of dimensions of 
E = np.zeros((d, N))
lam = 1# diffusion decay rate (units of 1/L where L is the distance between two adjacent somas) 
soma_locs =[(i * lam, j * lam) for i in range(1, n+1) for j in range(1, n+1)]  # the physical location of each soma on the grid (vectorized column-major order) 
K = N # number of desired tap points

# Optimal Encoding & Soma Assignments
somas, encs = get_soma_encoder_assignments(n, d)
xs = [(np.cos(2*np.pi*t) , np.sin(2*np.pi*t)) for t in np.linspace(0,1)]
alphas = [1 for soma in somas.values()]
betas = alphas
enc_rates, opt_currs  = encode_stimulus_direct(somas, encs, xs, alphas, betas, LIF_rate)


# Inject current via K tap points
num_train_points = opt_currs.shape[0]

R = get_diffuser_response_matrix(soma_locs, lambda a,b : curr_a_b(a, b, lam))
Rbig = np.concatenate([R for i in range(num_train_points)], axis = 0)


curr_big = np.concatenate([opt_currs[i,:] for i in range(num_train_points)])






diff_currs = inject_target_current(Rbig, curr_big, [1 for i in range(len(curr_big))], [1 for i in range(len(curr_big))], K)[0:N]

physical_rates= LIF_rate(diff_currs)

plt.imshow(np.reshape(physical_rates,(n,n),order='F'))
plt.show()


# Decode & Plotting
D_direct = get_decoder(enc_rates, xs)

rate_slice = np.reshape(enc_rates[25,:], (n,n), order='F')


#project_assigned_encoders_onto_dims(somas, encs, n, d)

fig, ax = plt.subplots(1,2, figsize = (16, 9))
plt.suptitle('2 Dimensional Encoding Directly Into Soma Grid')


plt.subplot(131)
img = plt.imshow(rate_slice,  aspect='equal')
cbar = plt.colorbar()
cbar.set_label('Firing Rate (Hz)')
ax_x = plt.axes([.2, .05, .65, .03], facecolor='lightgoldenrodyellow')
plt.subplots_adjust(bottom = .3)
x_plot = plt.subplot(133)
x_actual = plt.scatter(xs[25][0], xs[25][1], label='Encoded Stimulus')

x_hat = enc_rates[25,:] @ D_direct  
x_decoded = plt.scatter(x_hat[0], x_hat[1],marker='x', label='Decoded Estimate')

plt.legend()
x_plot.set_aspect('equal')
plt.xlim([-1.2, 1.2])
plt.ylim([-1.2, 1.2])
plt.xlabel('Dim 1')
plt.ylabel('Dim 2')




#plt.plot plot x1/x2 decoded x1/x2 actual

# plot rmse over entire set but highlight current with vertical line 



slider_x = Slider(ax_x, "Stimulus Number", valmin=int(0), valmax=int(len(xs)-1), valinit=int(0), valstep = 1.0)


def update(val):
    
    new_slice = np.reshape(enc_rates[int(val),:], (n,n), order='F')
    img.set_data(new_slice)
    x_hat = enc_rates[int(val),:] @ D_direct  
    x_decoded.set_offsets([x_hat[0], x_hat[1]])
    x_actual.set_offsets([xs[int(val)][0], xs[int(val)][1]])
    fig.canvas.draw_idle()

slider_x.on_changed(update)

 

# what have we done so far? 
    # Generated N = n**2 encoding vectors that uniformly tile the d-hypersphere
    # Assigned these encoding vectors to somas on our diffusive grid, clustering vectors based on their angle to each other
    # Computed the smallest injection of current into each soma that best encodes an arbitrary d-vector into the grid of somas
    
# Now we need to choose the smallest number of tap points that produces this current injection:  
    # engineering heuristic:  choose top k current injections to be k tap points. If encoding performance not acceptable increase k
    #                         can also adjust alpha_i / beta_i to best meet the desired current injection 
plt.show()

    
    
    
# how to measure the performance of tap point placement and simulation encoding?
    # generate stimulus 
    # measure current into each soma based on stimulus and tap point assignment 
    # compute firing rate
    # decode firing rate
    # compare decoded to stimulus 
    # repeat with adjustments to alpha_i/beta_i
    # repeat with larger/smaller k 
    # RMSE vs number of tap points: with alpha_i/beta_i adjustments, without 
