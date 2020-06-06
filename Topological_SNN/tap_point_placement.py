'''
Created on Jun 5, 2020

@author: fritz
'''



import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import nnls
import random
 

def tuple_dot(a, b):
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
            test_grid[row, col] = tuple_dot(e_hat_i, e_b) 
        
        plt.figure()
        plt.imshow(test_grid)
        plt.colorbar()
        plt.title('Projection of Soma Assignment Against dim %i'%(i + 1))
        
    

n = 16  # number of somas in neuron pool ( n^2)
N = n**2
d = 2   # number of dimensions of 
E = np.zeros((d, N))
lam = 1  # diffusion decay rate 
soma_locs =[(i * lam, j * lam) for i in range(1, n+1) for j in range(1, n+1)]  # the physical location of each soma on the grid (vectorized column-major order) 
K = 4 # number of desired tap points

somas, encs = get_soma_encoder_assignments(n, d)
project_assigned_encoders_onto_dims(somas, encs, n, d)
 
for i in range(N):
    E[:,i] = encs[i]
     


 
gamma = np.zeros((N,N))
for i in range(N):
    gamma[i,:] = np.asarray([curr_a_b(soma_locs[j], soma_locs[i], lam) for j in range(N)])
         
 
 
#argmin (x) such that 
 
bigGamma = np.concatenate([gamma for j in range(N)], axis = 0)
bigB = np.concatenate([E.T @ (E @ E.T @ E[:,i]) for i in  range(N)], axis = 0)
bigB[bigB < 0] = 0
 
currsnn, residual = nnls(bigGamma, bigB)
print(residual)

currs = np.linalg.pinv(bigGamma) @ bigB
#topK = np.sort(currs)[-K:]
#currs[currs<topK[0]] = 0

 
 
currsnn = np.reshape(currsnn,(n,n),order = 'F')
plt.figure()
plt.imshow(currsnn)
plt.colorbar()
plt.title('Optimal Nonnegative Currents')

currs = np.reshape(currs,(n,n),order = 'F')
plt.figure()
plt.imshow(currs)
plt.colorbar()
plt.title('Optimal Currents')

plt.show()


# what have we done so far? 
    # Generated N = n**2 encoding vectors that uniformly tile the d-hypersphere
    # Assigned these encoding vectors to somas on our diffusive grid, clustering vectors based on their angle to each other
    # Computed the smallest injection of current into each soma that best encodes an arbitrary d-vector into the grid of somas
    
# Now we need to choose the smallest number of tap points that produces this current injection:  
    # engineering heuristic:  choose top k current injections to be k tap points. If encoding performance not acceptable increase k
    #                         can also adjust alpha_i / beta_i to best meet the desired current injection 
    
# how to measure the performance of tap point placement and simulation encoding?
    # generate stimulus 
    # measure current into each soma based on stimulus and tap point assignment 
    # compute firing rate
    # decode firing rate
    # compare decoded to stimulus 
    # repeat with adjustments to alpha_i/beta_i
    # repeat with larger/smaller k 
    # RMSE vs number of tap points: with alpha_i/beta_i adjustments, without 
