'''
Utility Functions 
'''
import random
import math
import numpy as np
import sympy



### Vector Operations ###
# A Vector is defined as an iterable list of scalars # 

def list_check(x):
    '''
    Preprocessing to ensure object is iterable,
    or if integer convert to iterable. 
    '''
    try:
        _ = [y for y in x]
    except:
        x = [x]
    return x

def check_same_length(a,b):
    '''
    Ensure that two given vectors are the same length
    '''
    n = len(list_check(a))
    m = len(list_check(b))
    assert(n==m),'objects to not have equal length (%i vs %i)'%(n, m)
    return n

def k_means_clustering(num_clusters, data, data_tags, distance_measure, data_mean, num_iters = 100, convergence_thresh = 1e-3):
    ''' 
    Given an iterable dataset and a distance measure, cluster
    the dataset into num_clusters clusters via k-means clustering algorithm
    distance_measure takes two data points as arguments and returns a scalare measurement value,
    data_sum takes a list of data points and computes their sum 
    '''
    assert(len(data) > num_clusters), "Dataset should have more elements (length %i) than requested number of clusters (%i)"%(len(data), num_clusters)
    # initialize num_centroids as random choices from data set 
    # data structs: 
        # clusters are dict objects {cluster_num : ([cluster elements], centroid)}
        
    try:
        for d in data:
            pass
    except:
        print('Data to be clustered should be an iterable of objects')
    
    clusters = {}
    data_to_cluster = {}
    for cluster_num in range(num_clusters):
        init_cent = random.choice(data)
        clusters[cluster_num] = ([init_cent], init_cent)
        data_to_cluster[tuple(init_cent)] = cluster_num
         
    not_converged = True
    iter_num = 0
    old_residual = float('inf')
    
    while not_converged and iter_num < num_iters:
        for d in data: # assign each vector to the cluster with the nearest centroid
            
            clust_dists = [(c, distance_measure(d, clusters[c][1])) for c in clusters]
            clust_dists.sort(key=lambda tup: tup[1])
            
            cluster_assignment = clust_dists[0][0]
            clusters[cluster_assignment][0].append(d)
            
            data_to_cluster[data_tags[d]] = cluster_assignment
            
        for c in clusters:# update centroids as average of cluster entries
            clusters[c] = (clusters[c][0], data_mean(clusters[c][0]))
            
        
        dists = [distance_measure(d, clusters[data_to_cluster[data_tags[tuple(d)]]][1]) for d in data]
        new_residual = sum(dists) 
        if  abs(old_residual - new_residual) < convergence_thresh:
            not_converged = False
        else:
            old_residual = new_residual
        
        iter_num += 1
        
    return clusters

def dot_product(a, b):
    '''
    return the dot product of objects a and b  of equal length defined as the 
    sum a[i] * b[i] for all i in
    '''
    n = check_same_length(a, b)
    return sum([a[i] * b[i] for i in range(n)])

def vector_sum(a,b):
    '''
    Return the elementwise sum of two vectors as a list with
    the same length. First operand is summed in place (mutated)
    '''
    n = check_same_length(a, b)
    for i in range(n):
        a[i] += b[i]
    return a
    
def vector_pow(a, pow):
    '''
    raise vector a to a given power elementwise
    operation is performed in place (a is mutated)
    '''
    for i in range(len(list_check(a))):
        a[i] = a[i]**pow
    return a

def multiply_by_scalar(a, c):
    '''
    Scale each element of a vector a by the scalar c. the vector is scaled in place (mutated)
    '''
    assert(len(list_check(c)) == 1), "c must be a scalar but had length %i"%(len(list_check(c)))
    for i in range(len(list_check(a))):
        a[i] *= c
    return a
     
def l2_norm(a):
    '''
    Given a vector a, return its l2_norm 
    '''
    return dot_product(a, a)**.5

def angle(a, b, radians=True):
    '''
     given two vectors, find the angle between them. 
     returns units of radians by default or degrees 
     '''
    cos_theta = dot_product(a, b) / (l2_norm(a) * l2_norm(b) )
    try:
        assert(cos_theta <= 1 and cos_theta >= -1), "cos theta out of domain [-1, 1]:  %f"%cos_theta
    except(AssertionError):
        cos_theta = round(cos_theta, 9)
    theta = math.acos(cos_theta)
    return theta if radians else theta * (180 / math.pi)

def idx_to_grid(idx, n, order):
    '''
    given an index i which gives locations of a vectorized nxn grid, 
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

def grid_to_idx(grid_coord, n, order): 
    '''
    Perform the inverse operation of idx_to_grid above: given a tuple
    grid_coord = (row, col), return the corresponding index in the vectorized n x n grid 
    according to order C or F (row or column major)
    '''
    row, col = grid_coord
    if order.upper() == 'F':
        idx = (row * n) + col 
    elif order.upper() == 'C':
        idx = (col * n) + row

    else:
        assert(False), " Order must be either F (column major), or C (row major), but was %s"%order
        
    return idx

def get_quadrant(vec):
        ''' 
        given a d-vectors, use the sign of each component to return a quadrant number:
        quadrant are determiend by a binary code 0 is positive, 1 is negative so that 
        quadrant[0] = [0, 0, ..., 0, 0],
        quadrant[1] = [0, 0, ..., 0, 1],
        ....
        quadrant[2**d] = [1,1,...,1]
        '''
        signs = [1 if v < 0 else 0 for v in list_check(vec)]    
        return sum([2**i for i in range(len(signs)) if signs[i] > 0])
        
def rmse(a, b):
    '''
    return the root mean square error between two vectors
    if given two M x N matrices, return the rmse of their
    MN x 1 flattened vectors
    '''
    npa = np.asarray(a)
    npb = np.asarray(b)
    n = check_same_length(npa, npb)
    
    assert(npa.shape == npb.shape), 'Given arguments do not have same shape. a = {0}, b = {1}'.format(a.shape, b.shape)
    
    mean_square_err = np.mean(np.square(npa.flatten() - npb.flatten()), axis=0)
    
    return np.sqrt(mean_square_err)
    
def cart_to_polar(a):
    ''' Convert numpy array a to polar coordinates'''
    A = np.sqrt(np.real(a)**2 + np.imag(a)**2)
    theta = np.arctan2(np.imag(a) , np.real(a))
    return A * np.exp(1j * theta)

def pad_to_N_diag_matrix(vec, N):
    '''
    Given a vector [vec], return a N x N diagonal Numpy array
    with [vec][i,i] along the first len(vec) entries, and
    the remaining padded with zeros.
    '''     
    assert(len(list_check(vec)) <= N), " Vector length ({0}) must be less than N ({1}) ".format(len(list_check(vec)), N)
    assert(len(list_check(vec[0])) == 1), "Vector must be a d x 1 array but the first element has size {0}".format(len(list_check(vec[0])))
    
    
    
    diag_matrix = np.zeros((N,N))
    
    for i in range(len(vec)):
        diag_matrix[i,i] =vec[i]
        
    return diag_matrix 
       
def has_real_eigvals(M):
    ''' Check whether a matrix has real or complex eigenvalues'''
    lam, _ = np.linalg.eig(M)
    return np.all(
        np.imag(np.real_if_close(lam)) == np.zeros(lam.shape)
        )


def widen_to_N(M, N, square=True ):
    '''
    Given a p x q matrix, widen matrix to p x N for N > q. If N <= q do nothing.
    If square=True, repeat along axis p to get an N x N matrix. Throws an error if square and p >= N
    '''
    if M.shape[1] < N:
        M = np.hstack((
            M, np.zeros((M.shape[0], N - M.shape[1]))
        ))

    if square:
        if M.shape[0] == N:
            return M
        else:
            assert(M.shape[0] < N), "Matrix with shape {0} cannot be padded to {1}x{2} square matrix".format(M.shape, N, N)
            M = np.vstack((
                M, np.zeros((N - M.shape[0], M.shape[1]))
            ))

    return M

def is_diagonal(M):
    ''' check if a matrix M (numpy 2d array) is diagonal.'''
    try:
        N = M.copy()
        for j in range(M.shape[0]):
            N[j, j] = 0
        if np.all(np.isclose(N, np.zeros(N.shape))):
            return True
        else:
            n_diags = [(i,j) for i in range(N.shape[0]) for j in range(N.shape[1]) if N[i,j] != 0 and i != j]
            print("Matrix has nondiagonal elements at {1}".format(M, n_diags))
            return False

    except Exception as e:
        print("Exception in is_diagonal:  {0}".format(e))
        return False

def real_jordan_form(M):
    '''
     given a real square matrix M, return its real jordan form,  i.e, return J, P where
     A = P^-1 J P,where J is either diagonal or block diagonal and real-valued
    '''
    assert(np.all(np.isreal(M))), "Given matrix is not real"
    assert(M.shape[0]==M.shape[1]), "expected square matrix but had shape {0}".format(M.shape)
    j,p = np.linalg.eig(M)
    J = np.diag(j)
    P = np.zeros(p.shape, dtype=np.complex128)
    dim = M.shape[0]



    for i in range(dim):
        if np.isclose(J[i,i], 0.0):
            continue

        elif np.isreal(J[i,i]):
            P[:, i] = p[:, i]

        else:

            a = np.real(J[i,i])
            b = np.imag(J[i,i])
            J[i, i] = a
            J[i+1, i + 1] = a
            J[i, i+1] = -b
            J[i+1, i] = b
            v= np.hstack((
                np.real(p[:, i:i+1]),
                np.imag(p[:, i:i+1])
            ))
            P[:, i:i+2] = v

    assert (np.all(np.isreal(J))), "Output matrix {0} is not real".format(J)
    assert (np.all(np.isreal(P))), "Output matrix {0} is not real".format(P)

    return np.real(J), np.real(P)




