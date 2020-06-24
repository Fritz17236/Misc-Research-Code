'''
Utility Functions 
'''
import random
import math




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
    ''' return the root mean square error between two vectors'''
    n = check_same_length(a, b)
    err = vector_sum(a, multiply_by_scalar(b, -1))
    mean_square_err = dot_product(err, err) / n
    return math.sqrt(mean_square_err)
     

