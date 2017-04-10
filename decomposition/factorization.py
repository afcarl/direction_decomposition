import sys, os, math, pickle, argparse, scipy.sparse, numpy as np
from tqdm import tqdm
sys.path.append('pyOptSpace/')
import optspace

'''
Returns dictionary mapping states of the form (i,j)
to an index representing the column in the value matrix.
Assumes there is a file '0.p' in the folder data_path/
'''
def initialize_states(data_path):
    path = os.path.join(data_path, '0.p')
    info = pickle.load( open(path, 'r') )
    shape = info['map'].shape

    states = [(i,j) for i in range(shape[0]) for j in range(shape[1])]
    num_states = len(states)
    states_dict = {states[i]: i for i in range(num_states)}
    return states, states_dict

'''
Builds (num_worlds*states_per_world) x (goals_per_world)
sparse value matrix based on instructions in data_path.
By default, states_per_world = goals_per_world = 400
(although <=45 are observed per map layout)
'''
def build_value_matrix(data_path, num_worlds, states_dict):
    num_states = len(states_dict)
    value_mat = np.zeros( (num_worlds * num_states, num_states) )
    
    for outer in tqdm(range(num_worlds)):
        path = os.path.join(data_path, str(outer) + '.p')
        info = pickle.load( open(path, 'r') )
        goals = info['goals']
        values = info['values']
        configurations = len(goals)

        for inner in range(configurations):
            goal = goals[inner]
            value_map = values[inner].flatten()

            i_low = outer * num_states
            i_high = (outer+1) * num_states
            j = states_dict[goal]
            value_mat[i_low:i_high, j] = value_map

    return value_mat

'''
Converts sparse np array value_mat to list representation
of the form [(i,j,val)] for all (i,j) for which 
value_mat[i][j] != 0
'''
def sparsify_values(value_mat):
    value_sparse = []
    for i in tqdm(range(value_mat.shape[0])):
        for j in range(value_mat.shape[1]):
            val = value_mat[i][j]
            if val != 0:
                # print i, j
                value_sparse.append( (i, j, val) )
    return value_sparse

'''
Get low-rank embeddings of sparse np array 
represented as list (output of sparsify_values) 

'''
def low_rank(sparse_list, rank_n, num_iter = 10000, verbosity = 1, outfile = ''):
    (X, S, Y) = optspace.optspace(sparse_list, rank_n = rank_n,
                    num_iter = num_iter,
                    tol = 1e-2,
                    verbosity = verbosity,
                    outfile = outfile
                )
    U = np.dot(X, S)
    V = Y
    recon = np.dot(U, V.T)
    return U, V, recon



# states, states_dict = initialize_states(args.data_path)

# print '<Main> Building value matrix'
# value_mat = build_value_matrix(args.data_path, args.num_worlds, states_dict)
# nonzero = np.nonzero(value_mat)

# print '<Main> Converting to sparse representation'
# value_sparse = sparsify_values(value_mat)
# cols = max([j+1 for (i,j,_) in value_sparse])
# assert cols == 400, 'Did not observe goal (19,19)'

# U, V, recon = low_rank(value_sparse, args.rank, num_iter=10, verbosity=0)

# print '<Main> Building state observations'
# state_obs = build_state_observations(args.data_path, args.num_worlds, states, states_dict)
# print '    states:', state_obs.shape, '-->', U.shape

# print '<Main> Building goal observations'
# goal_obs, instruct_obs, targets = build_goal_observations(args.data_path, args.num_worlds, states_dict, V)
# print '    goals:', goal_obs.shape, 'x', len(instruct_obs), '-->', targets.shape

# # print instruct_obs

# print U.shape, V.shape, recon.shape

# # print value_mat
# # value_sparse = scipy.sparse.lil_matrix(value_mat)
# # for i in value_sparse:
# #     print i, type(i)



#     # pass
# # print 'sparse: ', value_sparse




# # print value_mat.shape, recon.shape

# # print 'saving'
# # fig, (ax0, ax1) = plt.subplots( 1,2, sharey=True )
# # print 1
# # ax0.pcolor(value_mat)
# # print 2
# # ax1.pcolor(recon)
# # print 'done'
# # plt.savefig('factor.png')
#     # print i, info['goals'],len(info['goals'])
#     # print 'values: ', info['values'][0].shape, info['values'][0].flatten().shape

# print a.b


# from matplotlib import pyplot as plt

# U = np.random.randn(100,10)
# V = np.random.randn(50,10)

# matrix = np.dot(U, V.T)
# missing_mask = np.ma.masked_greater(np.random.uniform(size = matrix.shape), 0.1).mask.astype(int)
# print 'mask: ', missing_mask
# observed = matrix.copy()
# observed[missing_mask] = 0
# print 'observed: ', observed
# print 'sum: ', observed.sum()

# mf = MF()
# recon, U_pred, V_pred = mf.solve( matrix, missing_mask )
# print recon.shape, U_pred.shape, V_pred.shape

# for i in range(10):
#     print i

