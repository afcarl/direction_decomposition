#!/om/user/janner/anaconda2/envs/pytorch/bin/python

import sys, os, argparse, numpy as np, torch
sys.path.append('/om/user/janner/mit/urop/direction_decomposition')
import decomposition, pipeline, models, sanity_check

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='data/train_reorg/')
parser.add_argument('--test_path', type=str, default='data/train_reorg/')
parser.add_argument('--save_path', type=str, default='logs/trial0/')
parser.add_argument('--num_worlds', type=int, default=10)
parser.add_argument('--num_test', type=int, default=10)
parser.add_argument('--rank', type=int, default=10)

parser.add_argument('--state_embed', type=int, default=3)
parser.add_argument('--obj_embed', type=int, default=3)
parser.add_argument('--goal_hid', type=int, default=15)

parser.add_argument('--lstm_inp', type=int, default=15)
parser.add_argument('--lstm_hid', type=int, default=15)
parser.add_argument('--lstm_layers', type=int, default=1)
parser.add_argument('--lstm_out', type=int, default=15)

parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=int, default=0.001)
parser.add_argument('--phi_iters', type=int, default=1)
parser.add_argument('--psi_iters', type=int, default=1)
parser.add_argument('--optspace_iters', type=int, default=10)
parser.add_argument('--sanity_check', type=int, default=0)
args = parser.parse_args()

print '\n', args

pipeline.mkdir(args.save_path)

states, states_dict = decomposition.initialize_states(args.data_path)

###### U, V decomposition ######

print '\n<Main> Building value matrix from', args.data_path; sys.stdout.flush()
value_mat = decomposition.build_value_matrix(args.data_path, args.num_worlds, states_dict)
nonzero = np.nonzero(value_mat)

print '\n<Main> Low-rank factorization'; sys.stdout.flush()
value_sparse = decomposition.sparsify_values(value_mat)
cols = max([j+1 for (i,j,_) in value_sparse])
print 'cols: ', cols
assert cols == 100, 'Did not observe goal (9,9)'
U, V, recon = decomposition.low_rank(value_sparse, args.rank, num_iter=args.optspace_iters, verbosity=0, outfile=os.path.join(args.save_path, 'optspace_out.txt'))
print '<Main> U: ', U.shape, 'V:', V.shape, 'recon:', recon.shape, 'original:', value_mat.shape

###### Visualize embeddings #####
if args.sanity_check > 0:
    matrix_vis_path = os.path.join(args.save_path, 'matrix_vis')
    print '\n<Main> Visualizing', args.sanity_check, 'low-rank embeddings in', matrix_vis_path; sys.stdout.flush()
    pipeline.mkdir(matrix_vis_path)
    sanity_check.check_factorization(recon, value_mat, matrix_vis_path, num_vis=args.sanity_check)


######### Observations #########

print '\n<Main> Building state observations from', args.data_path; sys.stdout.flush()
state_obs = decomposition.build_state_observations(args.data_path, range(args.num_worlds), states, states_dict)

print '<Main> Building goal observations from args.data_path'; sys.stdout.flush()
goal_obs, instruct_obs, targets, _ = decomposition.build_goal_observations(args.data_path, range(args.num_worlds), states_dict, V=V)

########## Instructions ##########

state_vocab_size = len( np.unique(state_obs) )
obj_vocab_size = len( np.unique(goal_obs) )
text_vocab = pipeline.word_indices(instruct_obs)
text_vocab_size = len(text_vocab) + 1 ## add 1 for 0-padding
print '\n<Main> Text vocabulary: ', text_vocab
print '<Main> Vocabulary size: state %d | objects %d | text %d' % (state_vocab_size, obj_vocab_size, text_vocab_size)
sys.stdout.flush()

########### State model ###########
state_obs = torch.Tensor(state_obs).long().cuda()
U = torch.Tensor(U).cuda()

print '\n<Main> Training phi:', state_obs.size(), '-->', U.size(); sys.stdout.flush()
phi = models.Phi(state_vocab_size, args.state_embed, state_obs[0].size(), args.rank).cuda()
phi = pipeline.Trainer(phi, args.lr, args.batch_size)
phi.train(state_obs, U, iters=args.phi_iters)

######## Instruction model ########

goal_obs = torch.Tensor(goal_obs).long().cuda()
indices_obs = pipeline.instructions_to_indices(instruct_obs, text_vocab)
indices_obs = torch.Tensor(indices_obs).long().cuda()
targets = torch.Tensor(targets).cuda()

text_model = models.TextModel(text_vocab_size, args.lstm_inp, args.lstm_hid, args.lstm_layers, args.lstm_out)
object_model = models.ObjectModel(obj_vocab_size, args.obj_embed, goal_obs[0].size(), args.lstm_out)
psi = models.Psi(text_model, object_model, args.lstm_out, args.goal_hid, args.rank).cuda()
psi = pipeline.Trainer(psi, args.lr, args.batch_size)

print '\n<Main> Training psi: (', goal_obs.size(), 'x', indices_obs.size(), ') -->', targets.size()
psi.train( (goal_obs, indices_obs), targets, iters=args.psi_iters)


# torch.save( phi.model, os.path.join(args.save_path, 'phi.t7') )
# torch.save( psi.model, os.path.join(args.save_path, 'psi.t7') )

######## Build test set ########
print 'Building test set from', args.test_path
test_set = decomposition.build_test_set(args.test_path, range(args.num_test), states, states_dict)
test_set = decomposition.test_set_indices(test_set, pipeline.instructions_to_indices, text_vocab)

# print [i.shape for i in test_set[0]]

compositor = models.CompositorModel(phi.model, psi.model)
print compositor
vis_path = os.path.join(args.save_path, 'vis')
pipeline.mkdir(vis_path)
test_err = pipeline.evaluate(compositor, test_set, vis_path)




# indices_obs = instructions_to_indices(test_set[i][2], text_vocab)
# indices_obs = torch.Tensor(indices_obs).long().cuda()


# phi.train()
# psi.train()

    # def __init__(self, model, criterion, optimizer, batch_size = 32):
    #     self.model = model
    #     self.criterion = criterion
    #     self.optimizer = optimizer
    #     self.batch_size = batch_size

    # def train(self, inputs, targets, repeats = 1):



