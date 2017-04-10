#!/om/user/janner/anaconda2/envs/pytorch/bin/python

import sys, os, argparse, numpy as np, torch
import decomposition, pipeline, models
## TODO: fix weird namespace issue when saving phi
## (for now: import independently)
sys.path.append('models/')
import state_model

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='data/train_reorg/')
parser.add_argument('--test_path', type=str, default='data/train_reorg/')
parser.add_argument('--save_path', type=str, default='logs/20_5/')
parser.add_argument('--num_worlds', type=int, default=50)
parser.add_argument('--num_test', type=int, default=10)
args = parser.parse_args()

print '\n', args

########## Load models #########
print 'Loading models from', args.save_path

phi = torch.load( os.path.join(args.save_path, 'phi.t7') )
psi = torch.load( os.path.join(args.save_path, 'psi.t7') )

compositor = models.CompositorModel(phi, psi)
print compositor
vis_path = os.path.join(args.save_path, 'vis')
pipeline.mkdir(vis_path)

######## Build test set ########
print 'Building test set from', args.test_path
states, states_dict = decomposition.initialize_states(args.data_path)
_, instruct_obs, _, _ = decomposition.build_goal_observations(args.data_path, range(args.num_worlds), states_dict)
text_vocab = pipeline.word_indices(instruct_obs)
print text_vocab

test_set = decomposition.build_test_set(args.test_path, range(args.num_test), states, states_dict)
test_set = decomposition.test_set_indices(test_set, pipeline.instructions_to_indices, text_vocab)

test_err = pipeline.evaluate(compositor, test_set, vis_path)

