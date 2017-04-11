#!/om/user/janner/anaconda2/envs/pytorch/bin/python

import sys, os, argparse, pickle, numpy as np, torch
sys.path.append('/om/user/janner/mit/urop/direction_decomposition')
import decomposition, pipeline, models, sanity_check

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='data/train_10/')
parser.add_argument('--load_path', type=str, default='pickle/')
parser.add_argument('--save_path', type=str, default='logs/trial_psi/')
parser.add_argument('--num_worlds', type=int, default=20)
# parser.add_argument('--rank', type=int, default=10)

parser.add_argument('--obj_embed', type=int, default=3)
parser.add_argument('--goal_hid', type=int, default=50)

parser.add_argument('--lstm_inp', type=int, default=50)
parser.add_argument('--lstm_hid', type=int, default=50)
parser.add_argument('--lstm_layers', type=int, default=1)
parser.add_argument('--lstm_out', type=int, default=50)

parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=int, default=0.0001)
parser.add_argument('--psi_iters', type=int, default=10000)
args = parser.parse_args()

print '\n', args

pipeline.mkdir(args.save_path)

states, states_dict = decomposition.initialize_states(args.data_path)

# ######### Observations #########

print '\n<Main> Building goal observations from args.data_path'; sys.stdout.flush()
goal_obs, instruct_obs, _, _ = decomposition.build_goal_observations(args.data_path, range(args.num_worlds), states_dict)

########## Instructions ##########

obj_vocab_size = len( np.unique(goal_obs) )
text_vocab = pipeline.word_indices(instruct_obs)
text_vocab_size = len(text_vocab) + 1 ## add 1 for 0-padding
print '\n<Main> Text vocabulary: ', text_vocab
print '<Main> Vocabulary size: objects %d | text %d' % (obj_vocab_size, text_vocab_size)
sys.stdout.flush()

######## Instruction model ########

goal_obs = pickle.load( open( os.path.join(args.load_path + 'goal_obs' + str(args.num_worlds) + '.p'), 'r') )
indices_obs = pickle.load( open( os.path.join(args.load_path + 'indices_obs' + str(args.num_worlds) + '.p'), 'r') )
targets = pickle.load( open( os.path.join(args.load_path, 'targets' + str(args.num_worlds) + '.p'), 'r') )
rank = targets.size(1)

text_model = models.TextModel(text_vocab_size, args.lstm_inp, args.lstm_hid, args.lstm_layers, args.lstm_out)
object_model = models.ObjectModel(obj_vocab_size, args.obj_embed, goal_obs[0].size(), args.lstm_out)
psi = models.Psi(text_model, object_model, args.lstm_out, args.goal_hid, rank).cuda()
psi = pipeline.Trainer(psi, args.lr, args.batch_size)

print '\n<Main> Training psi: (', goal_obs.size(), 'x', indices_obs.size(), ') -->', targets.size()
psi.train( (goal_obs, indices_obs), targets, iters=args.psi_iters)









