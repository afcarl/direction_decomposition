#!/om/user/janner/anaconda2/envs/pytorch/bin/python

import sys
sys.path.append('/om/user/janner/mit/urop/direction_decomposition/environment/')
from Generator import Generator
from MDP import MDP
from ValueIteration import ValueIteration
# import ValueIteration
import library
from visualization import *
from SpriteWorld import SpriteWorld
from tqdm import tqdm
import os, pickle, argparse, subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--lower', type=int, default=0)
parser.add_argument('--num_worlds', type=int, default=10)
parser.add_argument('--vis_path', type=str, default='../data/dump_new/')
parser.add_argument('--save_path', type=str, default='../data/train_new/')
args = parser.parse_args()

print args, '\n'

def mkdir(path):
    if not os.path.exists(path):
        subprocess.Popen(['mkdir', path])

mkdir(args.vis_path)
mkdir(args.save_path)

gen = Generator(library.objects, library.directions)

for outer in range(args.lower, args.lower + args.num_worlds):
    info = gen.new()
    configurations = len(info['rewards'])

    print 'Generating map', outer, '(', configurations, 'configuations )'
    sys.stdout.flush()

    world = info['map']
    rewards = info['rewards']
    terminal = info['terminal']
    instructions = info['instructions']
    values = []

    sprite = SpriteWorld(library.objects, library.background)
    sprite.makeGrid(world, args.vis_path + str(outer) + '_sprites')

    for inner in tqdm(range(configurations)):
        reward_map = rewards[inner]
        terminal_map = terminal[inner]
        instr = instructions[inner]

        mdp = MDP(world, reward_map, terminal_map)
        vi = ValueIteration(mdp)

        values_list, policy = vi.iterate()
        value_map = mdp.representValues(values_list)
        values.append(value_map)

        visualize_values(mdp, values_list, policy, '../data/dump/' + str(outer) + '_' + str(inner) + '_values', title=instr)

    info['values'] = values
    filename = os.path.join( args.save_path, str(outer) + '.p' )
    pickle.dump( info, open( filename, 'wb' ) )


