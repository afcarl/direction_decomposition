import sys, os, math, pickle, numpy as np
from tqdm import tqdm
sys.path.append('environment/')
import library

'''
Returns a two-channel state observation 
where the first channel marks grass positions with 1's
and puddle positions with 0's
and the second channel marks the agent position with a 2
'''
def reconstruct_state(world, position):
    not_puddles = np.ma.masked_not_equal(world, library.objects['puddle']['index'])
    not_puddles = not_puddles.mask.astype(int)
    position_channel = np.zeros( (not_puddles.shape) )
    # indices 0 and 1 are used for puddle and grass, respectively
    # 2 is used for agent
    position_channel[position] = 2
    state = np.stack( (not_puddles, position_channel) )
    return state

'''
Returns single-channel goal observation with
objects denoted by unique indices beginning at 1.
0's denote background (no object)
'''
def reconstruct_goal(world):
    world = world.copy()
    ## indices for grass and puddle
    background_inds = [obj['index'] for (name, obj) in library.objects.iteritems() if obj['background']]
    ## background mask
    background = np.in1d(world, background_inds)
    background = background.reshape( (world.shape) )
    ## set backgronud to 0
    world[background] = 0
    ## subtract largest background ind
    ## so indices of objects begin at 1
    world[~background] -= max(background_inds)
    world = np.expand_dims(np.expand_dims(world, 0), 0)
    return world


'''
returns np array of size (len(worlds_list) * 400) x state_obs.shape
'''
def build_state_observations(data_path, worlds_list, states, states_dict, state_channels = 2):
    num_worlds = len(worlds_list)
    num_states = len(states_dict)
    state_size = int(math.sqrt(num_states))
    state_obs = np.zeros( (num_worlds * num_states, state_channels, state_size, state_size) )

    for ind, world_num in enumerate(tqdm(worlds_list)):
        path = os.path.join(data_path, str(world_num) + '.p')
        info = pickle.load( open(path, 'r') )
        world = info['map']

        count = ind * num_states
        for position in states:
            # print state_obs.shape, reconstruct_state(world, position).shape
            state_obs[count,:,:,:] = reconstruct_state(world, position)
            # print count, state
            count += 1

        # print count, state_obs.shape
    return state_obs

'''
returns 
1. np array of size (len(worlds_list) * 400) x goal_obs.shape
2. list of instructions for each goal
3. target embedding for each goal
'''
def build_goal_observations(data_path, worlds_list, states_dict, V = None, goal_channels = 1):
    num_worlds = len(worlds_list)
    num_states = len(states_dict)
    state_size = int(math.sqrt(num_states))

    goal_obs = np.zeros( (0, goal_channels, state_size, state_size) )
    instruct_obs = []
    if V != None:
        targets = np.zeros( (0, V.shape[1]) )
    else:
        targets = None
    values = np.zeros( (0, num_states) )

    for world_num in tqdm(worlds_list):
        path = os.path.join(data_path, str(world_num) + '.p')
        info = pickle.load( open(path, 'r') )
        
        world = info['map']
        goals = info['goals']
        vals = info['values']
        instruct = info['instructions']
        configurations = len(goals)

        configurations = len(goals)
        for ind, goal in enumerate(goals):
            obs = reconstruct_goal(world)
            val_map = vals[ind].flatten()

            goal_obs = np.vstack( (goal_obs, obs) )
            instruct_obs.append( instruct[ind] )
            values = np.vstack( (values, val_map) )

            if V != None:
                ## states_dict maps goal positions to column of value matrix
                goal_col = states_dict[goal]
                ## V is num_goals x rank
                targ = V[goal_col, :]
                targets = np.vstack( (targets, targ) )

    return goal_obs, instruct_obs, targets, values

'''
test_set[world_num] = (state_obs, goal_obs, instruct_obs, values)
where state_obs is state at every position, 
goal_obs is the same object observation repeated len(instruct_obs) times
instruct_obs is a list of instructions for the map in state_obs
and values is a np array of len(instruct_obs) x num_states

values[i,:] is the value of _all_ state_obs at goal_obs[i] and instruct[i]
'''
def build_test_set(data_path, worlds_list, states, states_dict):
    num_states = len(states_dict)
    state_size = int(math.sqrt(num_states))

    test_set = {}

    for world_num in tqdm(worlds_list):
        # print 'world num: ', world_num
        state_obs = build_state_observations(data_path, [world_num], states, states_dict)
        goal_obs, instruct_obs, _, values = build_goal_observations(data_path, [world_num], states_dict)

        # print 'test set:', state_obs.shape, goal_obs.shape, len(instruct_obs), values.shape
        test_set[world_num] = (state_obs, goal_obs, instruct_obs, values)

    return test_set













