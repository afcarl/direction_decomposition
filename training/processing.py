import os, subprocess, numpy as np

'''
word indices start at 1, not 0
so that 0-padding is possible
'''
def word_indices(instructions):
    words = [word for phrase in instructions for word in phrase.split(' ')]
    unique = list(set(words))
    indices = { unique[i]: i+1 for i in range(len(unique)) }
    # num_unique = len(indices)
    # print indices
    return indices

'''
0-pads indices so that all sequences
are the same length
'''
def instructions_to_indices(instructions, ind_dict):
    ## split strings to list of words
    instructions = [instr.split(' ') for instr in instructions]
    num_instructions = len(instructions)
    max_instr_length = max([len(instr) for instr in instructions])
    indices_obs = np.zeros( (num_instructions, max_instr_length) )
    for count in range(num_instructions):
        indices = [ind_dict[word] for word in instructions[count]]
        # print indices
        indices_obs[count,-len(indices):] = indices
    # print 'num instr: ', num_instructions
    # print 'max length: ', max_instr_length
    # print indices_obs
    return indices_obs

def mkdir(path):
    if not os.path.exists(path):
        subprocess.Popen(['mkdir', path])