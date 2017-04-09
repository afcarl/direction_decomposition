import numpy as np

class MDP:

    def __init__(self, world, rewards, terminal):
        self.world = world
        self.reward_map = rewards
        self.terminal_map = terminal
        self.shape = self.reward_map.shape

        self.actions = [(-1,0),(1,0),(0,-1),(0,1)]
        self.states = [(i,j) for i in range(self.shape[0]) for j in range(self.shape[1])]

    def getActions(self):
        return [i for i in range(len(self.actions))]

    def getStates(self):
        return self.states

    def transition(self, position, action_ind, fullstate=False):
        action = self.actions[action_ind]
        # print 'transitioning: ', action, position
        candidate = tuple(map(sum, zip(position, action)))
        
        ## if new location is valid, 
        ## update the position
        if self.valid(candidate):
            position = candidate
        
        if fullstate:
            state = self.observe(position)
        else:
            state = position

        return state

    def valid(self, position):
        x, y = position[0], position[1]
        if x >= 0 and x < self.shape[0] and y >= 0 and y < self.shape[1]:
            return True
        else:
            return False

    def reward(self, position):
        rew = self.reward_map[position]
        return rew

    def terminal(self, position):
        term = self.terminal_map[position]
        return term

    def representValues(self, values):
        value_map = np.zeros( self.shape )
        for pos, val in values.iteritems():
            assert(value_map[pos] == 0)
            value_map[pos] = val
        return value_map

