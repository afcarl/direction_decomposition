import numpy as np
import scipy.misc

class SpriteWorld:

    def __init__(self, objects, background, dim = 20):
        self.dim = dim
        background = self.loadImg(background)
        self.sprites = {0: background}
        for name, obj in objects.iteritems():
            ind = obj['index']
            # print name
            sprite = self.loadImg( obj['sprite'] )
            if not obj['background']:
                overlay = background.copy()
                masked = np.ma.masked_greater( sprite[:,:,-1], 0 ).mask
                overlay[masked] = sprite[:,:,:-1][masked]
            else:
                overlay = sprite

            self.sprites[ind] = overlay


    def loadImg(self, path):
        img = scipy.misc.imread(path)
        img = scipy.misc.imresize(img, (self.dim, self.dim) )
        return img

    def makeGrid(self, world, filename):
        shape = world.shape
        state = np.zeros( (shape[0]*self.dim, shape[1]*self.dim, 3) )
        for i in range(shape[0]):
            for j in range(shape[1]):
                # if world[i,j] != 0:
                # print 'adding', world[i,j], 'to', i, j
                sprite = self.sprites[world[i,j].astype('int')]
                state[i*self.dim:(i+1)*self.dim, j*self.dim:(j+1)*self.dim, :] = sprite
        scipy.misc.imsave(filename + '.png', state)
        return state
