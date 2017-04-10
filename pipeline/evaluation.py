import os, math, torch
from torch.autograd import Variable
import matplotlib; matplotlib.use('Agg')
from matplotlib import cm
from matplotlib import pyplot as plt

'''
test set is dict from
world number --> (state_obs, goal_obs, instruct_inds, values)
'''

def evaluate(model, test_set, savepath=None):
    for key, (state_obs, goal_obs, instruct_words, instruct_inds, targets) in test_set.iteritems():
        # print torch.Tensor(state_obs).long().cuda()
        state = Variable( torch.Tensor(state_obs).long().cuda() )
        objects = Variable( torch.Tensor(goal_obs).long().cuda() )
        instructions = Variable( torch.Tensor(instruct_inds).long().cuda() )
        targets = torch.Tensor(targets)
        # print state.size(), objects.size(), instructions.size()
        
        preds = model.forward(state, objects, instructions).data.cpu()
        # num_goals = objects.size(0)
        # for ind in range(num_goals):
            # val = model.forward( state, objects[ind], instructions[ind])
        print 'preds:   ', preds.size()
        print 'targets: ', targets.size()
        if savepath:
            num_goals = preds.size(0)
            for goal_num in range(num_goals):
                fullpath = os.path.join(savepath, \
                            str(key) + '_' + str(goal_num) + '.png')
                pred = preds[goal_num].numpy()
                targ = targets[goal_num].numpy()
                instr = instruct_words[goal_num]
                dim = int(math.sqrt(pred.size))
                vmin = min(pred.min(), targ.min())
                vmax = max(pred.max(), targ.max())
                
                plt.clf()
                fig, (ax0,ax1) = plt.subplots(1,2,sharey=True)
                ax0.pcolor(pred.reshape(dim,dim), vmin=vmin, vmax=vmax, cmap=cm.jet)
                # ax0.invert_yaxis()
                ax0.set_title(instr)
                heatmap = ax1.pcolor(targ.reshape(dim,dim), vmin=vmin, vmax=vmax, cmap=cm.jet)
                ax1.invert_yaxis()
                ax1.set_title('target')

                fig.subplots_adjust(right=0.8)
                cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
                fig.colorbar(heatmap, cax=cbar_ax)
                print 'saving to: ', fullpath
                plt.savefig(fullpath, bbox_inches='tight')
                plt.close(fig)

                print pred.shape, targ.shape







