import math
import matplotlib; matplotlib.use('Agg')
from matplotlib import cm
from matplotlib import pyplot as plt

def visualize_value_map(pred, targ, save_path, title='prediction'):
    # print 'in vis: ', pred.shape, targ.shape
    dim = int(math.sqrt(pred.size))
    vmin = min(pred.min(), targ.min())
    vmax = max(pred.max(), targ.max())

    plt.clf()
    fig, (ax0,ax1) = plt.subplots(1,2,sharey=True)
    ax0.pcolor(pred.reshape(dim,dim), vmin=vmin, vmax=vmax, cmap=cm.jet)
    ax0.set_title(title)
    heatmap = ax1.pcolor(targ.reshape(dim,dim), vmin=vmin, vmax=vmax, cmap=cm.jet)
    ax1.invert_yaxis()
    ax1.set_title('target')

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(heatmap, cax=cbar_ax)
    # print 'saving to: ', fullpath
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

    # print pred.shape, targ.shape