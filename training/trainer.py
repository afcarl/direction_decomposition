import sys, math
import numpy as np
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

class Trainer:
    def __init__(self, model, lr, batch_size):
        self.model = model
        self.batch_size = batch_size
        self.criterion = nn.MSELoss(size_average=True).cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def __epoch(self, inputs, targets, repeats = 1):
        self.model.train()
        if type(inputs) == tuple:
            data_size = inputs[0].size(0)
        else:
            data_size = inputs.size(0)
        num_batches = int(math.ceil(data_size / float(self.batch_size)) * repeats)

        err = 0
        for i in range(num_batches):
            inp, targ = self.__get_batch(inputs, targets)
            self.optimizer.zero_grad()
            out = self.model.forward(inp)
            loss = self.criterion(out, targ)
            loss.backward()
            self.optimizer.step()
            err += loss.data[0]
        err = err / float(num_batches)
        return err

    def __get_batch(self, inputs, targets):
        data_size = targets.size(0)

        inds = torch.floor(torch.rand(self.batch_size) * data_size).long().cuda()
        # bug: floor(rand()) sometimes gives 1
        inds[inds >= data_size] = data_size - 1

        if type(inputs) == tuple:
            inp = tuple([Variable( i.index_select(0, inds).cuda() ) for i in inputs])
        else:
            inp = Variable( inputs.index_select(0, inds).cuda() )

        targ = Variable( targets.index_select(0, inds).cuda() )
        return inp, targ

    def train(self, inputs, targets, iters = 10):
        t = trange(iters)
        for i in t:
            err = self.__epoch(inputs, targets)
            t.set_description( str(err) )
        return self.model


# def random_net(out_dim = 7, mode='state'):
#     if mode == 'state':
#         model = Net(out_dim)
#     elif mode == 'goal':
#         model = Net_goal(out_dim)
#     else:
#         raise RuntimeError('Mode not recognized: ' + mode)
#     if CUDA:
#         model = model.cuda()
#     return model

# def run(inputs, targets, out_dim = 7, lr = 0.001, momentum = 0, iters = 2000, mode='state'):
#     inputs = torch.Tensor(inputs)
#     targets = torch.Tensor(targets)
#     assert( inputs.size()[0] == targets.size()[0] )
#     if mode == 'state':
#         model = Net(out_dim)
#     elif mode == 'goal':
#         model = Net_goal(out_dim)
#     else:
#         raise RuntimeError('Mode not recognized: ' + mode)
#     criterion = nn.MSELoss(size_average=True)

#     if CUDA:
#         model = model.cuda()
#         criterion = criterion.cuda()
#         # inputs = inputs.cuda()
#         # targets = targets.cuda()

#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     trainer = Trainer(model, criterion, optimizer)
#     # inputs = torch.randn(300,400)
#     # targets = inputs.sum(1).repeat(1,15)
#     # for i in tqdm(range(iters)):
#     t = trange(iters)
#     for i in t:
#         err = trainer.train(inputs, targets)
#         t.set_description( str(err) )
#         # print i, err
#         # sys.stdout.write( str(err) )
#     return model







