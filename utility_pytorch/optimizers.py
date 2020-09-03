import math
import numpy as np
import torch.optim as optim
from .utility import write


class FakeOptimizer(object):

    def __call__(self, i):
        pass

    def step(self):
        pass

    def zero_grad(self):
        pass

    def info(self):
        pass


class Mix(object):

    def __init__(self, p_models, lr1, lr2, momentum, schedule=[100, 150], lr_decay=0.1, weight_decay1=1.0e-4, weight_decay2=1.0e-4, betas=(0.9, 0.98)):
        p_model1, p_model2 = p_models
        self.momentum = momentum
        self.schedule, self.lr_decay = schedule, lr_decay
        self.weight_decay1, self.weight_decay2 = weight_decay1, weight_decay2
        self.optimizer1 = optim.SGD(p_model1, lr=lr1, momentum=momentum, weight_decay=weight_decay1, nesterov=True)
        self.optimizer2 = optim.Adam(p_model2, lr=lr2, betas=betas, weight_decay=weight_decay2)

    def __call__(self, i):
        if i in self.schedule:
            for p in self.optimizer1.param_groups:
                previous_lr = p['lr']
                new_lr = p['lr'] * self.lr_decay
                print('{}->{}'.format(previous_lr, new_lr))
                p['lr'] = new_lr
            for p in self.optimizer2.param_groups:
                previous_lr = p['lr']
                new_lr = p['lr'] * self.lr_decay
                print('{}->{}'.format(previous_lr, new_lr))
                p['lr'] = new_lr
            self.info()

    def step(self):
        self.optimizer1.step()
        self.optimizer2.step()

    def zero_grad(self):
        self.optimizer1.zero_grad()
        self.optimizer2.zero_grad()

    def info(self):
        write('Optimizer')
        keys = self.__dict__.keys()
        for key in keys:
            write('    {}: {}'.format(key, self.__dict__[key]))


class MomentumSGD(object):

    def __init__(self, model_parameters, lr, momentum, schedule=[100, 150], lr_decay=0.1, weight_decay=1.0e-4):
        self.lr, self.momentum = lr, momentum
        self.schedule, self.lr_decay, self.weight_decay = schedule, lr_decay, weight_decay
        self.optimizer = optim.SGD(model_parameters, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)

    def __call__(self, i):
        if i in self.schedule:
            for p in self.optimizer.param_groups:
                previous_lr = p['lr']
                new_lr = p['lr'] * self.lr_decay
                print('{}->{}'.format(previous_lr, new_lr))
                p['lr'] = new_lr
            self.lr = new_lr
            self.info()

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def info(self):
        write('Optimizer')
        keys = self.__dict__.keys()
        for key in keys:
            if key == 'model':
                continue
            else:
                write('    {}: {}'.format(key, self.__dict__[key]))


class MomentumSGD2(object):

    def __init__(self, model_parameters, lr, momentum, schedule=[100, 150], lr_decay=0.1, weight_decay=1.0e-4):
        self.lr, self.momentum = lr, momentum
        self.schedule, self.lr_decay, self.weight_decay = schedule, lr_decay, weight_decay
        self.optimizer = sgd.SGD(model_parameters, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)

    def __call__(self, i):
        if i in self.schedule:
            for p in self.optimizer.param_groups:
                previous_lr = p['lr']
                new_lr = p['lr'] * self.lr_decay
                print('{}->{}'.format(previous_lr, new_lr))
                p['lr'] = new_lr
            self.lr = new_lr
            self.info()

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def info(self):
        write('Optimizer')
        keys = self.__dict__.keys()
        for key in keys:
            if key == 'model':
                continue
            else:
                write('    {}: {}'.format(key, self.__dict__[key]))
