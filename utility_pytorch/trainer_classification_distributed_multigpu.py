import copy
import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision import datasets, transforms

try:
    from .log import Log
    from .utility import create_progressbar, write
except:
    from log import Log
    from utility import create_progressbar, write

torch.backends.cudnn.benchmark = True


def main(args):
    ngpus_per_node = torch.cuda.device_count()
    mp.spawn(worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def worker(gpu, ngpus_per_node, args):
    # mp.spawn gives gpu ids according to range(0, ngpus_per_node)
    args.gpu = gpu
    args.rank = gpu
    # modify parameters since models are distributed
    args.train_batch_size = int(args.train_batch_size / ngpus_per_node)
    args.num_workers = int(args.num_workers / ngpus_per_node)
    args.log = {}
    args.log['train_loss'] = Log(args.save_path, 'train_loss.log')

    # For multiprocessing distributed training, rank needs to be the
    # global rank among all the processes
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=ngpus_per_node, rank=gpu)

    # copy model
    model = copy.deepcopy(args.model)

    # loading model if it's given
    if args.load_model is not None:
        write('GPU: {} -> load {}'.format(args.load_model))
        model.load_state_dict(torch.load(args.load_model))

    # to gpu and distributed data parallel
    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # define optimizer
    optimizer = eval('{}'.format(args.optimizer))

    # create loader
    #    train dataset
    train_dataset = datasets.ImageFolder(
        args.train_path,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ]))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=ngpus_per_node, rank=gpu)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler, collate_fn=fast_collate)
    prefetcher = data_prefetcher(train_loader, args.gpu)

    for epoch in range(args.start_epoch, args.epochs):
        # init
        train_sampler.set_epoch(epoch)
        optimizer(epoch)

        # train for one epoch
        train_loss = train_one_epoch(prefetcher, model, optimizer, epoch, args)
        args.log['train_loss'].write('{}'.format(train_loss / len(train_sampler)), debug='{} -> GPU: {} -> Loss Train '.format(epoch, args.gpu))
        save(model, '{}/{}_{}_{}.model'.format(args.save_path, args.gpu, model.name, epoch))


def train_one_epoch(prefetcher, model, optimizer, epoch, args):
    # init
    model.train()
    x, t = prefetcher.next()
    sum_loss = 0

    while x is not None:

        if args.gpu is not None:
            x = x.cuda(args.gpu, non_blocking=True)
        t = t.cuda(args.gpu, non_blocking=True)

        # compute output
        y = model(input)
        loss = model.calc_loss(y, t)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_loss += loss.item() * args.train_batch_size

        # next batch
        x, t = prefetcher.next()

    return sum_loss


def save(model, path):
    model.eval()
    model = model.cpu()
    torch.save(model.state_dict(), path)


def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if(nump_array.ndim < 3):
            # deal with gray image
            nump_array = np.expand_dims(nump_array, axis=-1)
        # transpose
        nump_array = np.rollaxis(nump_array, 2)
        tensor[i] += torch.from_numpy(nump_array)
    return tensor, targets


class data_prefetcher(object):

    def __init__(self, loader, gpu):
        self.loader = iter(loader)
        self.gpu = gpu
        self.stream = torch.cuda.Stream(gpu)
        # transforms.ToTensor() normalize the image between 0 and 1
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda(gpu).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda(gpu).view(1, 3, 1, 1)
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)

        except StopIteration:
            self.next_input = None
            self.next_target = None
            return None, None

        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(self.gpu, non_blocking=True)
            self.next_target = self.next_target.cuda(self.gpu, non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target


import argparse
import math
import functools
import numpy as np
from torchvision.models import *
import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import sys, os
sys.path.append(os.pardir)
from models import resnet
from utility_pytorch.optimizers import MomentumSGD, AdamW

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch cifar10 Example')
    parser.add_argument('--save_path', type=str, default='./log', metavar='N',
                        help='log and model will be saved here')
    parser.add_argument('--train_path', type=str, default='/home/nutszebra/Downloads/ILSVRC/Data/CLS-LOC/train', metavar='N',
                        help='path for train images')
    parser.add_argument('--test_path', type=str, default='/home/nutszebra/Downloads/ILSVRC/Data/CLS-LOC/val', metavar='N',
                        help='path for test images')
    parser.add_argument('--load_model', default=None, metavar='N',
                        help='pretrained model')
    parser.add_argument('--train_batch_size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                        help='epochs start from this number')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--num_workers', type=int, default=8, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--model', type=str, default='resnet.resnet50(False,num_classes=1000,ratio=0.1)', metavar='M',
                        help='alexnet, densenet[121, 161, 169, 201], inception, inception3, resnet[18, 34, 50, 101, 152], squeezenet, vgg[11, 13, 16, 19], vgg[11, 13, 16, 19]_bn')
    parser.add_argument('--optimizer', type=str, default='MomentumSGD(model,0.1,0.9,schedule=[100,150],weight_decay=1.0e-4)', metavar='M',
                        help='optimizer definition here')
    parser.add_argument('--dist_backend', type=str, default='nccl', metavar='M',
                        help='backend')
    parser.add_argument('--dist_url', type=str, default="tcp://127.0.0.1:8080", metavar='M',
                        help='url')
    args = parser.parse_args()
    print('Args')
    print('    {}'.format(args))
    exec('args.model = {}'.format(args.model))
    x = Variable(torch.randn(1, 3, 224, 224))
    args.model(x)
    main(args)
