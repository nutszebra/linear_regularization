import subprocess
import torch
from torchvision import datasets, transforms
from torch.autograd import Variable
from .utility import remove_slash, make_dir, create_progressbar, write, save_pickle

torch.backends.cudnn.benchmark = True


class Cifar10Trainer(object):

    def __init__(self, model, optimizer, gpu=-1, save_path='./', train_transform=None, test_transform=None, train_batch_size=64, save_period=10, seed=1):
        self.model, self.optimizer = model, optimizer
        self.gpu, self.save_path = gpu, remove_slash(save_path)
        self.train_transform, self.test_transform = train_transform, test_transform
        self.train_batch_size, self.save_period = train_batch_size, save_period
        # initialize transform
        self.init_transform()
        # create directory
        make_dir(save_path)
        # load pretrained model if possible
        self.model.weight_initialization()

    def init_transform(self):
        if self.train_transform is None:
            print('your train_transform will be used')
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        if self.test_transform is None:
            print('your test_transform will be used')
            self.test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

    def create_dataset(self, transformer, batch_size, train, shuffle):
        # arguments for gpu mode
        kwargs = {}
        if self.check_gpu():
            kwargs = {'num_workers': 1,
                      'pin_memory': False}
        loader = torch.utils.data.DataLoader(datasets.CIFAR10('./data_cifar10',
                                             train=train, download=True,
                                             transform=transformer),
                                             batch_size=batch_size,
                                             shuffle=shuffle, **kwargs)
        return loader

    def check_gpu(self):
        return self.gpu >= 0 and torch.cuda.is_available()

    def to_gpu(self):
        if self.check_gpu():
            self.model = self.model.cuda(self.gpu)

    def to_cpu(self):
        self.model = self.model.cpu()

    def train_one_epoch(self, dataset, dataset_test=None):
        progressbar = create_progressbar(dataset, desc='train')
        for i, (x, t) in enumerate(progressbar):
            # periodically save
            if (i % self.save_period) == 0:
                self.save(i)
                easiness = self.test_one_epoch(dataset=dataset_test)
                save_pickle(easiness, '{}/{}_{}.pkl'.format(self.save_path, self.model.name, i))
            self.to_gpu()
            self.model.train()
            # forward and backward here
            if self.check_gpu():
                x, t = x.cuda(self.gpu), t.cuda(self.gpu)
            x, t = Variable(x, volatile=False), Variable(t, volatile=False)
            self.optimizer.zero_grad()
            y = self.model(x)
            loss = self.model.calc_loss(y, t)
            loss.backward()
            self.optimizer.step()
        return True

    def test_one_epoch(self, dataset=None):
        self.to_gpu()
        self.model.eval()
        if dataset is None:
            dataset = self.create_dataset(transformer=self.test_transform, batch_size=self.train_batch_size, train=True, shuffle=False)
        progressbar = create_progressbar(dataset, desc='test')
        result = []
        for i, (x, t) in enumerate(progressbar):
            if self.check_gpu():
                x, t = x.cuda(self.gpu), t.cuda(self.gpu)
            x, t = Variable(x, volatile=False), Variable(t, volatile=False)
            y = self.model(x)
            loss = self.model.calc_loss(y, t, reduction='none').cpu().data.numpy().tolist()
            result += loss
        return result

    def save(self, i):
        print('save: {}'.format(i))
        self.model.eval()
        self.to_cpu()
        torch.save(self.model.state_dict(), '{}/{}_{}.model'.format(self.save_path, self.model.name, i))

    def run(self):
        hash_git = subprocess.check_output('git log -n 1', shell=True).decode('utf-8').split(' ')[1].split('\n')[0]
        print('Execution on {}'.format(hash_git))
        self.train_one_epoch(self.create_dataset(transformer=self.train_transform, batch_size=self.train_batch_size, train=True, shuffle=True))
