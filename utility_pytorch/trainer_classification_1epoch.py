import subprocess
import torch
from torchvision import datasets, transforms
from torch.autograd import Variable
from .utility import remove_slash, make_dir, create_progressbar, write, save_pickle

torch.backends.cudnn.benchmark = True


class ClassificationTrainer(object):

    def __init__(self, model, optimizer, train_path=None, gpu=-1, save_path='./', train_transform=None, train_batch_size=64, save_period=100):
        self.model, self.optimizer = model, optimizer
        self.train_path = train_path
        self.gpu, self.save_path = gpu, remove_slash(save_path)
        self.train_transform, self.train_batch_size = train_transform, train_batch_size
        self.save_period = save_period
        # load mnist
        self.init_dataset()
        # create directory
        make_dir(save_path)
        # load pretrained model if possible
        self.model.weight_initialization()

    def init_transform(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        print('your train_transform will be used')
        self.train_transform = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        print('your test_transform will be used')
        self.test_transform = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    def init_dataset(self):
        # initialize transform
        self.init_transform()
        # arguments for gpu mode
        kwargs = {}
        if self.check_gpu():
            kwargs = {'num_workers': 1,
                      'pin_memory': False}

        # load dataset
        if self.train_path is not None:
            train_dataset = datasets.ImageFolder(self.train_path,
                                                 transform=self.train_transform)
            self.train_loader = torch.utils.data.DataLoader(train_dataset,
                                                            batch_size=self.train_batch_size,
                                                            shuffle=True,
                                                            **kwargs)
        test_dataset = datasets.ImageFolder(self.train_path,
                                            transform=self.test_transform)
        self.test_loader = torch.utils.data.DataLoader(test_dataset,
                                                       batch_size=self.train_batch_size,
                                                       shuffle=False,
                                                       **kwargs)

    def check_gpu(self):
        return self.gpu >= 0 and torch.cuda.is_available()

    def to_gpu(self):
        if self.check_gpu():
            self.model = self.model.cuda(self.gpu)

    def to_cpu(self):
        self.model = self.model.cpu()

    def train_one_epoch(self, dataset):
        progressbar = create_progressbar(dataset, desc='train')
        for i, (x, t) in enumerate(progressbar):
            # periodically save
            if (i % self.save_period) == 0:
                self.save(i)
                easiness = self.test_one_epoch()
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

    def test_one_epoch(self):
        self.to_gpu()
        self.model.eval()
        dataset = self.test_loader
        progressbar = create_progressbar(dataset, desc='test')
        result = []
        for i, (x, t) in enumerate(progressbar):
            if self.check_gpu():
                x, t = x.cuda(self.gpu), t.cuda(self.gpu)
            x, t = Variable(x), Variable(t)
            with torch.no_grad():
                y = self.model(x)
                loss = self.model.calc_loss(y, t, reduction='none').cpu().data.numpy().tolist()
                result += loss
        return result

    def save(self, i):
        self.to_cpu()
        self.model.eval()
        torch.save(self.model.state_dict(), '{}/{}_{}.model'.format(self.save_path, self.model.name, i))

    def run(self):
        hash_git = subprocess.check_output('git log -n 1', shell=True).decode('utf-8').split(' ')[1].split('\n')[0]
        print('Execution on {}'.format(hash_git))
        self.train_one_epoch(self.train_loader)
