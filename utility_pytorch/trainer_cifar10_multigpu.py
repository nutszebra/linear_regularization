import subprocess
import torch
from torchvision import datasets, transforms
from torch.autograd import Variable
from .utility import remove_slash, make_dir, create_progressbar, write
from .log import Log
from .loader_cifar_c import CIFAR10C

torch.backends.cudnn.benchmark = True


class Cifar10TrainerMultiGPU(object):

    def __init__(self, model, optimizer, gpu=(0, 1), save_path='./', load_model=None, train_transform=None, test_transform=None, train_batch_size=64, test_batch_size=256, start_epoch=1, epochs=200, num_workers=8, eval_c=None, train_t=False, test_t=False):
        self.model, self.optimizer = torch.nn.DataParallel(model, device_ids=gpu), optimizer
        self.gpu, self.save_path, load_model = gpu, remove_slash(save_path), load_model
        self.train_transform, self.test_transform = train_transform, test_transform
        self.train_batch_size, self.test_batch_size = train_batch_size, test_batch_size
        self.start_epoch, self.epochs, self.num_workers = start_epoch, epochs, num_workers
        self.eval_c, self.train_t, self.test_t = eval_c, train_t, test_t
        # load mnist
        self.init_dataset()
        # create directory
        make_dir(save_path)
        # load pretrained model if possible
        self.load(load_model)
        # init log
        self.init_log()

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

    def init_dataset(self):
        # initialize transform
        self.init_transform()
        # arguments for gpu mode
        kwargs = {}
        if self.check_gpu():
            kwargs = {'num_workers': self.num_workers,
                      'pin_memory': False}

        # load dataset
        self.train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data_cifar10', train=True, download=True,
                             transform=self.train_transform),
            batch_size=self.train_batch_size, shuffle=True, **kwargs)

        self.test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data_cifar10', train=False,
                             transform=self.test_transform),
            batch_size=self.test_batch_size, shuffle=False, **kwargs)

        if self.eval_c is not None:
            self.test_loader_c = torch.utils.data.DataLoader(
                    CIFAR10C(root=self.eval_c, transform=self.test_transform),
                    batch_size=self.test_batch_size, shuffle=False, **kwargs)

    def init_log(self):
        self.log = {}
        self.log['train_loss'] = Log(self.save_path, 'train_loss.log')
        self.log['test_loss'] = Log(self.save_path, 'test_loss.log')
        self.log['test_accuracy'] = Log(self.save_path, 'test_accuracy.log')
        if self.eval_c is not None:
            self.log['test_loss_c'] = Log(self.save_path, 'test_loss_c.log')
            self.log['test_accuracy_c'] = Log(self.save_path, 'test_accuracy_c.log')

    def check_gpu(self):
        return self.gpu[0] >= 0 and torch.cuda.is_available()

    def to_gpu(self):
        if self.check_gpu():
            self.model = self.model.cuda(self.gpu[0])

    def to_cpu(self):
        self.model = self.model.cpu()

    def train_one_epoch(self):
        self.to_gpu()
        self.model.train()
        sum_loss = 0
        progressbar = create_progressbar(self.train_loader, desc='train')
        for x, t in progressbar:
            if self.check_gpu():
                if type(t) == list:
                    x = x.cuda(self.gpu[0])
                    x = Variable(x, volatile=False)
                else:
                    x, t = x.cuda(self.gpu[0]), t.cuda(self.gpu[0])
                    x, t = Variable(x, volatile=False), Variable(t, volatile=False)
            self.optimizer.zero_grad()
            if self.train_t is True:
                y = self.model(x, t=t)
            else:
                y = self.model(x)
            loss = self.model.module.calc_loss(y, t)
            loss.backward()
            self.optimizer.step()
            sum_loss += loss.item() * self.train_batch_size
        self.to_cpu()
        return sum_loss / len(self.train_loader.dataset)

    def test_one_epoch(self, keep=False, loader=None):
        self.to_gpu()
        self.model.eval()
        sum_loss = 0
        accuracy = 0
        progressbar = create_progressbar(loader, desc='test')
        if keep:
            results = []
        for x, t in progressbar:
            if self.check_gpu():
                x, t = x.cuda(self.gpu[0]), t.cuda(self.gpu[0])
            with torch.no_grad():
                if self.test_t is True:
                    y = self.model(x, t=t)
                else:
                    y = self.model(x)
            # loss
            loss = self.model.module.calc_loss(y, t)
            sum_loss += loss.item() * self.test_batch_size
            # accuracy
            y = y.data.max(1, keepdim=True)[1]
            accuracy += y.eq(t.data.view_as(y)).sum().item()
            if keep:
                y = y.cpu()
                y = y.numpy()[:, 0].tolist()
                results += y
        sum_loss /= len(loader.dataset)
        accuracy /= len(loader.dataset)
        self.to_cpu()
        if keep:
            return sum_loss, accuracy, results
        else:
            return sum_loss, accuracy

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

    def save(self, i):
        self.model.eval()
        torch.save(self.model.state_dict(), '{}/{}.model'.format(self.save_path, self.model.module.name))

    def load(self, path=None):
        if path is not None:
            write('load {}'.format(path))
            self.model.eval()
            self.model.load_state_dict(torch.load(path))
        else:
            write('weight initilization')
            self.model.module.weight_initialization()

    def calculate_easiness(self, dataset):
        self.to_gpu()
        self.model.eval()
        progressbar = create_progressbar(dataset, desc='calc easiness')
        result = []
        for i, (x, t) in enumerate(progressbar):
            if self.check_gpu():
                x, t = x.cuda(self.gpu), t.cuda(self.gpu)
            with torch.no_grad():
                x, t = Variable(x, volatile=False), Variable(t, volatile=False)
                y = self.model(x)
                loss = self.model.module.calc_loss(y, t, reduction='none').cpu().data.numpy().tolist()
            result += loss
        return result

    def run(self):
        hash_git = subprocess.check_output('git log -n 1', shell=True).decode('utf-8').split(' ')[1].split('\n')[0]
        for i in create_progressbar(self.epochs + 1, desc='epoch {}'.format(hash_git), stride=1, start=self.start_epoch):
            train_loss = self.train_one_epoch()
            self.log['train_loss'].write('{}'.format(train_loss), debug='Loss Train {}:'.format(i))
            self.save(i)
            self.optimizer(i)
            test_loss, test_accuracy = self.test_one_epoch(loader=self.test_loader)
            self.log['test_loss'].write('{}'.format(test_loss), debug='Loss Test {}:'.format(i))
            self.log['test_accuracy'].write('{}'.format(test_accuracy), debug='Accuracy Test {}:'.format(i))
        if self.eval_c is not None:
            test_loss_c, test_accuracy_c = self.test_one_epoch(loader=self.test_loader_c)
            self.log['test_loss_c'].write('{}'.format(test_loss_c), debug='Loss Test-C {}:'.format(i))
            self.log['test_accuracy_c'].write('{}'.format(test_accuracy_c), debug='Accuracy Test-C {}:'.format(i))
