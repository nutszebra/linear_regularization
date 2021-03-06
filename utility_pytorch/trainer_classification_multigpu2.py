import subprocess
import numpy as np
import torch
import torchvision
torchvision.set_image_backend('accimage')
from torchvision import datasets, transforms
from .utility import remove_slash, make_dir, create_progressbar, write
from .log import Log

torch.backends.cudnn.benchmark = True


class ClassificationTrainerMultiGPU(object):

    def __init__(self, model, optimizer, train_path=None, test_path=None, gpu=(0, 1), save_path='./', load_model=None, train_transform=None, test_transform=None, train_batch_size=64, test_batch_size=256, start_epoch=1, epochs=200, seed=1, num_workers=8):
        self.model, self.optimizer = torch.nn.DataParallel(model, device_ids=gpu), optimizer
        self.train_path, self.test_path = train_path, test_path
        self.gpu, self.save_path, load_model = gpu, remove_slash(save_path), load_model
        self.train_transform, self.test_transform = train_transform, test_transform
        self.train_batch_size, self.test_batch_size = train_batch_size, test_batch_size
        self.start_epoch, self.epochs, self.seed, self.num_workers = start_epoch, epochs, seed, num_workers
        # load mnist
        self.init_dataset()
        # initialize seed
        self.init_seed()
        # create directory
        make_dir(save_path)
        # load pretrained model if possible
        self.load(load_model)
        # init log
        self.init_log()

    def init_transform(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        if self.train_transform is None:
            print('your train_transform will not be used')
            self.train_transform = transforms.Compose([
                transforms.RandomSizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])

        if self.test_transform is None:
            print('your test_transform will not be used')
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
            kwargs = {'num_workers': self.num_workers,
                      'pin_memory': True}

        # load dataset
        if self.train_path is not None:
            train_dataset = datasets.ImageFolder(self.train_path,
                                                 transform=self.train_transform)
            self._train_loader = torch.utils.data.DataLoader(train_dataset,
                                                             batch_size=self.train_batch_size,
                                                             shuffle=True,
                                                             drop_last=True,
                                                             **kwargs)
        if self.test_path is not None:
            test_dataset = datasets.ImageFolder(self.test_path,
                                                transform=self.test_transform)
            self.test_loader = torch.utils.data.DataLoader(test_dataset,
                                                           batch_size=self.test_batch_size,
                                                           shuffle=False,
                                                           **kwargs)

    def init_log(self):
        self.log = {}
        self.log['train_loss'] = Log(self.save_path, 'train_loss.log')
        self.log['test_loss'] = Log(self.save_path, 'test_loss.log')
        self.log['test_accuracy'] = Log(self.save_path, 'test_accuracy.log')

    def check_gpu(self):
        return self.gpu[0] >= 0 and torch.cuda.is_available()

    def init_seed(self):
        torch.manual_seed(self.seed)
        if self.check_gpu():
            torch.cuda.manual_seed(self.seed)

    def to_gpu(self):
        if self.check_gpu():
            self.model = self.model.cuda(self.gpu[0])

    def to_cpu(self):
        self.model = self.model.cpu()

    def train_one_epoch(self):
        self.to_gpu()
        self.model.train()
        sum_loss = 0
        train_loader = data_prefetcher(self._train_loader, self.gpu[0])
        progressbar = create_progressbar(len(train_loader), desc='train')
        for i in progressbar:
            with torch.set_grad_enabled(True):
                x, t = train_loader.next()
                if x is None:
                    break
                self.optimizer.zero_grad()
                y = self.model(x)
                loss = self.model.module.calc_loss(y, t)
                loss.backward()
                self.optimizer.step()
            sum_loss += loss.item() * self.train_batch_size
            del x, t, y, loss
        self.to_cpu()
        return sum_loss / len(train_loader._loader.dataset)

    def test_one_epoch(self, keep=False):
        self.to_gpu()
        self.model.eval()
        sum_loss = 0
        accuracy = 0
        progressbar = create_progressbar(self.test_loader, desc='test')
        results = None
        for x, t in progressbar:
            if self.check_gpu():
                x, t = x.cuda(self.gpu[0], async=True), t.cuda(self.gpu[0], async=True)
            with torch.no_grad():
                y = self.model(x)
            # loss
            loss = self.model.module.calc_loss(y, t)
            sum_loss += loss.item() * self.test_batch_size
            if keep:
                _y = torch.softmax(y, 1).cpu()
                if results is None:
                    results = _y
                else:
                    results = np.vstack((results, _y))
            # accuracy
            _y = y
            y = y.data.max(1, keepdim=True)[1]
            accuracy += y.eq(t.data.view_as(y)).sum().item()
        sum_loss /= len(self.test_loader.dataset)
        accuracy /= len(self.test_loader.dataset)
        self.to_cpu()
        if keep:
            return sum_loss, accuracy, results
        else:
            return sum_loss, accuracy

    def save(self, i):
        self.model.eval()
        torch.save(self.model.state_dict(), '{}/{}_{}.model'.format(self.save_path, self.model.module.name, i))

    def load(self, path=None):
        if path is not None:
            write('load {}'.format(path))
            self.model.eval()
            self.model.load_state_dict(torch.load(path))
        else:
            write('weight initilization')
            self.model.module.weight_initialization()

    def run(self):
        hash_git = subprocess.check_output('git log -n 1', shell=True).decode('utf-8').split(' ')[1].split('\n')[0]
        for i in create_progressbar(self.epochs + 1, desc='epoch {}'.format(hash_git), stride=1, start=self.start_epoch):
            train_loss = self.train_one_epoch()
            self.log['train_loss'].write('{}'.format(train_loss), debug='Loss Train {}:'.format(i))
            self.save(i)
            self.optimizer(i)
            test_loss, test_accuracy = self.test_one_epoch()
            self.log['test_loss'].write('{}'.format(test_loss), debug='Loss Test {}:'.format(i))
            self.log['test_accuracy'].write('{}'.format(test_accuracy), debug='Accuracy Test {}:'.format(i))


class data_prefetcher(object):

    def __init__(self, loader, gpu):
        self._loader = loader
        self.loader = iter(loader)
        self.gpu = gpu
        self.stream = torch.cuda.Stream(gpu)
        # transforms.ToTensor() normalize the image between 0 and 1
        self.mean = torch.tensor([0.485, 0.456, 0.406]).cuda(gpu).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).cuda(gpu).view(1, 3, 1, 1)
        self.preload()

    def __iter__(self):
        return self

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
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target

    def __len__(self):
        return len(self._loader)
