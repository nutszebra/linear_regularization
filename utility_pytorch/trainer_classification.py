import subprocess
import torch
from torchvision import datasets, transforms
from torch.autograd import Variable
from .utility import remove_slash, make_dir, create_progressbar, write
from .log import Log

torch.backends.cudnn.benchmark = True


class ClassificationTrainer(object):

    def __init__(self, model, optimizer, train_path=None, test_path=None, gpu=-1, save_path='./', load_model=None, train_transform=None, test_transform=None, train_batch_size=64, test_batch_size=256, start_epoch=1, epochs=200, seed=1):
        self.model, self.optimizer = model, optimizer
        self.train_path, self.test_path = train_path, test_path
        self.gpu, self.save_path, load_model = gpu, remove_slash(save_path), load_model
        self.train_transform, self.test_transform = train_transform, test_transform
        self.train_batch_size, self.test_batch_size = train_batch_size, test_batch_size
        self.start_epoch, self.epochs, self.seed = start_epoch, epochs, seed
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
            print('your train_transform will be used')
            self.train_transform = transforms.Compose([
                transforms.RandomSizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

        if self.test_transform is None:
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
        return self.gpu >= 0 and torch.cuda.is_available()

    def init_seed(self):
        torch.manual_seed(self.seed)
        if self.check_gpu():
            torch.cuda.manual_seed(self.seed)

    def to_gpu(self):
        if self.check_gpu():
            self.model = self.model.cuda(self.gpu)

    def to_cpu(self):
        self.model = self.model.cpu()

    def train_one_epoch(self):
        if self.train_path is None:
            return None
        self.to_gpu()
        self.model.train()
        sum_loss = 0
        progressbar = create_progressbar(self.train_loader, desc='train')
        for x, t in progressbar:
            if self.check_gpu():
                x, t = x.cuda(self.gpu), t.cuda(self.gpu)
            x, t = Variable(x, volatile=False), Variable(t, volatile=False)
            self.optimizer.zero_grad()
            y = self.model(x)
            loss = self.model.calc_loss(y, t)
            loss.backward()
            self.optimizer.step()
            sum_loss += loss.cpu().data[0] * self.train_batch_size
        self.to_cpu()
        return sum_loss / len(self.train_loader.dataset)

    def test_one_epoch(self, keep=False):
        if self.test_path is None:
            return None, None
        self.to_gpu()
        self.model.eval()
        sum_loss = 0
        accuracy = 0
        progressbar = create_progressbar(self.test_loader, desc='test')
        if keep:
            results = []
        for x, t in progressbar:
            if self.check_gpu():
                x, t = x.cuda(self.gpu), t.cuda(self.gpu)
            with torch.no_grad():
                x, t = Variable(x), Variable(t)
                y = self.model(x)
                # loss
                loss = self.model.calc_loss(y, t)
            sum_loss += loss.item() * self.test_batch_size
            # accuracy
            y = y.data.max(1, keepdim=True)[1]
            accuracy += y.eq(t.data.view_as(y)).cpu().sum().item()
            if keep:
                y = y.cpu()
                y = y.numpy()[:, 0].tolist()
                results += y
        sum_loss /= len(self.test_loader.dataset)
        accuracy /= len(self.test_loader.dataset)
        self.to_cpu()
        if keep:
            return sum_loss, accuracy, results
        else:
            return sum_loss, accuracy

    def save(self, i):
        self.model.eval()
        torch.save(self.model.state_dict(), '{}/{}_{}.model'.format(self.save_path, self.model.name, i))

    def load(self, path=None):
        if path is not None:
            write('load {}'.format(path))
            self.model.eval()
            self.model.load_state_dict(torch.load(path))
        else:
            write('weight initilization')
            self.model.weight_initialization()

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
