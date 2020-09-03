import subprocess
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.autograd import Variable
from .utility import remove_slash, make_dir, create_progressbar, write
from .log import Log

torch.backends.cudnn.benchmark = True

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class AdversarialPerturbationTrainer(object):

    def __init__(self, generator, pretrained_model, optimizer, train_path=None, test_path=None, gpu=-1, save_path='./', load_generator=None, load_pretrained_model=None, train_transform=None, test_transform=None, train_batch_size=64, test_batch_size=256, start_epoch=1, epochs=200, num_workers=8):
        self.generator, self.pretrained_model = torch.nn.DataParallel(generator, device_ids=gpu[:1]), torch.nn.DataParallel(pretrained_model, device_ids=gpu)
        self.optimizer, self.train_path, self.test_path = optimizer, train_path, test_path
        self.gpu, self.save_path = gpu, remove_slash(save_path)
        self.load_generator, self.load_pretrained_model = load_generator, load_pretrained_model
        self.train_transform, self.test_transform = train_transform, test_transform
        self.train_batch_size, self.test_batch_size = train_batch_size, test_batch_size
        self.start_epoch, self.epochs = start_epoch, epochs
        self.num_workers = num_workers
        # load mnist
        self.init_dataset()
        # create directory
        make_dir(save_path)
        # load pretrained model if possible
        self.generator, self.pretrained_model = self.load(self.generator, load_generator), self.load(self.pretrained_model, load_pretrained_model)
        # init log
        self.init_log()

    def init_transform(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        if self.train_transform is None:
            print('defaul train_transform will be used')
            self.train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

        if self.test_transform is None:
            print('default test_transform will be used')
            self.test_transform = transforms.Compose([
                transforms.Resize(256),
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
            self.train_loader = torch.utils.data.DataLoader(train_dataset,
                                                            batch_size=self.train_batch_size,
                                                            shuffle=True,
                                                            **kwargs)
            # self.train_loader.dataset.imgs = np.array(self.train_loader.dataset.imgs)[np.random.permutation(len(self.train_loader.dataset))[:50000]].tolist()
            # self.train_loader.dataset.samples = self.train_loader.dataset.imgs
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
        self.log['test_fooling_rate'] = Log(self.save_path, 'test_fooling_rate.log')

    def check_gpu(self):
        return self.gpu[0] >= 0 and torch.cuda.is_available()

    def to_gpu(self):
        if self.check_gpu():
            self.generator = self.generator.cuda(self.gpu[0])
            self.pretrained_model = self.pretrained_model.cuda(self.gpu[0])

    def to_cpu(self):
        self.generator = self.generator.cpu()
        self.pretrained_model = self.pretrained_model.cpu()

    def train_one_epoch(self):
        self.to_gpu()
        self.generator.train()
        self.pretrained_model.eval()
        progressbar = create_progressbar(self.train_loader, desc='train')
        sum_loss = 0
        for x, _ in progressbar:
            if self.check_gpu() is True:
                x = x.cuda(self.gpu[0])
            x = Variable(x, volatile=False)
            with torch.no_grad():
                t = torch.argmin(self.pretrained_model(x),  dim=1).detach()
            self.optimizer.zero_grad()
            adv = self.generator(x)
            x_adv = self.generator.module.clip(x, adv)
            y = self.pretrained_model(x_adv)
            loss = self.generator.module.calc_loss(y, t)
            loss.backward()
            sum_loss += loss.item()
            self.optimizer.step()
        self.to_cpu()
        return sum_loss / len(self.train_loader.dataset)

    def test_one_epoch(self):
        self.to_gpu()
        self.generator.eval()
        self.pretrained_model.eval()
        # calculate fooling rate on test dataset
        fr = self.fooling_rate(self.test_loader)
        self.to_cpu()
        return fr

    def fooling_rate(self, data_loader):
        self.to_gpu()
        self.generator.eval()
        self.pretrained_model.eval()
        progressbar = create_progressbar(data_loader, desc='fooling_rate')
        non_fool = 0
        for x, t in progressbar:
            with torch.no_grad():
                if self.check_gpu():
                    x, t = x.cuda(self.gpu[0]), t.cuda(self.gpu[0])
                y = self.pretrained_model(x)
                adv = self.generator(x)
                x_adv = self.generator.module.clip(x, adv)
                y_adv = self.pretrained_model(x_adv)
                y, y_adv = y.cpu(), y_adv.cpu()
                non_fool += (torch.argmax(y, dim=1) == torch.argmax(y_adv, dim=1)).sum().item()
        self.to_cpu()
        return (len(data_loader.dataset) - non_fool) / len(data_loader.dataset)

    def save(self, i):
        self.generator.eval()
        torch.save(self.generator.state_dict(), '{}/{}_{}.model'.format(self.save_path, self.generator.module.name, i))

    @staticmethod
    def load(model, path=None):
        if path is not None:
            write('load {}'.format(path))
            model.eval()
            model.load_state_dict(torch.load(path))
        else:
            write('initialize weights randomly')
        return model

    def generate_imgs(self, epoch, howmany):
        self.to_gpu()
        self.generator.eval()
        self.pretrained_model.eval()
        progressbar = create_progressbar(self.test_loader, desc='test')
        with torch.no_grad():
            adv = self.generator()
            self._save_img(self.generator.module.normalize_back(adv.clone())[0], 'Adversarial Noise: {}'.format(epoch), '{}/noise_{}.jpg'.format(self.save_path, epoch))
        counter = 0
        for x, t in progressbar:
            if counter >= howmany:
                break
            with torch.no_grad():
                if self.check_gpu() is True:
                    x, t = x.cuda(self.gpu), t.cuda(self.gpu)
                y = torch.argmax(self.pretrained_model(x), dim=1)
                x_adv = self.generator.module.clip(x, adv.clone())
                y_adv = torch.argmax(self.pretrained_model(x_adv), dim=1)
            x_adv = self.generator.module.normalize_back(x_adv)
            for i in range(len(x)):
                if counter >= howmany:
                    break
                title = '{}->{}'.format(y[i].item(), y_adv[i].item())
                name = '{}/{}_{}.jpg'.format(self.save_path, epoch, counter)
                counter += 1
                self._save_img(x_adv[i], title, name)

    def _test_one_epoch(self):
        self.to_gpu()
        self.generator.eval()
        self.pretrained_model.eval()
        sum_loss = 0
        accuracy = 0
        progressbar = create_progressbar(self.test_loader, desc='test')
        for x, t in progressbar:
            if self.check_gpu():
                x, t = x.cuda(self.gpu[0]), t.cuda(self.gpu[0])
            with torch.no_grad():
                y = self.pretrained_model(x)
            # loss
            loss = self.pretrained_model.module.calc_loss(y, t)
            sum_loss += loss.item() * self.test_batch_size
            # accuracy
            y = y.data.max(1, keepdim=True)[1]
            accuracy += y.eq(t.data.view_as(y)).sum().item()
        sum_loss /= len(self.test_loader.dataset)
        accuracy /= len(self.test_loader.dataset)
        self.to_cpu()
        return sum_loss, accuracy

    def _save_img(self, x, title, name):
        plt.clf()
        plt.imshow(x.cpu().data.numpy().transpose(1, 2, 0))
        plt.title(title)
        plt.savefig(name)

    def run(self):
        hash_git = subprocess.check_output('git log -n 1', shell=True).decode('utf-8').split(' ')[1].split('\n')[0]
        for i in create_progressbar(self.epochs + 1, desc='epoch {}'.format(hash_git), stride=1, start=self.start_epoch):
            train_loss = self.train_one_epoch()
            self.log['train_loss'].write('{}'.format(train_loss), debug='Train Loss {}:'.format(i))
            self.save(i)
            self.optimizer(i)
            test_fooling_rate = self.test_one_epoch()
            self.log['test_fooling_rate'].write('{}'.format(test_fooling_rate), debug='Test Fooling Rate {}:'.format(i))
            # self.generate_imgs(i, 10)
