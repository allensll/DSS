import os
import time
import shutil
import math
from contextlib import contextmanager
from itertools import combinations

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

from scipy import io

from resnet_sm import resnet20

data_path = os.path.join(os.path.dirname(__file__), 'data')
torch.random.manual_seed(1)


@contextmanager
def timer(text=''):
    """Helper for measuring runtime"""

    time_start = time.perf_counter()
    yield
    print('---{} time: {:.5f} s'.format(text, time.perf_counter()-time_start))


def load_MNIST(batch_size, test_batch_size=1000, **kwargs):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_path, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_path, train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=test_batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader


def load_CIFAR10(batch_size, test_batch_size=1000, **kwargs):
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(data_path, train=True, download=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(data_path, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])),
        batch_size=test_batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader


def load_CIFAR100(batch_size, test_batch_size=1000, **kwargs):
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(data_path, train=True, download=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4866, 0.4412), (0.2673, 0.2564, 0.2761)),
        ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(data_path, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4866, 0.4412), (0.2673, 0.2564, 0.2761)),
        ])),
        batch_size=test_batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader


def load_CelebA(batch_size, test_batch_size=1000, **kwargs):
    train_loader = torch.utils.data.DataLoader(
        dataset=datasets.ImageFolder(
            os.path.join(data_path, 'CelebA', 'train'),
            transforms.Compose([
                transforms.Resize(80),
                transforms.RandomCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        dataset=datasets.ImageFolder(
            os.path.join(data_path, 'CelebA', 'test'),
            transforms.Compose([
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])),
        batch_size=test_batch_size, shuffle=True, **kwargs)

    return train_loader, test_loader


def load_TinyImageNet(batch_size, test_batch_size=1000, **kwargs):
    train_loader = torch.utils.data.DataLoader(
        dataset=datasets.ImageFolder(
            os.path.join(data_path, 'tiny-imagenet-200', 'train'),
            transforms.Compose([
                transforms.RandomCrop(64, padding=8),
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        dataset=datasets.ImageFolder(
            os.path.join(data_path, 'tiny-imagenet-200', 'test'),
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ])),
        batch_size=test_batch_size, shuffle=True, **kwargs)

    return train_loader, test_loader


def CosS(p, q):
    sim = F.cosine_similarity(p, q)
    return 0.5 * (sim + 1)


def JSDiv(p, q):
    p, q = F.softmax(p, dim=1), F.softmax(q, dim=1)
    p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
    m = (0.5 * (p + q)).log()
    kl1 = F.kl_div(m, p.log(), reduction='batchmean', log_target=True)
    kl2 = F.kl_div(m, q.log(), reduction='batchmean', log_target=True)
    js = 0.5 * (kl1 + kl2)
    return 1 - js / math.log(2)


def L2Sim(p, q):
    dis = F.pairwise_distance(p, q, p=2)
    sim = 1 / (1 + dis)
    return sim


def multisim1(outputs, device, mod='cos', alpha=0., e=False):
    if mod == 'cos':
        f = CosS
    elif mod == 'jsd':
        f = JSDiv
    elif mod == 'l2s':
        f = L2Sim
    l = len(outputs)
    sims = torch.zeros(l).to(device)
    for i in range(l):
        for j in range(l):
            if i != j:
                sim = f(outputs[i], outputs[j])
                sim = torch.mean(sim)
                sims[i] += sim
    sims /= l - 1
    if e:
        print(sims)
    # return torch.mean(sims) / (l - 1)
    return torch.mean(sims) + alpha * torch.max(sims)


def max_ce(output_subs, target, device):
    l = len(output_subs)
    loss_subs = torch.zeros(l).to(device)
    for i in range(l):
        loss_subs[i] = 1 / (1 + F.cross_entropy(output_subs[i], target))
    loss = torch.max(loss_subs)
    return loss


def supsub(output_subs, output, target, device, e=False):
    l = len(output_subs)
    loss_subs = torch.zeros(l).to(device)
    for i in range(l):
        loss_subs[i] = 1 / (1 + F.cross_entropy(output - output_subs[i], target))
    # loss = torch.mean(loss_subs)
    loss = torch.max(loss_subs)
    if e:
        print(loss)
    return loss


def supsub2(output_subs, _, target, device, e=False):
    l = len(output_subs)
    loss_subs = torch.zeros(2**l-1).to(device)
    p = 0
    idxs = [i for i in range(l)]
    for n in range(1, l):
        idx = list(combinations(idxs, n))
        for id in idx:
            output = sum([output_subs[i] for i in id])
            loss_subs[p] = 1 / (1 + F.cross_entropy(output, target))
            p += 1
    loss = torch.max(loss_subs)
    if e:
        print(loss)
    return loss


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    https://github.com/ildoonet/pytorch-gradual-warmup-lr

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


if __name__ == '__main__':

    move_valimg()
