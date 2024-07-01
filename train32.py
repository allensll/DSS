import os
import argparse
import sys
from itertools import combinations

absPath = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(absPath)
# print(sys.path)

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import utils
from model import Shar


def train(model, device, data_loader, optimizer, epoch, mod='cos', method=1, lam=1, alpha=0):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output_subs, output = model(data)

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        loss = F.cross_entropy(output, target)
        e = epoch % 5 == 0 and batch_idx % 100 == 0
        if method == 0:
            # Cross-Entropy
            loss1 = loss2 = 0
        elif method == 1:
            # Basic DSS
            loss1 = utils.max_ce(output_subs, target, device)
            loss2 = 0
        elif method == 2:
            # MinSim DSS
            loss1 = utils.multisim1(output_subs, device, mod=mod, e=e)
            loss2 = 0
        elif method == 3:
            loss1 = utils.multisim1(output_subs, device, mod=mod, e=e)
            loss2 = utils.supsub(output_subs, output, target, device, e=e)
        elif method == 4:
            # MinSim+ DSS
            loss1 = utils.multisim1(output_subs, device, mod=mod, e=e)
            loss2 = utils.supsub2(output_subs, output, target, device, e=e)
        else:
            loss1 = loss2 = 0
        loss = loss + lam * loss1 + alpha * loss2
        loss.backward()

        optimizer.step()

        if batch_idx % 200 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                       100. * batch_idx / len(data_loader), loss.item()))


def test(model, device, data_loader, epoch):
    model.eval()
    test_loss = 0
    corrects = [0] * (model.n_party + 1)
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output_subs, output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()

            for i in range(len(output_subs)):
                pred = output_subs[i].argmax(dim=1, keepdim=True)
                corrects[i] += pred.eq(target.view_as(pred)).sum().item()
            pred = output.argmax(dim=1, keepdim=True)
            corrects[-1] += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(data_loader.dataset)
    print('\r\nTest set: Avg loss: {:.4f}, '.format(test_loss), end='')
    for i in range(model.n_party):
        print('Acc{}: {:.2f}%, '.format(i+1, 100. * corrects[i] / len(data_loader.dataset)), end='')
    print('- Acc: {:.2f}%'.format(100. * corrects[-1] / len(data_loader.dataset)))

    return 100. * corrects[-1] / len(data_loader.dataset)


def main(args):
    lr = 0.1
    m = 0.9
    wd = 0.0001

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    if args.dataset == 'cifar10':
        train_loader, test_loader = utils.load_CIFAR10(args.batch_size, **kwargs)
    elif args.dataset == 'cifar100':
        train_loader, test_loader = utils.load_CIFAR100(args.batch_size, **kwargs)
    elif args.dataset == 'imagenet':
        train_loader, test_loader = utils.load_ImageNet(args.batch_size, **kwargs)

    model = Shar(arch=args.arch, n_party=args.n_party, num_classes=args.num_cls)
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr, momentum=m, weight_decay=wd)

    wp = True
    if wp:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 70], gamma=0.1)
        scheduler_warmup = utils.GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=10, after_scheduler=lr_scheduler)
    else:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)

    best_acc = 0
    optimizer.zero_grad()
    optimizer.step()
    for e in range(1, args.epoch + 1):
        if wp:
            scheduler_warmup.step(e)
        with utils.timer('epoch time: '):
            train(model, device, train_loader, optimizer, e, args.mod, args.method, args.lam, args.alpha)
            acc = test(model, device, test_loader, e)
        lr_scheduler.step()

        # remember best acc
        is_best = acc > best_acc
        if is_best:
            best_acc = acc
            if args.save_model:
                torch.save({
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                }, 'pretrained/{}_checkpoint_best_M{}{}_{}_{}_{}.pth'.format(args.arch, args.method, args.mod, args.lam, args.alpha, args.n_party))

    print('Best Acc : {}%'.format(best_acc))


def acc_party(args):
    """ use to compute MC accuracy """

    test_batch_size = 1000

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    # device = torch.device('cpu')
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    if args.dataset == 'cifar10':
        _, test_loader = utils.load_CIFAR10(args.batch_size, test_batch_size=test_batch_size, **kwargs)
    elif args.dataset == 'cifar100':
        _, test_loader = utils.load_CIFAR100(args.batch_size, test_batch_size=test_batch_size, **kwargs)

    model = Shar(arch=args.arch, n_party=args.n_party, num_classes=args.num_cls)
    model = model.to(device)
    if not args.random_init:
        data_path = 'pretrained/{}_checkpoint_best_M{}{}_{}_{}_{}.pth'.format(args.arch, args.method, args.mod, args.lam, args.alpha, args.n_party)
        state_dict = torch.load(data_path, map_location=torch.device('cpu'))['state_dict']
        model.load_state_dict(state_dict)

    model.eval()
    idxs = [i for i in range(args.n_party)]
    res = [[] for i in range(args.n_party)]
    for n in range(1, args.n_party+1):
        idx = list(combinations(idxs, n))
        test_loss = 0
        corrects = 0
        for id in idx:
            corrects_all = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output_subs, _ = model(data)

                    output = output_subs[id[0]]
                    for j in range(1, n):
                        output += output_subs[id[j]]
                    test_loss += F.cross_entropy(output, target, reduction='sum').item()
                    pred = output.argmax(dim=1, keepdim=True)
                    corrects += pred.eq(target.view_as(pred)).sum().item()
                    corrects_all += pred.eq(target.view_as(pred)).sum().item()
            res[n-1].append(100 * corrects_all / len(test_loader.dataset))
        test_loss /= len(idx)
        corrects /= len(idx)
        test_loss /= len(test_loader.dataset)
        print('\r\nTest set {} party: Avg loss: {:.4f}, - Acc: {:.2f}%'.format(n, test_loss, 100. * corrects / len(test_loader.dataset)))
    print('\r\n All: {}'.format(res))


def acc_subnet(args):
    """ use to compute WS accuracy """

    test_batch_size = 1
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    # device = torch.device('cpu')
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    if args.dataset == 'cifar10':
        _, test_loader = utils.load_CIFAR10(args.batch_size, test_batch_size=test_batch_size, **kwargs)
    elif args.dataset == 'cifar100':
        _, test_loader = utils.load_CIFAR100(args.batch_size, test_batch_size=test_batch_size, **kwargs)

    model = Shar(arch=args.arch, n_party=args.n_party, num_classes=args.num_cls)
    model = model.to(device)
    model.eval()
    if not args.random_init:
        data_path = 'pretrained/{}_checkpoint_best_M{}{}_{}_{}_{}.pth'.format(args.arch, args.method, args.mod, args.lam, args.alpha, args.n_party)
        state_dict = torch.load(data_path, map_location=torch.device('cpu'))['state_dict']
        model.load_state_dict(state_dict)

    corrects = [[0 for _ in range(args.n_party+1)] for _ in range(args.num_cls)]
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output_subs, output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            corrects[0][-1] += pred.eq(target.view_as(pred)).sum().item()
            for i in range(model.n_party):
                for j in range(args.num_cls):
                    kth = args.num_cls - j
                    pred = torch.kthvalue(output_subs[i], kth)[1]
                    corrects[j][i] += pred.eq(target.view_as(pred)).sum().item()

    for j in range(args.num_cls):
        print('\r\nTest set {} largest: '.format(j+1), end='')
        for i in range(model.n_party):
            print('Acc{}: {:.2f}%, '.format(i+1, 100. * corrects[j][i] / len(test_loader.dataset)), end='')
        print('- Acc: {:.2f}%'.format(100. * corrects[j][-1] / len(test_loader.dataset)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--cuda', default=True)
    parser.add_argument('--arch', type=str, default='resnet20')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--num_cls', type=int, default=10)
    parser.add_argument('--n_party', type=int, default=2)
    parser.add_argument('--n_threshold', type=int, default=None)
    parser.add_argument('--mod', type=str, default='cos')
    parser.add_argument('--method', type=int, default=4)
    parser.add_argument('--lam', type=float, default=30)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--random_init', action='store_true')
    parser.add_argument('--save-model', action='store_true', default=True, help='For Saving the current Model')
    args = parser.parse_args()
    if args.n_threshold is None:
        args.n_threshold = args.n_party

    # torch.manual_seed(1)
    cudnn.benchmark = True
    if args.dataset == 'cifar10':
        args.num_cls = 10
    elif args.dataset == 'cifar100':
        args.num_cls = 100

    main(args)
    acc_subnet(args)
    acc_party(args)
