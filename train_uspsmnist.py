import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from data_list import ImageList
import os
from torch.autograd import Variable
import loss as loss_func
import numpy as np
import network


def train(args, model, ad_net, random_layer, train_loader, train_loader1, optimizer, optimizer_ad, epoch, start_epoch,
          method):
    model.train()
    len_source = len(train_loader)
    len_target = len(train_loader1)
    if len_source > len_target:
        num_iter = len_source
    else:
        num_iter = len_target

    for batch_idx in range(num_iter):
        if batch_idx % len_source == 0:
            iter_source = iter(train_loader)
        if batch_idx % len_target == 0:
            iter_target = iter(train_loader1)
        data_source, label_source = iter_source.next()
        data_source, label_source = data_source.cuda(), label_source.cuda()
        data_target, label_target = iter_target.next()
        data_target = data_target.cuda()
        optimizer.zero_grad()
        optimizer_ad.zero_grad()
        feature_source, output_source = model(data_source)
        feature_target, output_target = model(data_target)
        feature = torch.cat((feature_source, feature_target), 0)
        output = torch.cat((output_source, output_target), 0)

        labels_target_fake = torch.max(nn.Softmax(dim=1)(output_target), 1)[1]
        labels = torch.cat((label_source, labels_target_fake))

        loss = nn.CrossEntropyLoss()(output.narrow(0, 0, data_source.size(0)), label_source)
        softmax_output = nn.Softmax(dim=1)(output)
        if epoch > start_epoch:
            entropy = loss_func.Entropy(softmax_output)
            loss += loss_func.CDAN([feature, softmax_output], ad_net, entropy,
                                   network.calc_coeff(num_iter * (epoch - start_epoch) + batch_idx), random_layer)

        loss = loss + args.mdd_weight * loss_func.mdd_digit(
            feature, labels) + args.entropic_weight * loss_func.EntropicConfusion(feature)

        loss.backward()
        optimizer.step()
        if epoch > start_epoch:
            optimizer_ad.step()
        if (batch_idx + epoch * num_iter) % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.4f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * args.batch_size, num_iter * args.batch_size,
                       100. * batch_idx / num_iter, loss.item()))


def test(args, epoch, config, model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        feature, output = model(data)
        test_loss += nn.CrossEntropyLoss()(output, target).item()
        pred = output.data.cpu().max(1, keepdim=True)[1]
        correct += pred.eq(target.data.cpu().view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    log_str = 'epoch:{} Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(epoch,
                                                                                            test_loss, correct,
                                                                                            len(test_loader.dataset),
                                                                                            100. * correct / len(
                                                                                                test_loader.dataset))
    config["out_file"].write(log_str)
    config["out_file"].flush()
    print(log_str)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='CDAN USPS MNIST')
    parser.add_argument('--method', type=str, default='CDAN-E', choices=['CDAN', 'CDAN-E', 'DANN'])
    parser.add_argument('--task', default='MNIST2USPS', help='MNIST2USPS or MNIST2USPS')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=1000,
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--gpu_id', default='0', type=str,
                        help='cuda device id')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--random', type=bool, default=False,
                        help='whether to use random')
    parser.add_argument('--mdd_weight', type=float, default=0.05)
    parser.add_argument('--entropic_weight', type=float, default=0)
    parser.add_argument("--use_seed", type=bool, default=True)
    args = parser.parse_args()
    import random
    if (args.use_seed):
        torch.manual_seed(args.seed)

        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
    import os.path as osp
    import datetime
    config = {}
    config["output_path"] = "snapshot/" + args.task
    config['seed'] = args.seed
    config["torch_seed"] = torch.initial_seed()
    config["torch_cuda_seed"] = torch.cuda.initial_seed()

    config["mdd_weight"] = args.mdd_weight
    config["entropic_weight"] = args.entropic_weight
    if not osp.exists(config["output_path"]):
        os.system('mkdir -p ' + config["output_path"])
    config["out_file"] = open(osp.join(config["output_path"], "log_{}_{}.txt".
                                       format(args.task, str(datetime.datetime.utcnow()))),
                              "w")

    torch.manual_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if args.task == 'USPS2MNIST':
        source_list = 'data/usps2mnist/usps_train.txt'
        target_list = 'data/usps2mnist/mnist_train.txt'
        test_list = 'data/usps2mnist/mnist_test.txt'
        start_epoch = 1
        decay_epoch = 6
    elif args.task == 'MNIST2USPS':
        source_list = 'data/usps2mnist/mnist_train.txt'
        target_list = 'data/usps2mnist/usps_train.txt'
        test_list = 'data/usps2mnist/usps_test.txt'
        start_epoch = 1
        decay_epoch = 5
    else:
        raise Exception('task cannot be recognized!')

    train_loader = torch.utils.data.DataLoader(
        ImageList(open(source_list).readlines(), transform=transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]), mode='L'),
        batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)
    train_loader1 = torch.utils.data.DataLoader(
        ImageList(open(target_list).readlines(), transform=transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]), mode='L'),
        batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        ImageList(open(test_list).readlines(), transform=transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]), mode='L'),
        batch_size=args.test_batch_size, shuffle=True, num_workers=1)

    model = network.LeNet()
    model = model.cuda()
    class_num = 10

    if args.random:
        random_layer = network.RandomLayer([model.output_num(), class_num], 500)
        ad_net = network.AdversarialNetwork(500, 500)
        random_layer.cuda()
    else:
        random_layer = None
        ad_net = network.AdversarialNetwork(model.output_num() * class_num, 500)
    ad_net = ad_net.cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.0005, momentum=0.9)
    optimizer_ad = optim.SGD(ad_net.parameters(), lr=args.lr, weight_decay=0.0005, momentum=0.9)
    config["out_file"].write(str(config) + "\n")
    config["out_file"].flush()
    for epoch in range(1, args.epochs + 1):
        if epoch % decay_epoch == 0:
            for param_group in optimizer.param_groups:
                param_group["lr"] = param_group["lr"] * 0.5
        train(args, model, ad_net, random_layer, train_loader, train_loader1, optimizer, optimizer_ad, epoch,
              start_epoch, args.method)
        test(args, epoch, config, model, test_loader)


if __name__ == '__main__':
    main()
