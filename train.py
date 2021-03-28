"""Train Glow on CIFAR-10.
Train script adapted from: https://github.com/kuangliu/pytorch-cifar/
"""
import argparse
import numpy as np
import os
import random
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import util

import models.flow as flow

def build_flow(args, device):
    # Model
    print('Building flow model..')
    flow_net = flow.Glow(num_channels=args.num_channels,
               num_levels=args.num_levels,
               num_steps=args.num_steps)
    flow_net = flow_net.to(device)
    if device == 'cuda':
        flow_net = torch.nn.DataParallel(flow_net, args.gpu_ids)
        cudnn.benchmark = args.benchmark

    start_epoch = 0
    best_loss = 0
    global_step = 0
    
    if args.resume_flow:
        # Load checkpoint.
        print('Resuming from checkpoint at ckpts/best.pth.tar...')
        assert os.path.isdir('ckpts'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('ckpts/best.pth.tar')
        flow_net.load_state_dict(checkpoint['net'])
        best_loss = checkpoint['test_loss']
        start_epoch = checkpoint['epoch']
        global_step = start_epoch * len(trainset)

    loss_fn = util.NLLLoss().to(device)
    optimizer = optim.Adam(flow_net.parameters(), lr=args.lr)
    scheduler = sched.LambdaLR(optimizer, lambda s: min(1., s / args.warm_up))
    
    return flow_net, loss_fn, optimizer, scheduler, start_epoch, best_loss, global_step

def build_ebm(args):
    pass


def main(args):
    # Set up main device and scale batch size
    device = 'cuda' if torch.cuda.is_available() and args.gpu_ids else 'cpu'
    args.batch_size *= max(1, len(args.gpu_ids))

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # No normalization applied, since Glow expects inputs in (0, 1)
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])

    trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    if args.mode == 'flow':
        flow_net, loss_fn, optimizer, scheduler, start_epoch, best_loss, global_step = build_flow(args, device)
        for epoch in range(start_epoch, start_epoch + args.num_epochs):
            flow.train_full(epoch, flow_net, trainloader, device, optimizer, scheduler,
                  loss_fn, args.max_grad_norm, global_step)
            flow.test(epoch, flow_net, testloader, device, loss_fn, args.num_samples)

    elif args.mode == 'ebm':
        pass
    elif args.mode == 'coopNet':
        pass
    

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Glow on CIFAR-10')

    def str2bool(s):
        return s.lower().startswith('t')

    parser.add_argument('--batch_size', default=64, type=int, help='Batch size per GPU')
    parser.add_argument('--benchmark', type=str2bool, default=True, help='Turn on CUDNN benchmarking')
    parser.add_argument('--gpu_ids', default=[0], type=eval, help='IDs of GPUs to use')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=-1., help='Max gradient norm for clipping')
    parser.add_argument('--num_channels', '-C', default=512, type=int, help='Number of channels in hidden layers')
    parser.add_argument('--num_levels', '-L', default=3, type=int, help='Number of levels in the Glow model')
    parser.add_argument('--num_steps', '-K', default=32, type=int, help='Number of steps of flow in each level')
    parser.add_argument('--num_epochs', default=100, type=int, help='Number of epochs to train')
    parser.add_argument('--num_samples', default=64, type=int, help='Number of samples at test time')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loader threads')
    parser.add_argument('--resume_flow', type=str2bool, default=False, help='Resume flow from checkpoint')
    parser.add_argument('--resume_ebm', type=str2bool, default=False, help='Resume ebm from checkpoint')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--warm_up', default=500000, type=int, help='Number of steps for lr warm-up')
    parser.add_argument('mode', choices = ['flow', 'ebm', 'coopNet'])

    

    main(parser.parse_args())
