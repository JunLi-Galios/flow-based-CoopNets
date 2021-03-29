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
from tqdm import tqdm

import models.flow as flow
import models.EBM as ebm


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
    flow_best_loss = 0
    
    if args.resume_flow:
        # Load checkpoint.
        print('Resuming from checkpoint at ckpts/best.pth.tar...')
        assert os.path.isdir('ckpts'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('ckpts/best.pth.tar')
        flow_net.load_state_dict(checkpoint['net'])
        flow_best_loss = checkpoint['test_loss']
        start_epoch = checkpoint['epoch']

    loss_fn = util.NLLLoss().to(device)
    optimizer = optim.Adam(flow_net.parameters(), lr=args.lr)
    scheduler = sched.LambdaLR(optimizer, lambda s: min(1., s / args.warm_up))
    
    return flow_net, loss_fn, optimizer, scheduler, start_epoch, flow_best_loss

def build_ebm(args, device):
    
    print('Building ebm model..')
    ebm_net = ebm.F(n_c=3, n_f=64)
    ebm_net = ebm_net.to(device)
    if device == 'cuda':
        ebm_net = torch.nn.DataParallel(ebm_net, args.gpu_ids)
        cudnn.benchmark = args.benchmark

    start_epoch = 0
    ebm_best_loss = 0
    
    if args.resume_ebm:
        # Load checkpoint.
        print('Resuming from checkpoint at ckpts/ebm_best.pth.tar...')
        assert os.path.isdir('ckpts'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('ckpts/ebm_best.pth.tar')
        ebm_net.load_state_dict(checkpoint['net'])
        ebm_best_loss = checkpoint['test_loss']
        start_epoch = checkpoint['epoch']
        
    optimizer = torch.optim.Adam(ebm_net.parameters(), lr=1e-4, betas=[.9, .999])    
    scheduler = None
        
    return ebm_net, optimizer, scheduler, start_epoch, ebm_best_loss


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
        flow_net, loss_fn, optimizer, scheduler, start_epoch, flow_best_loss = build_flow(args, device)
        for epoch in range(start_epoch, start_epoch + args.num_epochs):
            flow.train_full(epoch, flow_net, trainloader, device, optimizer, scheduler,
                  loss_fn, args.max_grad_norm)
            flow_best_loss = flow.test(epoch, flow_net, testloader, device, loss_fn, args.num_samples, flow_best_loss)

    elif args.mode == 'ebm':
        ebm_net, optimizer, scheduler, start_epoch, ebm_best_loss = build_ebm(args, device)
        for epoch in range(start_epoch, start_epoch + args.num_epochs):
            ebm.train_full(epoch, ebm_net, trainloader, device, optimizer, scheduler)
            ebm_best_loss = ebm.test(epoch, ebm_net, testloader, device, args.num_samples, ebm_best_loss)
    elif args.mode == 'coopNet':
        flow_net, flow_loss_fn, flow_optimizer, flow_scheduler, flow_start_epoch, flow_best_loss = build_flow(args, device)
        ebm_net, ebm_optimizer, ebm_scheduler, ebm_start_epoch, ebm_best_loss = build_ebm(args, device)
        
        for epoch in range(ebm_start_epoch, ebm_start_epoch + args.num_epochs):
            with tqdm(total=len(trainloader.dataset)) as progress_bar:
                for x, _ in trainloader:
                    x = x.to(device)

                    # train flow for single step
                    flow.train_single_step(flow_net, x, device, flow_optimizer, flow_loss_fn, args.max_grad_norm)

                    # sample from flow
                    x_f = flow.sample(flow_net, args.batch_size, device)

                    # train ebm for single step
                    x_e = ebm.train_single_step(ebm_net, x, device, ebm_optimizer, x_f)

                    # train flow with samples from ebm
                    x_e = (x_e - x_e.min()) / (x_e.max() - x_e.min() + 1e-5)
                    flow.train_single_step(flow_net, x_e, device, flow_optimizer, flow_loss_fn, args.max_grad_norm)

                    if flow_scheduler != None:
                        flow_scheduler.step()
                    if ebm_scheduler != None:
                        ebm_scheduler.step()

            ebm_best_loss = ebm.test(epoch, ebm_net, testloader, device, args.num_samples, ebm_best_loss)
                
       

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
