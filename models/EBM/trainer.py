import torch
from tqdm import tqdm

@torch.enable_grad()
def train_full(epoch, net, trainloader, device, optimizer, scheduler, global_step, sampler):
    print('\nEpoch: %d' % epoch)
    net.train()
    loss_meter = util.AverageMeter()
    with tqdm(total=len(trainloader.dataset)) as progress_bar:
        for x, _ in trainloader:
            x = x.to(device)
            optimizer.zero_grad()
            x_q = sampler.sample()
            loss = f(x_q).mean() - f(x_p_d).mean()
            loss_meter.update(loss.item(), x.size(0))
            loss.backward()
 
            optimizer.step()
            scheduler.step(global_step)

            progress_bar.set_postfix(nll=loss_meter.avg,
                                     bpd=util.bits_per_dim(x, loss_meter.avg),
                                     lr=optimizer.param_groups[0]['lr'])
            progress_bar.update(x.size(0))
            global_step += x.size(0)
          
@torch.enable_grad()
def train_single_step(net, x, device, optimizer, sampler):
    net.train()
    x = x.to(device)
    optimizer.zero_grad()
    x_q = sampler.sample()
    loss = f(x_q).mean() - f(x_p_d).mean()
    loss_meter.update(loss.item(), x.size(0))
    loss.backward()
    optimizer.step()
