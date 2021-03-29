import torch
from tqdm import tqdm
import util
from models.EBM.sampler import sample
import os

@torch.enable_grad()
def train_full(epoch, net, trainloader, device, optimizer, scheduler):
    print('\nEpoch: %d' % epoch)
    net.train()
    loss_meter = util.AverageMeter()
    with tqdm(total=len(trainloader.dataset)) as progress_bar:
        for x, _ in trainloader:
            x = x.to(device)
            optimizer.zero_grad()
            x_q = sample(net, m=64, n_ch=3, im_w=32, im_h=32, K=100, device=device)
            loss = net(x_q).mean() - net(x_p_d).mean()
            loss_meter.update(loss.item(), x.size(0))
            loss.backward()
 
            optimizer.step()
            if scheduler != None:
                scheduler.step()

            progress_bar.set_postfix(nll=loss_meter.avg,
                                     bpd=util.bits_per_dim(x, loss_meter.avg),
                                     lr=optimizer.param_groups[0]['lr'])
            progress_bar.update(x.size(0))
          
@torch.enable_grad()
def train_single_step(net, x, device, optimizer, p_0=None):
    net.train()
    x = x.to(device)
    optimizer.zero_grad()
    x_q = sample(net, m=64, n_ch=3, im_sz=32, im_sz=32, K=100, device, p_0)
    loss = net(x_q).mean() - net(x_p_d).mean()
    loss_meter.update(loss.item(), x.size(0))
    loss.backward()
    optimizer.step()
    
@torch.no_grad()
def test(epoch, net, testloader, device, num_samples, best_loss):
    net.eval()
    loss_meter = util.AverageMeter()
    with tqdm(total=len(testloader.dataset)) as progress_bar:
        for x, _ in testloader:
            x = x.to(device)
            x_q = sample(net, m=64, n_ch=3, im_sz=32, im_sz=32, K=100, device)
            loss = net(x_q).mean() - net(x_p_d).mean()
            loss_meter.update(loss.item(), x.size(0))
            progress_bar.set_postfix(nll=loss_meter.avg,
                                     bpd=util.bits_per_dim(x, loss_meter.avg))
            progress_bar.update(x.size(0))

    # Save checkpoint
    if loss_meter.avg < best_loss:
        print('Saving...')
        state = {
            'net': net.state_dict(),
            'test_loss': loss_meter.avg,
            'epoch': epoch,
        }
        os.makedirs('ckpts', exist_ok=True)
        torch.save(state, 'ckpts/best_ebm.pth.tar')
        best_loss = loss_meter.avg

    # Save samples and data
    images = sample(net, m=64, n_ch=3, im_sz=32, im_sz=32, K=100, device)
    os.makedirs('ebm_samples', exist_ok=True)
    images_concat = torchvision.utils.make_grid(images, nrow=int(num_samples ** 0.5), padding=2, pad_value=255)
    torchvision.utils.save_image(images_concat, 'ebm_samples/epoch_{}.png'.format(epoch))
    
    return best_loss
