import torch

@torch.no_grad()
def sample(net, m=64, n_ch=3, im_sz=32, im_sz=32, K=100, device, p_0=None):
    if p_0 == None:
        sample_p_0 = lambda: torch.FloatTensor(m, n_ch, im_sz, im_sz).uniform_(-1, 1).to(device)
    else:
        sample_p_0 = lambda: p_0.uniform_(-1, 1).to(device)
    x_k = torch.autograd.Variable(sample_p_0(), requires_grad=True)
    for k in range(K):
        net_prime = torch.autograd.grad(net(x_k).sum(), [x_k], retain_graph=True)[0]
        x_k.data += net_prime + 1e-2 * torch.randn_like(x_k)
    return x_k.detach()
