import torch

@torch.enable_grad()
def sample(net, m=64, n_ch=3, im_w=32, im_h=32, K=10, device='cpu', p_0=None):
    if p_0 is None:
        sample_p_0 = lambda: torch.FloatTensor(m, n_ch, im_w, im_h).uniform_(-1, 1).to(device)
    else:
        sample_p_0 = lambda: p_0.uniform_(-1, 1).to(device)
    x_k = torch.autograd.Variable(sample_p_0(), requires_grad=True)
    for k in range(K):
        net_prime = torch.autograd.grad(net(x_k).sum(), [x_k], retain_graph=True)[0]
        x_k.data += net_prime + 1e-2 * torch.randn_like(x_k)
    return x_k.detach()
