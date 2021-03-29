import torch.nn as nn

class F(nn.Module):
    def __init__(self, n_c, n_f, l= 0.2):
        super(F, self).__init__()
        self.f = nn.Sequential(
            nn.Conv2d(n_c, n_f, 3, 1, 1),
            nn.LeakyReLU(l),
            nn.Conv2d(n_f, n_f ∗ 2 , 4, 2, 1),
            nn.LeakyReLU(l),
            nn.Conv2d(n_f ∗ 2, n_f ∗ 4, 4, 2, 1),
            nn.LeakyReLU(l),
            nn.Conv2d(n_f ∗ 4, n_f ∗ 8, 4, 2, 1),
            nn.LeakyReLU(l),
            nn.Conv2d(n_f ∗ 8, 1, 4, 1, 0) 
        )
        
    def forward(self, x):
        return self.f(x).squeeze()
