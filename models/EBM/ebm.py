import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.convt1 = nn.ConvTranspose2d(opt.z_size, 512, kernel_size=4, stride=1, padding=0)
        self.convt2 = nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.convt3 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.convt4 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.convt5 = nn.ConvTranspose2d(64, 3, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)
        self.leakyrelu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def forward(self, z):
        self.z = z
        out = self.convt1(z)
        out = self.bn1(out)
        out = self.leakyrelu(out)
        out = self.convt2(out)
        out = self.bn2(out)
        out = self.leakyrelu(out)
        out = self.convt3(out)
        out = self.bn3(out)
        out = self.leakyrelu(out)
        out = self.convt4(out)
        out = self.bn4(out)
        out = self.leakyrelu(out)
        out = self.convt5(out)
        out = self.tanh(out)
        return out
