import torch.nn as nn


class Q(nn.Module):
    def __init__(self, disc_lat_dim, conv_lat_dim):
        super(Q, self).__init__()
        self.disc_lat_dim = disc_lat_dim
        self.conv_lat_dim = conv_lat_dim

        self.conv = nn.Conv2d(1024, 128, 1, bias=False)
        self.bn = nn.BatchNorm2d(128)
        self.lReLU = nn.LeakyReLU(0.1, inplace=True)
        self.conv_disc = nn.Conv2d(128, self.lat_dim, 1)
        self.conv_mu = nn.Conv2d(128, self.conv_lat_dim, 1)
        self.conv_var = nn.Conv2d(128, self.conv_lat_dim, 1)

    def forward(self, x):
        y = self.conv(x)
        disc_logits = self.softmax(self.conv_disc(y).squeeze())

        mu = self.softmax(self.conv_mu(y).squeeze())
        var = self.conv_var(y).squeeze().exp()

        return disc_logits, mu, var


class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(1024, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.main(x).view(-1, 1)
        return output


class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(74, 1024, 1, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 128, 7, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.main(x)
        return output


class FrontEnd(nn.Module):
    """ front end part of discriminator and Q"""
    def __init__(self):
        super(FrontEnd, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 1024, 7, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        output = self.main(x)
        return output
