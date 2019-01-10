import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd as autograd

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import numpy as np
import general_methods as gm
import os
import cv2

# import np_plotter as np_plt

class log_gaussian:
    def __call__(self, x, mu, var):

        logli = -0.5*(var.mul(2*np.pi)+1e-6).log() - \
                (x-mu).pow(2).div(var.mul(2.0)+1e-6)

        return logli.sum(1).mean().mul(-1)


class Trainer:
    def __init__(self, G, FE, D, Q, Q_2, to_inverse):

        has_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if has_cuda else "cpu")
        self.G = G.to(self.device)
        self.FE = FE.to(self.device)
        self.D = D.to(self.device)
        self.Q = Q.to(self.device)
        self.Q_2 = Q_2.to(self.device)
        self.batch_size = 100
        self.current_time = gm.get_current_time()
        self.to_inverse = to_inverse
        self.saving_path = './results/' + str(self.to_inverse) + '/' + self.current_time
        if not os.path.exists(self.saving_path):
            os.makedirs(self.saving_path)
        self.transform_pad = 4
        self.img_size = 28



    def _noise_sample(self, dis_c, con_c, noise, bs):

        idx = np.random.randint(10, size=bs)
        c = np.zeros((bs, 10))
        c[range(bs),idx] = 1.0

        idx_2 = np.random.randint(2, size=bs)
        c_2 = np.zeros((bs, 2))
        c_2[range(bs), idx_2] = 1.0
        # print('c_2: ', c_2)

        dis_c.data.copy_(torch.Tensor(c))
        con_c.data.copy_(torch.Tensor(c_2))
        noise.data.uniform_(-1.0, 1.0)
        z = torch.cat([noise, dis_c, con_c], 1).view(-1, 74, 1, 1)

        return z, idx, idx_2

    def train(self):

        real_x = torch.FloatTensor(self.batch_size, 1, 28, 28).to(self.device)
        label = torch.FloatTensor(self.batch_size, 1).to(self.device)
        dis_c = torch.FloatTensor(self.batch_size, 10).to(self.device)
        con_c = torch.FloatTensor(self.batch_size, 2).to(self.device)
        size_c = torch.FloatTensor(self.batch_size, 2).to(self.device)
        noise = torch.FloatTensor(self.batch_size, 62).to(self.device)

        real_x = Variable(real_x)
        label = Variable(label, requires_grad=False)
        dis_c = Variable(dis_c)
        con_c = Variable(con_c)
        size_c = Variable(size_c)
        noise = Variable(noise)

        criterionD = nn.BCELoss().to(self.device)
        criterionQ_dis = nn.CrossEntropyLoss().to(self.device)
        # criterionQ_con = log_gaussian()
        criterionQ_con = nn.CrossEntropyLoss().to(self.device)

        optimD = optim.Adam([{'params':self.FE.parameters()}, {'params':self.D.parameters()}], lr=0.0002, betas=(0.5, 0.99))
        optimG = optim.Adam([{'params':self.G.parameters()}, {'params':self.Q.parameters()},
                             {'params':self.Q_2.parameters()}], lr=0.001, betas=(0.5, 0.99))

        big_small_transform = transforms.Compose([
            transforms.Pad(self.transform_pad),
            transforms.Scale(self.img_size),
            transforms.ToTensor()
        ])
        dataset = dset.MNIST('./dataset', transform=transforms.ToTensor(), download=True)
        if isinstance(self.to_inverse, bool):
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=1)
        elif self.to_inverse == 'mixed':
            dataloader = DataLoader(dataset, batch_size=int(self.batch_size/2), shuffle=True, num_workers=1)
        elif self.to_inverse == 'big_small':
            print('using big_small')
            original_ds = dset.MNIST('./dataset', transform=transforms.ToTensor(), download=True)
            transformed_ds = dset.MNIST('./dataset', transform=big_small_transform, download=True)
            dataset = torch.utils.data.ConcatDataset([original_ds, transformed_ds])
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=1)

        # fixed random variables
        # c = np.linspace(-1, 1, 10).reshape(1, -1)
        # c = np.repeat(c, 10, 0).reshape(-1, 1)
        #
        # c1 = np.hstack([c, np.zeros_like(c)])
        # c2 = np.hstack([np.zeros_like(c), c])

        idx = np.arange(10).repeat(10)
        one_hot = np.zeros((100, 10))
        one_hot[range(100), idx] = 1
        fix_noise = torch.Tensor(100, 62).uniform_(-1, 1)

        for epoch in range(100):
            for num_iters, batch_data in enumerate(dataloader, 0):

                # real part
                optimD.zero_grad()
                x, _ = batch_data

                # np_plt.plot_a_numpy_array(x[0].data.numpy())

                # print('what is to inverse: ', self.to_inverse)
                if self.to_inverse == 'mixed':
                    x = torch.cat((x, 1 - x), 0)
                elif self.to_inverse == True:
                    print('has come to inverse')
                    x = 1 - x

                bs = x.size(0)
                real_x.data.resize_(x.size())
                label.data.resize_(bs, 1)
                dis_c.data.resize_(bs, 10)
                con_c.data.resize_(bs, 2)
                size_c.data.resize_(bs, 2)
                noise.data.resize_(bs, 62)

                real_x.data.copy_(x)
                fe_out1 = self.FE(real_x)
                probs_real = self.D(fe_out1)
                label.data.fill_(1)
                loss_real = criterionD(probs_real, label)
                loss_real.backward()

                # fake part
                z, idx, idx_2 = self._noise_sample(dis_c, con_c, noise, bs)
                fake_x = self.G(z)
                fe_out2 = self.FE(fake_x.detach())
                probs_fake = self.D(fe_out2)
                label.data.fill_(0)
                loss_fake = criterionD(probs_fake, label)
                loss_fake.backward()

                D_loss = loss_real + loss_fake

                optimD.step()

                # G and Q part
                optimG.zero_grad()

                fe_out = self.FE(fake_x)
                probs_fake = self.D(fe_out)
                label.data.fill_(1.0)

                reconstruct_loss = criterionD(probs_fake, label)

                q_logits, q_mu, q_var = self.Q(fe_out)
                q_logits_2, q_mu_2, q_var_2 = self.Q_2(fe_out)
                class_ = torch.LongTensor(idx).to(self.device)
                class_2_ = torch.LongTensor(idx_2).to(self.device)
                target = Variable(class_)
                target_2 = Variable(class_2_)
                dis_loss = criterionQ_dis(q_logits, target)
                con_loss = 5 * criterionQ_dis(q_logits_2, target_2)

                # print('any dis loss: ', dis_loss)
                # print('any con loss: ', con_loss)

                G_loss = reconstruct_loss + dis_loss + con_loss
                G_loss.backward()
                optimG.step()

                if num_iters % 100 == 0:

                    print('Epoch/Iter:{0}/{1}, Dloss: {2}, Gloss: {3}'.format(
                        epoch, num_iters, D_loss.data.to(self.device).cpu().numpy(),
                        G_loss.data.to(self.device).cpu().numpy())
                    )

                    noise.data.copy_(fix_noise)
                    dis_c.data.copy_(torch.Tensor(one_hot))

                    # con_c.data.copy_(torch.from_numpy(c1))
                    c = np.ones(shape=[self.batch_size, 1])

                    c1 = np.hstack([c, np.zeros_like(c)])
                    c2 = np.hstack([np.zeros_like(c), c])

                    con_c.data.copy_(torch.from_numpy(c1))
                    # print('con_c before: ', con_c)
                    z = torch.cat([noise, dis_c, con_c], 1).view(-1, 74, 1, 1)
                    x_save_1 = self.G(z)
                    save_image(x_save_1.data, self.saving_path + '/c1.png', nrow=10)

                    con_c.data.copy_(torch.from_numpy(c2))
                    # print('con c after: ', con_c)
                    z = torch.cat([noise, dis_c, con_c], 1).view(-1, 74, 1, 1)
                    x_save_2 = self.G(z)
                    print('outputs are the same: ', torch.all(torch.eq(x_save_1, x_save_2)))
                    save_image(x_save_2.data, self.saving_path + '/c2.png', nrow=10)
