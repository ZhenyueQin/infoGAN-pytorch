import torch
import numpy as np
from model import G


class HigginsDisentanglementMetric:
    def __init__(self, dis, G, L):
        self.G = G
        self.G.eval()
        self.L = L
        self.dis = dis
        self.fix_noise = np.tile(np.linspace(-1, 1, num=5), 12)
        self.fix_noise = torch.from_numpy(
            np.tile(np.concatenate([self.fix_noise, np.linspace(-1, 1, num=2)]), 100).reshape([-1, 62]))
        addons = np.linspace(-0.5, 0.5, num=10)
        for i in range(len(self.fix_noise)):
            self.fix_noise[i] += addons[i % 10]


    def get_a_z_diff(self, k):
        v_1 = self.torch.randn(self.dis.shape)
        v_2 = self.torch.randn(self.dis.shape)

        v_1[k] = v_2[k] = self.dis[k]

        x_1 = self.a_VAE.decode(v_1)
        x_2 = self.a_VAE.decode(v_2)

        mu_1, logvar_1 = self.a_VAE.encode(x_1.view(-1, 64 * 64))
        z_1 = self.a_VAE.reparameterize(mu_1, logvar_1)

        mu_2, logvar_2 = self.a_VAE.encode(x_2.view(-1, 64 * 64))
        z_2 = self.a_VAE.reparameterize(mu_2, logvar_2)

        z_diff = np.abs((z_1 - z_2).data.numpy())
        return z_diff

    def get_L_z_diffs(self, k):
        z_diffs = np.expand_dims(np.zeros(shape=self.z.shape), axis=0)
        for l in range(self.L):
            z_diffs += self.get_a_z_diff(k)
        return z_diffs / self.L

    def iterate_all_z_dims(self):
        accuracy = 0
        for k in range(len(self.z)):
            z_diffs = self.get_L_z_diffs(k)
            predicted_k = np.argmin(z_diffs.squeeze())
            if predicted_k == k:
                accuracy += 1
        return accuracy / len(self.z)

