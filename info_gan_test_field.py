import torch
from model import D, FrontEnd, G, Q
import numpy as np
from torch.autograd import Variable
from torchvision.utils import save_image
import os

has_cuda = torch.cuda.is_available()
model_path = './results/False/2019-01-16-15-57-41'

device = torch.device("cuda" if has_cuda else "cpu")
b_sz = 100

# torch.manual_seed(1)


def _noise_sample(dis_c, con_c, noise, bs):
    idx = np.random.randint(10, size=bs)
    c = np.zeros((bs, 10))
    c[range(bs), idx] = 1.0

    idx_2 = np.random.randint(2, size=bs)
    c_2 = np.zeros((bs, 2))
    c_2[range(bs), idx_2] = 1.0
    # print('c_2: ', c_2)

    dis_c.data.copy_(torch.Tensor(c))
    con_c.data.copy_(torch.Tensor(c_2))
    noise.data.uniform_(-1.0, 1.0)
    print('noise: ', noise.shape)
    z = torch.cat([noise, dis_c, con_c], 1).view(-1, 74, 1, 1)

    return z, idx, idx_2


model_Q = Q().to(device)
model_FE = FrontEnd().to(device)
model_G = G().to(device)
model_D = D().to(device)

model_Q.load_state_dict(torch.load(model_path + '/model_Q.pytorch', map_location='cpu'))
model_D.load_state_dict(torch.load(model_path + '/model_D.pytorch', map_location='cpu'))
model_FE.load_state_dict(torch.load(model_path + '/model_FE.pytorch', map_location='cpu'))
model_G.load_state_dict(torch.load(model_path + '/model_G.pytorch', map_location='cpu'))

model_Q.eval()
model_D.eval()
model_FE.eval()
model_G.eval()

dis_c = torch.FloatTensor(b_sz, 10).to(device)
con_c = torch.FloatTensor(b_sz, 2).to(device)
noise = torch.FloatTensor(b_sz, 62).to(device)

dis_c = Variable(dis_c)
con_c = Variable(con_c)
noise = Variable(noise)

dis_c.data.resize_(b_sz, 10)
con_c.data.resize_(b_sz, 2)
noise.data.resize_(b_sz, 62)

fix_noise = np.tile(np.linspace(-1, 1, num=5), 12)
fix_noise = torch.from_numpy(np.tile(np.concatenate([fix_noise, np.linspace(-1, 1, num=2)]), 100).reshape([-1, 62]))
# fix_noise = torch.Tensor(100, 62).uniform_(-1, 1)
# print(fix_noise.shape)
addons = np.linspace(-0.5, 0.5, num=10)
print('len(fix_noise): ', len(fix_noise))
for i in range(len(fix_noise)):
    fix_noise[i] += addons[i % 10]
print(fix_noise)

fix_idx = np.arange(10).repeat(10)
one_hot = np.zeros((100, 10))
one_hot[range(100), fix_idx] = 1
noise.data.copy_(fix_noise)
dis_c.data.copy_(torch.Tensor(one_hot))

c = np.ones(shape=[b_sz, 1])
c1 = np.hstack([c, np.zeros_like(c)])

con_c.data.copy_(torch.from_numpy(c1))
z = torch.cat([noise, dis_c, con_c], 1).view(-1, 74, 1, 1)

x_save = model_G(z)
print('x_save.data: ', x_save.data.shape)

fname = '%s/rec/rec_test.png' % (model_path)
if not os.path.exists(os.path.dirname(fname)):
    os.makedirs(os.path.dirname(fname))
save_image(x_save.data, fname, nrow=10)

# noise.data.copy_(fix_noise)
# dis_c.data.copy_(torch.Tensor(one_hot))
#
# # con_c.data.copy_(torch.from_numpy(c1))
# c = np.ones(shape=[self.batch_size, 1])
#
# c1 = np.hstack([c, np.zeros_like(c)])
# c2 = np.hstack([np.zeros_like(c), c])
#
# con_c.data.copy_(torch.from_numpy(c1))
# # print('con_c before: ', con_c)
# z = torch.cat([noise, dis_c, con_c], 1).view(-1, 74, 1, 1)
#
# x_save_1 = self.G(z)
# save_image(x_save_1.data, self.saving_path +
#            '/epoch_' + str(epoch) + '_ite_' + str(num_iters) + 'c1.png', nrow=10)
#
# con_c.data.copy_(torch.from_numpy(c2))
# # print('con c after: ', con_c)
# z = torch.cat([noise, dis_c, con_c], 1).view(-1, 74, 1, 1)
# x_save_2 = self.G(z)
# # print('outputs are the same: ', torch.all(torch.eq(x_save_1, x_save_2)))
# save_image(x_save_2.data, self.saving_path +
#            '/epoch_' + str(epoch) + '_ite_' + str(num_iters) + 'c2.png', nrow=10)
