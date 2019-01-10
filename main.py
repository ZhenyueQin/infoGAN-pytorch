from model import *
from trainer import Trainer
import torch

has_cuda = torch.cuda.is_available()
device = torch.device("cuda" if has_cuda else "cpu")

fe = FrontEnd()
d = D()
q = Q()
g = G()
q_2 = Q_2()

for i in [fe, d, q, g]:
  i.to(device)
  i.apply(weights_init)

trainer = Trainer(g, fe, d, q, q_2, to_inverse=False)
trainer.train()
