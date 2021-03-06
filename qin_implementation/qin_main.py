from models import *
from training import Learner
import torch

has_cuda = torch.cuda.is_available()
device = torch.device("cuda" if has_cuda else "cpu")

fe = FrontEnd()
d = D()
q = Q()
g = G()

for i in [fe, d, q, g]:
  i.to(device)
  # i.apply(weights_init)

trainer = Learner(g, fe, d, q, to_inverse=False, is_debugging=False)
trainer.train()
