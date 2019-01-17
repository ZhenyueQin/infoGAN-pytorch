import numpy as np

a = np.random.multinomial(20, [1/6] * 6, size=1)

print(a)