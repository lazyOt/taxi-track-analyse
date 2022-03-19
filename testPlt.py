from scipy.cluster.vq import kmeans, vq, whiten
import numpy as np
import matplotlib.pyplot as plt

K = 3
spott = []
for i in range(500):
    spott.append((np.random.randint(50), np.random.randint(50)))
spott = np.array(spott)
print(spott.shape)