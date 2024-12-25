from matplotlib import pyplot as plt
import numpy as np


file = "/mnt/c/Users/mbpl/Desktop/ToMorizane/human_lung_air.mc2"
with open(file, "rb") as f:
    model = np.fromfile(f, dtype=np.float32).reshape((600, 600, 600), order='F')

plt.imshow(model[299, :, :], origin="lower")
plt.show()

file = "/mnt/c/Users/mbpl/Desktop/ToMorizane/discT.bin"
with open(file, "rb") as f:
    model = np.fromfile(f, dtype=np.uint8).reshape((600, 600, 600), order='F')

plt.imshow(model[246, :, :])
plt.show()