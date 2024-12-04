import numpy as np
from matplotlib import pyplot as plt
from cylinder import create_rotated_cylinder_mask
from matplotlib.colors import LogNorm
from calc_dvh import calc_dvh
from scipy.interpolate import interp1d

with open('/mnt/e/dataset/LUNG1-089/089_tumour_model.bin', 'rb') as f:
    data = np.fromfile(f, dtype=np.uint8).reshape((600,600,600), order='F')
    
with open('test_res.mc2', 'rb') as f:
    res = np.fromfile(f, dtype=np.float32).reshape((600,600,600), order='F')  


energy = 150*(10**-3)*667*np.pi*0.5*0.5*100
res *= energy 


cylinder_center0 = [248, 416, 384]  # 中心の座標
cylinder_direction0 = [15, -10, 95]

cylinder_mask_0 = create_rotated_cylinder_mask([600,600,600], cylinder_center0, 1, 22, cylinder_direction0)


data[cylinder_mask_0==1] = 4
x = 248
y = 416
z = 384


cum_rel_dvh_t, cum_rel_dvh_b, bincenter = calc_dvh(res, data)

Dx = interp1d(cum_rel_dvh_t, bincenter)
D95 = Dx([95])
print(D95)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(res[x,:,:], origin="lower")
axs[1].imshow(data[x,:,:], origin="lower")
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(res[:,y,:], origin="lower", norm=LogNorm())
axs[1].imshow(data[:,y,:], origin="lower")
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(res[:,:,z], origin="lower", norm=LogNorm())
axs[1].imshow(data[:,:,z], origin="lower")
plt.show()
