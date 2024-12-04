import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from cycler import cycler
from calc_dvh import calc_log_dvh_with_log, calc_dvh
from scipy.interpolate import interp1d
import icecream as ic


area = 1 # np.pi*2*0.2 # 円の表面積の場合np.pi*0.5*0.5
energy = 150*(10**-3)*667*area*100

with open("/home/raise/mcx_simulation/analysis_sensitivity/test_res.mc2", 'rb') as f:
    data = np.fromfile(f, dtype=np.float32).reshape([600,600,600,], order='F')
    data *= energy

with open("/home/raise/mcx_simulation/analysis_sensitivity/089_tumour_model.bin", "rb") as f:
    model = np.fromfile(f, dtype=np.uint8).reshape([600,600,600,], order='F')


nx, ny, nz = [248, 416, 384]


fig, axs = plt.subplots(1, 2, figsize=(15, 10))
axs[0].imshow(model[nx,:,:],origin="lower",cmap='Oranges')
axs[0].set_title("origin")

a = axs[1].imshow(np.where(data[nx,:, :] >= 10**-2, data[nx,:, :], np.nan),origin="lower", norm=LogNorm(vmin=10**-2))
axs[1].set_title("tumour")
divider = make_axes_locatable(axs[1]) #axに紐付いたAxesDividerを取得
cax = divider.append_axes("right", size="5%", pad=0.1) #append_axesで新しいaxesを作成
cbar = fig.colorbar(a, cax=cax)  # カラーバーのサイズをaxs[0]のサイズに合わせる

plt.show()


fig, axs = plt.subplots(1, 2, figsize=(15, 10))
axs[0].imshow(model[:,ny,:],origin="lower",cmap='Oranges')
axs[0].set_title("origin")

a = axs[1].imshow(np.where(data[:,ny, :] > 10**-2, data[:,ny, :], np.nan),origin="lower", norm=LogNorm(vmin=10**-2))
axs[1].set_title("tumour")
divider = make_axes_locatable(axs[1]) #axに紐付いたAxesDividerを取得
cax = divider.append_axes("right", size="5%", pad=0.1) #append_axesで新しいaxesを作成
cbar = fig.colorbar(a, cax=cax)  # カラーバーのサイズをaxs[0]のサイズに合わせる
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(15, 10))
axs[0].imshow(model[:,:,nz],origin="lower",cmap='Oranges')
axs[0].set_title("origin")

a = axs[1].imshow(np.where(data[:,:, nz] >= 10**-2, data[:,:, nz], np.nan),origin="lower", norm=LogNorm(vmin=10**-2))
axs[1].set_title("tumour")
divider = make_axes_locatable(axs[1]) #axに紐付いたAxesDividerを取得
cax = divider.append_axes("right", size="5%", pad=0.1) #append_axesで新しいaxesを作成
cbar = fig.colorbar(a, cax=cax)  # カラーバーのサイズをaxs[0]のサイズに合わせる
plt.show()

cum_rel_dvh_t, cum_rel_dvh_b, bincenter = calc_dvh(data, model)
# plt.plot(bincenter, cum_rel_dvh_b, label='Bronchi')
plt.plot(bincenter, cum_rel_dvh_t, label='Tumour')
plt.xlabel('Dose [J/cm2]')
plt.ylabel('Volume [%]')
plt.legend()
plt.show()


c = plt.get_cmap("Set2")
colors = c(np.arange(0, c.N))
# colors = [c(1.*i/7) for i in range(3)]
plt.rcParams["axes.prop_cycle"] = cycler("color", colors)

cum_rel_dvh_t, cum_rel_dvh_b, cum_rel_dvh_l, bin_edges = calc_log_dvh_with_log(data, model)
plt.plot(bin_edges[:-1], cum_rel_dvh_b, label='Bronchi')
plt.plot(bin_edges[:-1], cum_rel_dvh_t, label='Tumour')
plt.plot(bin_edges[:-1], cum_rel_dvh_l, label='Lung Tissue')
plt.xlabel('Dose [J/cm2]')
plt.ylabel('Volume [%]')
plt.xscale('log')
plt.legend()
plt.show()


Dx = interp1d(cum_rel_dvh_t, bin_edges[:-1])
d90 = ic.ic(Dx([90]))
d50 = ic.ic(Dx([50]))
d10 = ic.ic(Dx([10]))

Dx = interp1d(cum_rel_dvh_l, bin_edges[:-1])
# d99 = ic.ic(Dx([99]))
d90 = ic.ic(Dx([90]))
d50 = ic.ic(Dx([50]))
d10 = ic.ic(Dx([10]))

plt.show()


# cover_rate
cover_rate_100 = ic.ic(np.count_nonzero((model == 3) & (data >= 100)) / np.count_nonzero(model == 3) * 100)
cover_rate_10 = ic.ic(np.count_nonzero((model == 3) & (data >= 10)) / np.count_nonzero(model == 3) * 100)
cover_rate_1 = ic.ic(np.count_nonzero((model == 3) & (data >= 1)) / np.count_nonzero(model == 3) * 100)


"""
data[data==0] = 4
data[data == 1] = 5
data[data==2] = 6

data[data==4] = 1
data[data==5] = 2
data[data==6] = 3 
data.tofile('/mnt/e/dataset/LUNG1-089/089_tumour_model.bin')
"""
