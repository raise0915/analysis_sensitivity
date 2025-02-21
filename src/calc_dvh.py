
import numpy as np
from matplotlib import pyplot as plt

def calc_dvh(res, model):
    bin_w = 0.001
    
    size = np.arange(200*200*200, dtype=np.float32).reshape(200, 200, 200)
    tumour_mask = np.full_like(size, True)
    tumour_mask[model == 3] = False

    bronchi_mask = np.full_like(size, True)
    bronchi_mask[model == 2] = False
    
    tumour = np.ma.array(res, mask=tumour_mask)
    bronchi = np.ma.array(res, mask=bronchi_mask)
    
    # bin数導出
    num_of_bins = int(100/bin_w) + 1
    
    # 累DVH
    hist_total_t = np.zeros(num_of_bins)
    hist_total_b = np.zeros(num_of_bins)
    
    for plane in range(200):
        hist, edge  = np.histogram(
            tumour[:, :, plane].compressed(), bins=num_of_bins, range=(0, num_of_bins*bin_w)
        )
        hist_total_t += hist

    for plane in range(200):
        hist, edge_b  = np.histogram(
            bronchi[:, :, plane].compressed(), bins=num_of_bins, range=(0, num_of_bins*bin_w)
        )
        hist_total_b += hist
    bincenter = [(edge[i] + edge[i+1])/2 for i in range(edge.size - 1)]
    bincenter_b = [(edge_b[i] + edge_b[i+1])/2 for i in range(edge_b.size - 1)]

    volume_t = hist_total_t.sum()
    cum_dvh_t = hist_total_t.sum() - hist_total_t.cumsum()
    cum_rel_dvh_t = cum_dvh_t / volume_t * 100

    volume_b = hist_total_b.sum()
    cum_dvh_b = hist_total_b.sum() - hist_total_b.cumsum()
    cum_rel_dvh_b = cum_dvh_b / volume_b * 100 
    
    plt.xlabel('Dose [J/cm2]')
    plt.ylabel('Volume [%]')
    # plt.plot(bincenter_b, cum_rel_dvh_b)
    # plt.plot(bincenter, cum_rel_dvh_t)
    # plt.show()
    
    return cum_rel_dvh_t, cum_rel_dvh_b, bincenter

def calc_log_dvh_with_log(res, model):
    def calc_rel_dvh(hist_total: np.ndarray):
        volume = hist_total.sum()
        cum_dvh = volume - hist_total.cumsum()
        cum_rel_dvh = cum_dvh / volume * 100
        return cum_rel_dvh
    
    # bin数導出
    bin_max = 1e+2
    bin_min = 1e-7
    num_of_bins = int(np.log10(bin_max) - np.log10(bin_min)) * 10  # bin_minからbin_maxまで10の累乗ごとのbinを作成

    size = np.zeros((200, 200, 200), dtype=np.float32)
    tumour_mask = np.full_like(size, True, dtype=bool)
    tumour_mask[model == 3] = False

    bronchi_mask = np.full_like(size, True, dtype=bool)
    bronchi_mask[model == 2] = False

    lung_tissue = np.full_like(size, True, dtype=bool)
    lung_tissue[model == 1] = False
    
    tumour = np.ma.array(res, mask=tumour_mask)
    bronchi = np.ma.array(res, mask=bronchi_mask)
    lung_tissue = np.ma.array(res, mask=lung_tissue)

    # 累DVH
    hist_total_t = np.zeros(num_of_bins)
    hist_total_b = np.zeros(num_of_bins)
    hist_total_l = np.zeros(num_of_bins)

    bin_edges = np.logspace(np.log10(bin_min), np.log10(bin_max), num_of_bins + 1)

    for plane in range(200):
        hist, _  = np.histogram(
            tumour[:, :, plane].compressed(), bins=bin_edges
        )
        hist_total_t += hist

    for plane in range(200):
        hist, _  = np.histogram(
            bronchi[:, :, plane].compressed(), bins=bin_edges
        )
        hist_total_b += hist

    for plane in range(200):
        hist, _  = np.histogram(
            lung_tissue[:, :, plane].compressed(), bins=bin_edges
        )
        hist_total_l += hist
    
    cum_rel_dvh_t = calc_rel_dvh(hist_total_t)
    cum_rel_dvh_b = calc_rel_dvh(hist_total_b)
    cum_rel_dvh_l = calc_rel_dvh(hist_total_l)
    """
    fig, ax = plt.subplots()
    ax.set_xlabel('Dose [J/cm2]')
    ax.set_ylabel('Volume [%]')
    ax.plot(bin_edges[:-1], cum_rel_dvh_b, label='Bronchi')
    ax.plot(bin_edges[:-1], cum_rel_dvh_t, label='Tumour')
    ax.plot(bin_edges[:-1], cum_rel_dvh_l, label='Lung Tissue')
    ax.set_xscale('log')
    ax.legend()
    # plt.show()
    """
    return cum_rel_dvh_t, cum_rel_dvh_b, cum_rel_dvh_l, bin_edges
