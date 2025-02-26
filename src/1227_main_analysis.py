import numpy as np
import json

from analysis_sensitivity import SobolAnalysis
from send_mail import send_email

if __name__ == "__main__":
    with open("/home/mbpl/morizane/analysis_sensitivity/src/inputs/default_value.json") as f:
        default_value = json.load(f)
    print(default_value)

    # setting params
    num_samples = 1000  # サンプル数
    mua_normal = 0.37
    mua_tumour = 0.27
    mus_normal = 27
    mus_tumour = 20
    opt_std_devs = np.array([0.14, 10, 0.05, 4.6])

    variables = {
        "pos": np.array(default_value["pos_cut_100"]),
        "rot": default_value["rot_cut_100"],
        "opt": [mua_normal, mus_normal, mua_tumour, mus_tumour],
    }

    std_dev = {"opt": opt_std_devs * 0.5 / 3}

    ## 腫瘍
    # 3sigma = 5 mm
    # w = 0.05 # 30%/2
    # optical properties - 平均値から50%の範囲で
    for sigma_pos in [2, 4, 6, 8, 10]:
        for sigma_rot in [2, 4, 6, 8, 10]:
            std_dev["pos"] = sigma_pos / 3
            std_dev["rot"] = sigma_rot / 3
            sim = SobolAnalysis()
            sim.sobol_analysis(variables, std_dev, num_samples)
            send_email("Success", f"Position analysis sigma_pos:{sigma_pos} sigma_rot:{sigma_rot} is done")
    send_email("Success", "All analysis is done")
