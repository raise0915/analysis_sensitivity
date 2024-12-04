import numpy as np
import icecream as ic
from matplotlib import pyplot as plt
import pandas as pd

pos_x = 248
pos_y =  416
pos_z = 384
        
variables = [
    [pos_x, pos_y, pos_z] # position
    # [15, -10, 95], # direction
    # [0.21, 22, 0.9, 1.4], # optical_propeties for tumour
    # [0, 0, 1, 1] # optical_properties for normal
]

Y_A = pd.read_excel("/home/raise/mcx_simulation/analysis_sensitivity/input_A.xlsx").iloc[:, 3:].values.tolist()
V_Y =  (np.var(Y_A, axis=0, ddof=1))
Y_B = pd.read_excel("/home/raise/mcx_simulation/analysis_sensitivity/input_B.xlsx").iloc[:, 3:].values.tolist()
        

sobol_first = {i: {} for i in range(3)}
sobol_first_err = {i: {} for i in range(3)}

for i in range(2):
    Y_A_Bi = pd.read_excel(f"/home/raise/mcx_simulation/analysis_sensitivity/input_change_{i}.xlsx").iloc[:, 3:].values.tolist()
    for j in range(9):
        Y_A_Bi_column = np.array([row[j] for row in Y_A_Bi])
        Y_A_column = np.array([row[j] for row in Y_A])
        Y_B_column = np.array([row[j] for row in Y_B])
        sobol_first[i][j] = np.mean(Y_A_column * (Y_A_Bi_column - Y_B_column)) / V_Y[j]
        sobol_first_err[i][j] = np.std(np.array(Y_A_column) * (np.array(Y_A_Bi_column) - np.array(Y_B_column))) / V_Y[j]

ic.ic(sobol_first)
# プロット
columns = ["normal_D90", "normal_D50", "normal_D10", "tumour_D90", "tumour_D50", "tumour_D10", "cover_rate_100", "cover_rate_10", "cover_rate_1"]
vals_flatten = np.array(variables).flatten()
labels = ["x"] # [f"x{i+1}" for i in range(len(vals_flatten))]
x = [0] # np.arange(len(variables))


"""
@analysis_res.py: このプロットは、Sobol感度分析の結果を示しています。
各バーは、異なる出力変数（normal_D90, normal_D50, など）に対する感度指数を表しており、
エラーバーはその不確実性を示しています。
"""
fig, ax = plt.subplots(figsize=(12, 6))
for j in range(3,9):
    ax.bar(j, sobol_first[0][j], yerr=sobol_first_err[1][j], capsize=5, color='steelblue')
ax.set_xticks(range(3,9))
# ax.set_xticklabels([i for i in columns])
ax.set_ylabel("ST")
ax.set_title("Sobol Sensitivity Analysis")



plt.show()
