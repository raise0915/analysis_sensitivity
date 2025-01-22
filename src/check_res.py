import icecream as ic
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.random import default_rng


# 1. データの読み込み
# エクセルファイルのパスを指定
file_path = "/home/mbpl/morizane/analysis_sensitivity/input_optical_properties_1225_A_[0. 4. 0. 3.].xlsx"  # 例: "output_data.xlsx"
df = pd.read_excel(file_path)

# 位置変数の列名を指定
position_cols = ["pos_x", "pos_y", "pos_z"]
rotation_cols = ["rot_x", "rot_y", "rot_z"]
optical_cols = ["mua_normal", "mus_normal", "mua_tumour", "mus_tumour"]
function_cols = [col for col in df.columns if col not in (position_cols + rotation_cols + optical_cols)]

"""
# 合成位置変数 (例: 原点からの距離)
origin = np.array([247, 414, 382])
X = np.sqrt(
    (df[position_cols[0]] - origin[0]) ** 2
    + (df[position_cols[1]] - origin[1]) ** 2
    + (df[position_cols[2]] - origin[2]) ** 2
)
"""
# optical_colsを変数としてSobol解析
X = df[optical_cols].values


# 感度指数を計算
def first_order_sensitivity(X, Y):
    Y_mean = np.mean(Y)
    # Y_var = np.var(Y)
    S = np.mean((X - np.mean(X, axis=0)) * (Y.reshape(-1, 1) - Y_mean), axis=0) / np.var(Y, axis=0)
    return S


# Sobol一次感度指数を計算してリストに格納
sobol_indices = []
for func in function_cols:
    Y = df[func].values

    sobol_index = first_order_sensitivity(X, Y)
    sobol_indices.append([sobol_index])  # リストのリストに変換
    ic.ic(func)
    ic.ic(sobol_index)

plt.figure(figsize=(10, 6))
plt.boxplot([df[func].values for func in function_cols], vert=True, patch_artist=True, labels=function_cols) # type: ignore
plt.title("Distribution of Functions")
plt.xlabel("Function")
plt.ylabel("Value")
plt.xticks(rotation=90)
plt.show()
# 結果をデータフレームにまとめる
sobol_df = pd.DataFrame(sobol_indices, index=function_cols, columns=["Sobol Index"])

# エクセルファイルに出力
output_file_path = "sobol_indices_output_1.67.xlsx"
sobol_df.to_excel(output_file_path, index=True)

# 箱ひげ図を作成
plt.figure(figsize=(10, 6))
plt.boxplot(sobol_indices, vert=True, patch_artist=True, labels=function_cols)
plt.title("First-order Sobol Indices")
plt.xlabel("Function")
plt.ylabel("Sobol Index")
plt.xticks(rotation=90)
plt.show()
