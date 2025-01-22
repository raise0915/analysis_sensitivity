import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os 

num_samples = 1000

root_path = '/home/mbpl/morizane/analysis_sensitivity/'
# 位置変数の列名を指定
position_cols = ["pos_x", "pos_y", "pos_z"]
rotation_cols = ["rot_x", "rot_y", "rot_z"]
optical_cols = ["mua_normal", "mus_normal", "mua_tumour", "mus_tumour"]
function_cols = position_cols + rotation_cols + optical_cols

# 回転変数の変化量を計算
center = [248, 416, 384]
initial_rotation = [15, -10, 95]

# A, B, ABをエクセルから読み込み
A_file_path = os.path.join(root_path, "input_1227_A_1227_pos_10_rot_10.xlsx")
B_file_path = os.path.join(root_path, "input_1227_B_1227_pos_10_rot_10.xlsx")


A = pd.read_excel(A_file_path)
B = pd.read_excel(B_file_path)


# 中心からの距離を計算
for df in [A, B]:
    df['r'] = np.sqrt((df[position_cols[0]] - center[0])**2 + 
                    (df[position_cols[1]] - center[1])**2 + 
                    (df[position_cols[2]] - center[2])**2)

    df['rot_change'] = np.sqrt((df[rotation_cols[0]] - initial_rotation[0])**2 + 
                            (df[rotation_cols[1]] - initial_rotation[1])**2 + 
                            (df[rotation_cols[2]] - initial_rotation[2])**2)

# 評価関数のリストを取得
evaluation_functions = [col for col in A.columns if col not in ['r', 'rot_change'] + function_cols]

def first_order(A, B, AB):
    y = np.r_[A, B]
    return np.mean(B * (AB - A)) / np.var(y)

def total_order(A, B, AB):
    """
    Total order estimator following Saltelli et al. 2010 CPC, normalized by
    sample variance
    """
    y = np.r_[A, B]
    return 0.5 * np.mean((AB - A) ** 2) / np.var(y)

def second_order(A, ABj, ABk, BAj, B):
    """Second order estimator following Saltelli 2002"""
    y = np.r_[A, B]

    Vjk = np.mean(BAj * ABk - A * B, axis=0) / np.var(y)
    Sj = first_order(A, B, ABj)
    Sk = first_order(A, B, ABk)

    return Vjk - Sj - Sk

def read_ab(name):
    return pd.read_excel(os.path.join(root_path, f"input_1227_AB_pos_10_rot_10_1227_{name}.xlsx"))


# 一次感度指数の出力
sensitivity_indices = {func: {'first_order': {}, 'total_order': {}, 'first_conf': {}} for func in evaluation_functions}


for func in evaluation_functions:
    a = A[func].values
    b = B[func].values
    rng = np.random.default_rng().integers
    r = rng(num_samples, size=(num_samples, 100))

    for name in ['pos', 'rot', 'opt']:
        ab = read_ab(name)[func].values
        sensitivity_indices[func]['first_order'][name] = first_order(a, b, ab)
        sensitivity_indices[func]['total_order'][name] = total_order(a, b, ab)
        sensitivity_indices[func]['first_conf'][name] = first_order(a[r], b[r], ab[r])


    # A, B, ABについて全てまとめてYとし、Y=(Y - Y.mean()) / Y.std()
    # Y = np.r_[ab, b, ab]
    # Y = (Y - Y.mean()) / Y.std()
    # a = Y[0:a_size]
    # b = Y[a_size:a_size + b_size]
    # ab = Y[a_size + b_size:]

# sensitivity_indicesをDataFrameにまとめる
sensitivity_df = pd.DataFrame([
    {'function': func, 'order': order, 'name': name, 'value': sensitivity_indices[func][order][name]}
    for func in sensitivity_indices.keys()
    for order in sensitivity_indices[func].keys()
    for name in sensitivity_indices[func][order].keys()
])

# DataFrameを表示
print(sensitivity_df)
sensitivity_df.to_excel(os.path.join(root_path, "sensitivity_indices_10.xlsx"))

