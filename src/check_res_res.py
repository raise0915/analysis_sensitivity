

import numpy as np
import pandas as pd
import icecream as ic

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
df['r'] = X
"""



# 条件付き期待値と一次感度指数を計算する関数
def compute_sobol_index(df, input_var, output_var, num_bins=100):
    """
    Sobol一次感度指数を計算
    df: データフレーム
    input_var: 入力変数の名前
    output_var: 出力変数の名前
    num_bins: 入力変数を分割するビンの数
    """
    # 入力変数を等間隔にビン分割
    df['binned_input'] = pd.cut(df[input_var], bins=num_bins, labels=False)
    ic.ic(df['binned_input'])
    
    # 条件付き期待値を計算
    conditional_mean = df.groupby('binned_input')[output_var].mean().values
    
    # 条件付き期待値の分散
    variance_conditional_mean = np.var(conditional_mean)
    
    # 全体の分散
    variance_total = np.var(df[output_var])
    
    # Sobol一次感度指数
    S1 = variance_conditional_mean / variance_total
    
    return S1


# optical_colsの変数について一次感度指数を計算
optical_results = {}
for optical_var in optical_cols:
    for func in function_cols:
        S1 = compute_sobol_index(df, optical_var, func, num_bins=1000)
        optical_results[(optical_var, func)] = S1

# 結果を表示
for optical_var, S1 in optical_results.items():
    print(f"S1 (一次感度指数) for {optical_var}: {S1}")

# 評価関数ごとに一次感度指数を計算
results = {}
for func in function_cols:
    S1 = compute_sobol_index(df, 'r', func, num_bins=1000)
    results[func] = S1

# 結果を表示
for func, S1 in results.items():
    print(f"S1 (一次感度指数) for {func}: {S1}")
