import pandas as pd
import numpy as np
import os
from SALib.analyze import sobol
import icecream as ic

num_samples = 1000

root_path = "/home/mbpl/morizane/analysis_sensitivity/results"
file_name = "pos_10.0_rot_10.0"
calc_second_order = False


# 位置変数の列名を指定
position_cols = ["pos_x", "pos_y", "pos_z"]
rotation_cols = ["rot_x", "rot_y", "rot_z"]
optical_cols = ["mua_normal", "mus_normal", "mua_tumour", "mus_tumour"]
function_cols = position_cols + rotation_cols + optical_cols

# 回転変数の変化量を計算
center = [248, 416, 384]
initial_rotation = [15, -10, 95]

# A, B, ABをエクセルから読み込み
A_file_path = os.path.join(root_path, file_name, f"input_A_{file_name}.xlsx")
B_file_path = os.path.join(root_path, file_name, f"input_B_{file_name}.xlsx")


A = pd.read_excel(A_file_path)
B = pd.read_excel(B_file_path)


# 中心からの距離を計算
for df in [A, B]:
    df["r"] = np.sqrt(
        (df[position_cols[0]] - center[0]) ** 2
        + (df[position_cols[1]] - center[1]) ** 2
        + (df[position_cols[2]] - center[2]) ** 2
    )

    df["rot_change"] = np.sqrt(
        (df[rotation_cols[0]] - initial_rotation[0]) ** 2
        + (df[rotation_cols[1]] - initial_rotation[1]) ** 2
        + (df[rotation_cols[2]] - initial_rotation[2]) ** 2
    )

# 評価関数のリストを取得
evaluation_functions = [col for col in A.columns if col not in ["r", "rot_change"] + function_cols]
# Placeholder for results
sensitivity_indices = {
    func: {"first_order": {}, "total_order": {}, "second_order": {}, "first_conf": {}, "total_conf": {}, "second_conf": {}}
    for func in evaluation_functions
}


def read_second_orders(func, name):
    num_samples = 1000
    ab = pd.DataFrame(np.zeros((num_samples, 3)))
    ic.ic(len(pd.read_excel(os.path.join(root_path, file_name, f"input_{name}_{file_name}_pos.xlsx"))[func].values))
    ab.iloc[:, 0] = pd.read_excel(os.path.join(root_path, file_name, f"input_{name}_{file_name}_pos.xlsx"))[func].values
    ab.iloc[:, 1] = pd.read_excel(os.path.join(root_path, file_name, f"input_{name}_{file_name}_rot.xlsx"))[func].values
    ab.iloc[:, 2] = pd.read_excel(os.path.join(root_path, file_name, f"input_{name}_{file_name}_opt.xlsx"))[func].values
    ab = ab.values
    
    return ab


for func in evaluation_functions:
    D = 3
    # Define the problem for SALib
    problem = {"num_vars": 3, "names": ["pos", "rot", "opt"], "bounds": [[0, 1]] * 3}
    a = A[func].values
    b = B[func].values
    ab = read_second_orders(func, "AB")
    if calc_second_order:
            ba = read_second_orders(func, "BA")
    Y = np.zeros(num_samples * 3)

    step =  2*D  + 2 if calc_second_order else D+2
    N = len(a)
    Y = np.zeros(N * step)  # 元のYのサイズを予測（step * Nの長さ）
    Y[0 : Y.size : step] = a
    Y[(step - 1) : Y.size : step] = b

    for j in range(D):
        Y[(j + 1) : Y.size : step] = ab[:, j]
        if calc_second_order:
            Y[(j + 1 + D) : Y.size : step] = ba[:, j]
    

    # Perform Sobol sensitivity analysis
    Si = sobol.analyze(problem, Y, calc_second_order=False)

    for i, name in enumerate(problem["names"]):
        sensitivity_indices[func]["first_order"][name] = Si["S1"][i]
        sensitivity_indices[func]["total_order"][name] = Si["ST"][i]
        sensitivity_indices[func]["first_conf"][name] = Si["S1_conf"][i]
        sensitivity_indices[func]["total_conf"][name] = Si["ST_conf"][i]
        if calc_second_order:
            sensitivity_indices[func]["second_order"][name] = Si["S2"][i]
            sensitivity_indices[func]["second_conf"][name] = Si["S2_conf"][i]

# sensitivity_indicesをDataFrameにまとめる
sensitivity_df = pd.DataFrame(
    [
        {"function": func, "order": order, "name": name, "value": sensitivity_indices[func][order][name]}
        for func in sensitivity_indices.keys()
        for order in sensitivity_indices[func].keys()
        for name in sensitivity_indices[func][order].keys()
    ]
)


# DataFrameを表示
print(sensitivity_df)
sensitivity_df.to_excel(os.path.join(root_path, f"sensitivity_indices_{file_name}_Salib.xlsx"))
