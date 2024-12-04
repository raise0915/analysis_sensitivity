import numpy as np
import matplotlib.pyplot as plt

## Ishigami function 

# モデルの設定
def model(X):
    return np.sin(X[:, 0]) + 7 * np.sin(X[:, 1]) ** 2 + 0.1 * X[:, 2] ** 4 * np.sin(X[:, 0])

# Sobol感度解析のメイン関数
def sobol_analysis(num_vars, num_samples, w, max_order=1):
    std_dev =  1 # np.abs(w)
    A = np.random.normal(w, std_dev, (num_samples, num_vars))
    B = np.random.normal(w, std_dev, (num_samples, num_vars))

    Y_A = model(A)
    Y_B = model(B)
    V_Y = np.var(Y_A, ddof=1)

    sobol_first = np.zeros(num_vars)
    sobol_first_err = np.zeros(num_vars)

    for i in range(num_vars):
        A_Bi = A.copy()
        A_Bi[:, i] = B[:, i]
        Y_A_Bi = model(A_Bi)

        sobol_first[i] = np.mean(Y_A * (Y_A_Bi - Y_B)) / V_Y
        sobol_first_err[i] = np.std(Y_A * (Y_A_Bi - Y_B)) / V_Y 

    return sobol_first, sobol_first_err

a = np.random.normal(0, 1, 10000)

print(a)
print(len(a))

# パラメータ設定
num_vars = 3          # 入力変数の数
num_samples = 1000000 # サンプル数
w = 0               # 正規分布の平均

# 実行
sobol_first, sobol_first_err = sobol_analysis(num_vars, num_samples, w)

# プロット
labels = [f"x{i+1}" for i in range(num_vars)]
x = np.arange(num_vars)
plt.bar(x, sobol_first, yerr=sobol_first_err, capsize=5, color='steelblue')
plt.xticks(x, labels)
plt.ylabel("ST")
plt.title("Sobol Sensitivity Analysis")
plt.show()



## エクセルに自動入力させる