import numpy as np
from run_mcx import Runmcx
import pandas as pd
from calc_dvh import calc_log_dvh_with_log
from scipy.interpolate import interp1d
from cylinder import create_rotated_cylinder_mask
import icecream as ic
from numpy.random import default_rng


class SobolAnalysis(Runmcx):
    def set_variables(self, variables, std_dev, num_samples):
        """
        Generates samples for variables with given standard deviations.
        """
        samples = []
        covariance_matrix = np.diag(
            [std_dev**2, std_dev**2, std_dev**2]
        )  # 共分散行列（対角行列）
        variables = np.concatenate(variables)
        rng = default_rng()
        samples = rng.multivariate_normal(variables, covariance_matrix, num_samples)
        return samples

    def run_simulation(self, vals, label):
        """
        Runs simulations for given values.
        """
        result = []
        for i, val in enumerate(vals):
            cylinder_pos = [val[0], val[1], val[2]]
            cylinder_rot = [15, -10, 95]
            # cylinder_rot = [val[3], val[4], val[5]]
            # opt_tumour = [val[6], val[7], val[8], val[9]]
            # opt_normal = [val[10], val[11], val[12], val[13]]

            self.run_mcx(cylinder_pos)  # , cylinder_rot, opt_tumour, opt_normal)
            self.create_model(cylinder_pos, cylinder_rot)
            res_dict = self.evaluate_model()
            df = pd.DataFrame(
                {
                    "pos_x": [cylinder_pos[0]],
                    "pos_y": [cylinder_pos[1]],
                    "pos_z": [cylinder_pos[2]],
                    "D90_normal": res_dict["dvh"]["normal"]["d90"],
                    "D50_normal": res_dict["dvh"]["normal"]["d50"],
                    "D10_normal": res_dict["dvh"]["normal"]["d10"],
                    "D90_tumour": res_dict["dvh"]["tumour"]["d90"],
                    "D50_tumour": res_dict["dvh"]["tumour"]["d50"],
                    "D10_tumour": res_dict["dvh"]["tumour"]["d10"],
                    "cover_rate_100": [res_dict["cover_rate_100"]],
                    "cover_rate_10": [res_dict["cover_rate_10"]],
                    "cover_rate_1": [res_dict["cover_rate_1"]],
                }
            )
            result.append(
                [
                    res_dict["dvh"]["normal"]["d90"],
                    res_dict["dvh"]["normal"]["d50"],
                    res_dict["dvh"]["normal"]["d10"],
                    res_dict["dvh"]["tumour"]["d90"],
                    res_dict["dvh"]["tumour"]["d50"],
                    res_dict["dvh"]["tumour"]["d10"],
                    res_dict["cover_rate_100"],
                    res_dict["cover_rate_10"],
                    res_dict["cover_rate_1"],
                ]
            )
            if i == 0:
                df.to_excel(f"{self.input_name}_{label}.xlsx", index=False, header=True)
            else:
                with pd.ExcelWriter(
                    f"{self.input_name}_{label}.xlsx",
                    mode="a",
                    engine="openpyxl",
                    if_sheet_exists="overlay",
                ) as writer:
                    df.to_excel(
                        writer,
                        sheet_name="Sheet1",
                        index=False,
                        header=False,
                        startrow=i + 1,
                    )

        return result

    def create_model(self, cylinder_pos, cylinder_rot):
        with open(self.model_path, "rb") as f:
            self.model = np.fromfile(f, dtype=np.uint8).reshape(
                (600, 600, 600), order="F"
            )
        cylinder_mask = create_rotated_cylinder_mask(
            [600, 600, 600], cylinder_pos, self.radius, self.length, cylinder_rot
        )
        self.model[cylinder_mask == 1] = 4

    def evaluate_model(self):
        """
        Evaluates the model based on the simulation results.
        """
        with open(f"{self.OUTPUT_PATH}.mc2", "rb") as f:
            res = np.fromfile(f, dtype=np.float32).reshape((600, 600, 600), order="F")
            energy = 150 * (10**-3) * 667 * 100
            res *= energy

        # DVH (log)
        cum_rel_dvh_t, _, cum_rel_dvh_l, bin_edges = calc_log_dvh_with_log(
            res, self.model
        )

        # tumour tissue
        Dx = interp1d(cum_rel_dvh_t, bin_edges[:-1])
        try:
            d90_t = float(Dx([90])[0])
        except Exception:
            d90_t = np.nan
        try:
            d50_t = float(Dx([50])[0])
        except Exception:
            d50_t = np.nan
        try:
            d10_t = float(Dx([10])[0])
        except Exception:
            d10_t = np.nan

        # normal tissue
        Dx = interp1d(cum_rel_dvh_l, bin_edges[:-1])
        try:
            d90_n = float(Dx([90])[0])
        except Exception:
            d90_n = np.nan
        try:
            d50_n = float(Dx([50])[0])
        except Exception:
            d50_n = np.nan
        try:
            d10_n = float(Dx([10])[0])
        except Exception:
            d10_n = np.nan

        # cover_rate
        cover_rate_100 = (
            np.count_nonzero((self.model == 3) & (res >= 100))
            / np.count_nonzero(self.model == 3)
            * 100
        )
        cover_rate_10 = (
            np.count_nonzero((self.model == 3) & (res >= 10))
            / np.count_nonzero(self.model == 3)
            * 100
        )
        cover_rate_1 = (
            np.count_nonzero((self.model == 3) & (res >= 1))
            / np.count_nonzero(self.model == 3)
            * 100
        )

        return {
            "dvh": {
                "tumour": {
                    "d90": d90_t,
                    "d50": d50_t,
                    "d10": d10_t,
                },
                "normal": {"d90": d90_n, "d50": d50_n, "d10": d10_n},
            },
            "cover_rate_100": cover_rate_100,
            "cover_rate_10": cover_rate_10,
            "cover_rate_1": cover_rate_1,
        }

    def sobol_analysis(self, variables: list, std_dev: int, num_samples: int):
        """
        Sobol sensitivity analysis
        """

        # A = pd.read_excel("/home/raise/mcx_simulation/analysis_sensitivity/input_A.xlsx").iloc[:, 0:3].values.tolist()
        A = self.set_variables(variables, std_dev, num_samples)
        """
        samples = []
        for i in range(len(A)):
            samples.append(A[i][1])
        plt.hist(samples, bins=100)
        plt.show()
        """

        # B = pd.read_excel("/home/raise/mcx_simulation/analysis_sensitivity/input_B.xlsx").iloc[:, 0:3].values.tolist()
        # B = self.set_variables(variables, std_dev, num_samples)

        Y_A = self.run_simulation(A, f"A_{std_dev}")
        # Y_B = self.run_simulation(B, "B")
        # Y_B = pd.read_excel("/home/raise/mcx_simulation/analysis_sensitivity/input_B.xlsx").iloc[:, 3:].values.tolist()

        V_Y = []
        for i in range(9):
            V_Y.append(np.var([row[i] for row in Y_A]))
        ic.ic(V_Y)
        V_Y = np.array(V_Y)
        
        # V_Y = np.var(Y_A, ddof=1)

        sobol_first = np.zeros(9)
        sobol_first_err = np.zeros(9)
        ic.ic(sobol_first)

        for j in range(9):
            Y_A_column = np.array([row[j] for row in Y_A])
            sobol_first[j] = np.mean(Y_A_column) / V_Y[j]
            sobol_first_err[j] = np.std(np.array(Y_A_column)) / V_Y[j]

        """ 
        for i in range(3):
            if i == 0:
                Y_A_Bi = pd.read_excel("/home/raise/mcx_simulation/analysis_sensitivity/input_change_0.xlsx").iloc[:, 3:].values.tolist()
            else:
                A_Bi = copy.deepcopy(A)
                for j in range(num_samples):
                    A_Bi[j][i] = B[j][i]
                Y_A_Bi = self.run_simulation(A_Bi, f'change_{i}')
            ic.ic(len(Y_A))
            ic.ic(len(Y_A_Bi))
            for j in range(9):
                Y_A_Bi_column = np.array([row[j] for row in Y_A_Bi])
                Y_A_column = np.array([row[j] for row in Y_A])
                Y_B_column = np.array([row[j] for row in Y_B])
                sobol_first[i][j] = np.mean(Y_A_column * (Y_A_Bi_column - Y_B_column)) / V_Y[j]
                sobol_first_err[i][j] = np.std(np.array(Y_A_column) * (np.array(Y_A_Bi_column) - np.array(Y_B_column))) / V_Y[j]

            ic.ic(sobol_first)
        """

        return sobol_first, sobol_first_err


if __name__ == "__main__":
    # setting params
    num_samples = 1000  # サンプル数
    pos_x = 248
    pos_y = 416
    pos_z = 384

    ## 腫瘍
    # 3sigma = 5 mm
    # w = 0.05 # 30%/2
    for sigma in [5, 10, 15]:
        std_dev = round(sigma / 3, 2)
        ic.ic(std_dev)

        variables = [
            [pos_x, pos_y, pos_z]  # position
            # [15, -10, 95], # direction
            # [0.21, 22, 0.9, 1.4], # optical_propeties for tumour
            # [0, 0, 1, 1] # optical_properties for normal
        ]
        max_vals = [
            [600, 600, 600]
            # [180, 180, 180],
            # [10**5, 10**5, 10**5, 10**5],
            # [10**5, 10**5, 10**5, 10**5]
        ]

        # run
        sim = SobolAnalysis()
        sobol_first, sobol_first_err = sim.sobol_analysis(
            variables, std_dev, num_samples
        )

        # DataFrameに変換
        df_sobol_first = pd.DataFrame(sobol_first)
        df_sobol_first_err = pd.DataFrame(sobol_first_err)

        # エクセルに保存
        with pd.ExcelWriter(f"/home/mbpl/morizane/analysis_sensitivity/sobol_analysis_results_{sigma}.xlsx") as writer:
            df_sobol_first.to_excel(writer, sheet_name="Sobol First")
            df_sobol_first_err.to_excel(writer, sheet_name="Sobol First Error")

"""
    # プロット
    vals_flatten = np.array(variables).flatten()
    labels = [f"x{i+1}" for i in range(len(vals_flatten))]
    x = np.arange(len(variables))
    plt.bar(x, sobol_first[0], yerr=sobol_first_err, capsize=5, color="steelblue")
    plt.xticks(x, labels)
    plt.ylabel("ST")
    plt.title("Sobol Sensitivity Analysis")
    plt.show()
"""

