import numpy as np
from run_mcx import Runmcx
import pandas as pd
from calc_dvh import calc_log_dvh_with_log
from scipy.interpolate import interp1d
from cylinder import create_rotated_cylinder_mask
from numpy.random import default_rng
from send_mail import send_email
import traceback
import os
from rotate_fiber import rotate_fiber
import re


class SobolAnalysis(Runmcx):
    def set_variables(self, variables: dict, std_dev: dict, num_samples):
        """Generates samples for variables with given standard deviations."""
        samples = []
        covariance_matrix = np.diag(
            [std_dev["pos"] ** 2, std_dev["pos"] ** 2, std_dev["pos"] ** 2]
        )  # 共分散行列（対角行列）
        variables_pos = variables["pos"]
        rng = default_rng()
        samples_pos = rng.multivariate_normal(variables_pos, covariance_matrix, num_samples)

        samples_rot = rotate_fiber(variables_pos, self.length, variables["pos"], std_dev["rot"], num_samples)

        variables_opt = np.array(variables["opt"])
        std_devs = np.array(std_dev["opt"])
        samples = []
        for val, std in zip(variables_opt, std_devs):
            sample = np.random.normal(val, std, num_samples)
            samples.append(sample)
        samples_opt = np.array(samples).T

        return {"pos": samples_pos, "rot": samples_rot, "opt": samples_opt}

    def run_simulation(self, vals, label):
        """Runs simulations for given values."""
        match = re.search(r"pos_\d+\.\d+_rot_\d+\.\d+", label)
        if match:
            extracted_string = match.group()
        else:
            extracted_string = ""
        save_directory_path = os.path.join(self.save_path, extracted_string)
        if not os.path.exists(self.save_path):
            os.makedirs(os.path.join(self.save_path))
        if not os.path.exists(save_directory_path):
            os.makedirs(save_directory_path)

        for i in range(len(vals["pos"])):
            cylinder_pos = vals["pos"][i]
            cylinder_rot = vals["rot"][i]
            params = {
                "position": cylinder_pos,
                "rotation": cylinder_rot,
                "mua_normal": vals["opt"][i][0],
                "mus_normal": vals["opt"][i][1],
                "mua_tumour": vals["opt"][i][2],
                "mus_tumour": vals["opt"][i][3],
            }
            try:
                self.run_mcx(params)  # , cylinder_rot, opt_tumour, opt_normal)
                self.create_model(cylinder_pos, cylinder_rot)
                res_dict = self.evaluate_model()
            except Exception as e:
                send_email("Error", f"Error occurred in {label} analysis\n{e}")
                continue
            res_dict.update(
                {
                    "pos_x": params["position"][0],
                    "pos_y": params["position"][1],
                    "pos_z": params["position"][2],
                    "rot_x": params["rotation"][0],
                    "rot_y": params["rotation"][1],
                    "rot_z": params["rotation"][2],
                    "mua_normal": params["mua_normal"],
                    "mus_normal": params["mus_normal"],
                    "mua_tumour": params["mua_tumour"],
                    "mus_tumour": params["mus_tumour"],
                }
            )

            df = pd.DataFrame(res_dict, index=[0])
            print(df)

            save_file = os.path.join(save_directory_path, f"{label}.xlsx")
            if i == 0:
                df.to_excel(save_file, index=False, header=True)
            else:
                with pd.ExcelWriter(
                    save_file,
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

    def create_model(self, cylinder_pos, cylinder_rot):
        with open(self.model_path, "rb") as f:
            self.model = np.fromfile(f, dtype=np.uint8).reshape((100, 100, 100), order="F")
        print("aa")
        cylinder_mask = create_rotated_cylinder_mask(
            [100, 100, 100], cylinder_pos, self.radius, self.length, cylinder_rot
        )

        self.model[cylinder_mask == 1] = 4

    def evaluate_model(self):
        """
        Evaluates the model based on the simulation results.
        """
        with open(f"{self.OUTPUT_PATH}.mc2", "rb") as f:
            res = np.fromfile(f, dtype=np.float32).reshape((100, 100, 100), order="F")
            energy = 150 * (10**-3) * 667 * 100
            res *= energy

        # DVH (log)
        cum_rel_dvh_t, _, cum_rel_dvh_l, bin_edges = calc_log_dvh_with_log(res, self.model)

        res_dict = {}
        # tumour tissue

        def calc_dvh(Dx, i):
            try:
                return float(Dx([i])[0])
            except Exception:
                return np.nan

        def calc_cover_rate_tumour(i):
            return np.count_nonzero((self.model == 3) & (res >= i)) / np.count_nonzero(self.model == 3) * 100

        def calc_cover_rate_normal(i):
            return np.count_nonzero((self.model == 1) & (res >= i)) / np.count_nonzero(self.model == 1) * 100

        def calc_irrad_volume_normal(i):
            return np.count_nonzero((self.model == 1) & (res >= i))

        def calc_irrad_volume_tumour(i):
            return np.count_nonzero((self.model == 3) & (res >= i))

        # normal tissue
        Dx_tumour = interp1d(cum_rel_dvh_t, bin_edges[:-1])
        Dx_normal = interp1d(cum_rel_dvh_l, bin_edges[:-1])
        for i in [90, 80, 70, 60, 50, 40, 30, 20, 10]:
            res_dict[f"dvh_tumour_d{i}"] = calc_dvh(Dx_tumour, i)
            res_dict[f"dvh_normal_d{i}"] = calc_dvh(Dx_normal, i)

        for i in [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 1, 0.1]:
            res_dict[f"cover_rate_normal_{i}"] = calc_cover_rate_normal(i)
            res_dict[f"cover_rate_tumour_{i}"] = calc_cover_rate_tumour(i)
            res_dict[f"irrad_volume_normal_{i}"] = calc_irrad_volume_normal(i)
            res_dict[f"irrad_volume_tumour_{i}"] = calc_irrad_volume_tumour(i)

        return res_dict

    def create_Ci(self, A, B, i):
        C = A.copy()
        C[i] = B[i]
        return C

    def sobol_analysis(self, variables, std_dev, num_samples: int):
        """
        Sobol sensitivity analysis
        """

        # A = pd.read_excel("/home/raise/mcx_simulation/analysis_sensitivity/input_A.xlsx").iloc[:, 0:3].values.tolist()
        std_dev_pos = std_dev["pos"] * 3
        std_dev_rot = std_dev["rot"] * 3
        try:
            send_email("A simulation", "A simulation has started.")
            A = self.set_variables(variables, std_dev, num_samples)
            self.run_simulation(A, f"A_pos_{std_dev_pos}_rot_{std_dev_rot}")
            send_email("A simulation", "A simulation was successfully completed.")
        except Exception as e:
            tb = traceback.format_exc()
            send_email("Error", f"Error occurred in A simulation\n{e}\nTraceback:\n{tb}")

        try:
            send_email("B simulation", "B simulation has started.")
            B = self.set_variables(variables, std_dev, num_samples)
            self.run_simulation(B, f"B_pos_{std_dev_pos}_rot_{std_dev_rot}")
            send_email("B simulation", "B simulation was successfully completed.")
        except Exception as e:
            send_email("Error", f"Error occurred in B simulation\n{e}")

        """
        samples = []
        for i in range(len(A)):
            samples.append(A[i][1])
        plt.hist(samples, bins=100)
        plt.show()
        
        """

        for changes in ["pos", "rot", "opt"]:
            try:
                send_email("AB simulation", "AB simulation has started.")
                AB = self.create_Ci(A, B, changes)
                self.run_simulation(AB, f"AB_pos_{std_dev_pos}_rot_{std_dev_rot}_{changes}")
                send_email(
                    "AB simulation",
                    f"AB simulation {std_dev_pos}_rot_{std_dev_rot}_{changes} was successfully completed.",
                )

                send_email("BA simulation", "BA simulation has started.")
                BA = self.create_Ci(B, A, changes)
                self.run_simulation(BA, f"BA_pos_{std_dev_pos}_rot_{std_dev_rot}_{changes}")
                send_email(
                    "BA simulation",
                    f"BA simulation {std_dev_pos}_rot_{std_dev_rot}_{changes} was successfully completed.",
                )
            except Exception as e:
                send_email("Error", f"Error occurred in AB simulation\n{e}")
