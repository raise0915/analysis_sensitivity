import numpy as np
from run_mcx import Runmcx
import pandas as pd
from calc_dvh import calc_log_dvh_with_log
from scipy.interpolate import interp1d
from cylinder import create_rotated_cylinder_mask
import icecream as ic
from numpy.random import default_rng
from send_mail import send_email
import traceback

global today
today ='0217'


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
            if self.type == 'optical_properties':
                cylinder_pos = [248, 416, 384]
                cylinder_rot = [15, -10, 95]
                params = {
                    'position': cylinder_pos,
                    'rotation': cylinder_rot,
                    'mua_normal': val[0],
                    'mus_normal': val[1],
                    'mua_tumour': val[2],
                    'mus_tumour': val[3]
                }
            else:
                if self.type == 'direction':
                    cylinder_pos = [248, 416, 384]
                    cylinder_rot = [val[0], val[1], val[2]]
                else:
                    cylinder_pos = [val[0], val[1], val[2]]
                    cylinder_rot = [15, -10, 95]

                params = {
                    'position': [cylinder_pos[0], cylinder_pos[1], cylinder_pos[2]],
                    'rotation': [cylinder_rot[0], cylinder_rot[1], cylinder_rot[2]],
                    'mua_normal': 0.37,
                    'mus_normal': 27,
                    'mua_tumour': 0.27,
                    'mus_tumour': 20
                }
            try:
                self.run_mcx(params)  # , cylinder_rot, opt_tumour, opt_normal)
                self.create_model(cylinder_pos, cylinder_rot)
                res_dict = self.evaluate_model()
            except Exception as e:
                send_email('Error', e)
                continue
            df = pd.DataFrame(
                {
                    "pos_x": [params['position'][0]],
                    "pos_y": [params['position'][1]],
                    "pos_z": [params['position'][2]],
                    "rot_x": [params['rotation'][0]],
                    "rot_y": [params['rotation'][1]],
                    "rot_z": [params['rotation'][2]],
                    "mua_normal": [params['mua_normal']],
                    "mus_normal": [params['mus_normal']],
                    "mua_tumour": [params['mua_tumour']],
                    "mus_tumour": [params['mus_tumour']],
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
                (200, 200, 200), order="F"
            )
        print('aa')
        cylinder_mask = create_rotated_cylinder_mask(
            [200, 200, 200], cylinder_pos, self.radius, self.length, cylinder_rot
        )
        
        self.model[cylinder_mask == 1] = 4

    def evaluate_model(self):
        """
        Evaluates the model based on the simulation results.
        """
        with open(f"{self.OUTPUT_PATH}.mc2", "rb") as f:
            res = np.fromfile(f, dtype=np.float32).reshape((200, 200, 200), order="F")
            energy = 150 * (10**-3) * 667 * 100
            res *= energy

        # DVH (log)
        cum_rel_dvh_t, _, cum_rel_dvh_l, bin_edges = calc_log_dvh_with_log(
            res, self.model
        )

        res_dict = {}
        # tumour tissue
        
        def calc_dvh(Dx, i):
            try:
                return float(Dx([i])[0])
            except Exception:
                return np.nan
        
        def calc_cover_rate_tumour(i):
            return (
                np.count_nonzero((self.model == 3) & (res >= i))
                / np.count_nonzero(self.model == 3)
                * 100
            )
        
        def calc_cover_rate_normal(i):
            return (
                np.count_nonzero((self.model == 1) & (res >= i))
                / np.count_nonzero(self.model == 1)
                * 100
            )

        def calc_irrad_volume_normal(i):
            return np.count_nonzero((self.model == 1) & (res >= i))

        def calc_irrad_volume_tumour(i):
            return np.count_nonzero((self.model == 3) & (res >= i))
                       
        # normal tissue
        Dx_tumour = interp1d(cum_rel_dvh_t, bin_edges[:-1])
        Dx_normal = interp1d(cum_rel_dvh_l, bin_edges[:-1])
        for i in [90, 50, 10]:
            res_dict[f'dvh_tumour_d{i}'] = calc_dvh(Dx_tumour, i)
            res_dict[f'dvh_normal_d{i}'] = calc_dvh(Dx_normal, i)

        for i in  [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 1, 0.1]:
            res_dict[f'cover_rate_normal_{i}'] = calc_cover_rate_normal(i)
            res_dict[f'cover_rate_tumour_{i}'] = calc_cover_rate_tumour(i)
            res_dict[f'irrad_volume_normal_{i}'] = calc_irrad_volume_normal(i)
            res_dict[f'irrad_volume_tumour_{i}'] = calc_irrad_volume_tumour(i)
        
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
        std_dev_pos = std_dev['pos'] * 3
        std_dev_rot = std_dev['rot'] * 3
        try:
            send_email('A simulation', 'A simulation has started.')
            A = self.set_variables(variables, std_dev, num_samples)
            self.run_simulation(A, f"A_pos_{std_dev_pos}_rot_{std_dev_rot}")
            send_email('A simulation', 'A simulation was successfully completed.')
        except Exception as e:
            tb = traceback.format_exc()
            send_email('Error', f'Error occurred in A simulation\n{e}\nTraceback:\n{tb}')
        
        try:
            send_email('B simulation', 'B simulation has started.')
            B = self.set_variables(variables, std_dev, num_samples)
            self.run_simulation(B, f"B_pos_{std_dev_pos}_rot_{std_dev_rot}")
            send_email('B simulation', 'B simulation was successfully completed.')
        except Exception as e:
            send_email('Error', f'Error occurred in B simulation\n{e}')

        """
        samples = []
        for i in range(len(A)):
            samples.append(A[i][1])
        plt.hist(samples, bins=100)
        plt.show()
        
        """

        for changes in ['pos', 'rot', 'opt']:
            try:
                send_email('AB simulation', 'AB simulation has started.')   
                AB = self.create_Ci(A, B, changes)
                self.run_simulation(AB, f"AB_pos_{std_dev_pos}_rot_{std_dev_rot}_{today}_{changes}")
                send_email('AB simulation', f"AB simulation {std_dev_pos}_rot_{std_dev_rot}_{today}_{changes} was successfully completed.")

                send_email('BA simulation', 'BA simulation has started.')   
                BA = self.create_Ci(B, A, changes)
                self.run_simulation(BA, f"BA_pos_{std_dev_pos}_rot_{std_dev_rot}_{today}_{changes}")
                send_email('BA simulation', f"BA simulation {std_dev_pos}_rot_{std_dev_rot}_{today}_{changes} was successfully completed.")
            except Exception as e: 
                send_email('Error', f'Error occurred in AB simulation\n{e}')
            


