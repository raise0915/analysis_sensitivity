import numpy as np
import matplotlib.pyplot as plt
from run_mcx import Runmcx
import pandas as pd
from calc_dvh import calc_log_dvh_with_log
from scipy.interpolate import interp1d
from cylinder import create_rotated_cylinder_mask
import icecream as ic
from numpy.random import default_rng
import copy

class SobolAnalysis(Runmcx):
    def set_variables(self, variables, std_dev, num_samples):
        """
        Generates samples for variables with given standard deviations.
        """
        std_dev_pos = std_dev["position"]
        std_dev_rot = std_dev["rotation"]


        # position
        covariance_matrix = np.diag([std_dev_pos**2, std_dev_pos**2, std_dev_pos**2])  # 共分散行列（対角行列）
        pos_vals = np.concatenate([variables[0]])
        rng = default_rng()
        samples_pos = rng.multivariate_normal(pos_vals, covariance_matrix, num_samples)

        # direction
        rng = default_rng()
        def cartesian_to_spherical(x, y, z):
            r = np.sqrt(x**2 + y**2 + z**2)
            mtheta = np.arccos(z / r)
            phi = np.arctan2(y, x)
            return r, phi, mtheta

        r, _phi, _ = cartesian_to_spherical(15, -10, 95)
        theta = rng.normal(loc=_phi, scale=np.radians(std_dev_rot), size=num_samples) 
        phi = rng.uniform(0, 2 * np.pi, num_samples)
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        samples = {
            "position": samples_pos,
            "rotation": np.array([x, y, z]).T
        }

        return samples

    def run_simulation(self, vals, label, num_samples):
        """
        Runs simulations for given values.
        """
        result = []
        for i in range(num_samples):
            for key in vals:
                if key == "position":
                    cylinder_pos = vals[key][i]
                elif key == "rotation":
                    cylinder_rot = vals[key][i]
            # 光学特性値変更用
            # opt_tumour = [val[6], val[7], val[8], val[9]]
            # opt_normal = [val[10], val[11], val[12], val[13]]

            self.run_mcx(cylinder_pos, cylinder_rot) #, cylinder_rot, opt_tumour, opt_normal)
            self.create_model(cylinder_pos, cylinder_rot)
            res_dict = self.evaluate_model()
            df = pd.DataFrame({
                'pos_x': [cylinder_pos[0]],
                'pos_y': [cylinder_pos[1]],
                'pos_z': [cylinder_pos[2]],
                'rot_x': [cylinder_rot[0]],
                'rot_y': [cylinder_rot[1]],
                'rot_z': [cylinder_rot[2]],
                'D90_normal': res_dict['dvh']['normal']['d90'],
                'D50_normal': res_dict['dvh']['normal']['d50'],
                'D10_normal': res_dict['dvh']['normal']['d10'],
                'D90_tumour': res_dict['dvh']['tumour']['d90'],
                'D50_tumour': res_dict['dvh']['tumour']['d50'],
                'D10_tumour': res_dict['dvh']['tumour']['d10'],
                'cover_rate_100': [res_dict['cover_rate_100']],
                'cover_rate_10': [res_dict['cover_rate_10']],
                'cover_rate_1': [res_dict['cover_rate_1']]
            })
            result.append([res_dict['dvh']['normal']['d90'], res_dict['dvh']['normal']['d50'], res_dict['dvh']['normal']['d10'],
                              res_dict['dvh']['tumour']['d90'], res_dict['dvh']['tumour']['d50'], res_dict['dvh']['tumour']['d10'],
                              res_dict['cover_rate_100'], res_dict['cover_rate_10'], res_dict['cover_rate_1']])
            if i == 0:
                df.to_excel(f'{self.input_name}_{label}.xlsx', index=False, header=True)
            else:
                with pd.ExcelWriter(f'{self.input_name}_{label}.xlsx', mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                    df.to_excel(writer, sheet_name='Sheet1', index=False, header=False, startrow=i+1)

        return result

    def create_model(self, cylinder_pos, cylinder_rot):
        with open(self.model_path, 'rb') as f:
            self.model = np.fromfile(f, dtype=np.uint8).reshape((600,600,600), order='F')
        cylinder_mask = create_rotated_cylinder_mask([600,600,600], cylinder_pos, self.radius, self.length, cylinder_rot)
        self.model[cylinder_mask==1] = 4

    def evaluate_model(self):
        """
        Evaluates the model based on the simulation results.
        """
        with open(f'{self.OUTPUT_PATH}.mc2', 'rb') as f:
            res = np.fromfile(f, dtype=np.float32).reshape((600,600,600), order='F')  
            energy = 150*(10**-3)*667*100
            res *= energy

        # DVH (log)
        cum_rel_dvh_t, _, cum_rel_dvh_l, bin_edges = calc_log_dvh_with_log(res, self.model)
        
        # tumour tissue
        Dx = interp1d(cum_rel_dvh_t, bin_edges[:-1])
        try:
            d90_t = Dx([90])
        except Exception:
            d90_t = np.nan
        try:
            d50_t = Dx([50])
        except Exception:
            d50_t = np.nan
        try:
            d10_t = Dx([10])
        except Exception:
            d10_t = np.nan
            
        # normal tissue
        Dx = interp1d(cum_rel_dvh_l, bin_edges[:-1])
        try:
            d90_n = Dx([90])
        except Exception:
            d90_n = np.nan
        try:
            d50_n = Dx([50])
        except Exception:
            d50_n = np.nan
        try:
            d10_n = Dx([10])
        except Exception:
            d10_n = np.nan         

        # cover_rate
        cover_rate_100 = np.count_nonzero((self.model == 3) & (res >= 100)) / np.count_nonzero(self.model == 3) * 100
        cover_rate_10 = np.count_nonzero((self.model == 3) & (res >= 10)) / np.count_nonzero(self.model == 3) * 100
        cover_rate_1 = np.count_nonzero((self.model == 3) & (res >= 1)) / np.count_nonzero(self.model == 3) * 100

        return {
            "dvh":{
            "tumour":{
                "d90": d90_t,
                "d50": d50_t,
                "d10": d10_t,
            },
            "normal":{
                "d90": d90_n,
                "d50": d50_n,
                "d10": d10_n                
            }
            },
            "cover_rate_100": cover_rate_100,
            "cover_rate_10": cover_rate_10,
            "cover_rate_1": cover_rate_1
        }


    def sobol_analysis(self, variables:list, std_devs:dict, num_samples:int):
        """
        Sobol sensitivity analysis
        """
        A = self.set_variables(variables, std_devs, num_samples)
        B = self.set_variables(variables, std_devs, num_samples) 

        samples =  []
        for i in range(num_samples):
            samples.append(A['rotation'][i][0])
        plt.hist(samples, bins=100)
        plt.show()
        
        Y_A = self.run_simulation(A, f"A_{std_devs}", num_samples)
        Y_B = self.run_simulation(B, f"B_{std_devs}", num_samples)

        V_Y = np.zeros(9)
        for i in range(9):
            V_Y[i] =  np.var([row[i] for row in Y_A], axis=0)

        ic.ic(V_Y)

        sobol_first = {i: {} for i in range(9)}
        sobol_first_err = {i: {} for i in range(9)}

        for j in range(9):
            Y_A_column = np.array([row[j] for row in Y_A])
            sobol_first[j] = np.mean(Y_A_column) / V_Y[j]
            sobol_first_err[j] = np.std(Y_A_column) / V_Y[j]

        for index in ['position', 'rotation']:
            A_Bi = copy.deepcopy(A)
            for j in range(num_samples):
                A_Bi[index][j] = B[index][j]
            Y_A_Bi = self.run_simulation(A_Bi, f'change{index}_{std_devs}', num_samples)

            for j in range(9):
                Y_A_Bi_column = np.array([row[j] for row in Y_A_Bi])
                Y_A_column = np.array([row[j] for row in Y_A])
                Y_B_column = np.array([row[j] for row in Y_B])
                sobol_first[j] = np.mean(Y_A_column * (Y_A_Bi_column - Y_B_column)) / V_Y[j]
                sobol_first_err[j] = np.std(Y_A_column * (Y_A_Bi_column - Y_B_column)) / V_Y[j]

            ic.ic(sobol_first)                
        return sobol_first, sobol_first_err    

if __name__ == '__main__':
    # setting params
    num_samples = 5  # サンプル数
    pos_x = 248
    pos_y =  416
    pos_z = 384
    
    dir_x = 15
    dir_y = -10
    dir_z = 95

    ## 腫瘍
    # 3sigma = 5 mm 
    # w = 0.05 # 30%/2
    for sigma_pos in [5, 10]:
        for sigma_rot in [5, 10]:
            std_dev_pos = round(sigma_pos / 3, 2)
            std_dev_rot = round(sigma_rot / 3, 2) # degree
            std_devs = {"position": std_dev_pos, "rotation": std_dev_rot}  
            ic.ic(std_devs)
                    
            variables = [
                [pos_x, pos_y, pos_z], # default_position
                [dir_x, dir_y, dir_z], # default_direction
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
            sobol_first, sobol_first_err = sim.sobol_analysis(variables, std_devs, num_samples)

            # DataFrameに変換
            df_sobol_first = pd.DataFrame(sobol_first)
            df_sobol_first_err = pd.DataFrame(sobol_first_err)

            # エクセルに保存
            with pd.ExcelWriter(f'sobol_analysis_results_pos{sigma_pos}_rot{sigma_rot}.xlsx') as writer:
                df_sobol_first.to_excel(writer, sheet_name='Sobol First')
                df_sobol_first_err.to_excel(writer, sheet_name='Sobol First Error')

    # プロット
    """
    vals_flatten = np.array(variables).flatten()
    labels = [f"x{i+1}" for i in range(len(vals_flatten))]
    x = np.arange(len(variables))
    plt.bar(x, sobol_first[0], yerr=sobol_first_err, capsize=5, color='steelblue')
    plt.xticks(x, labels)
    plt.ylabel("ST")
    plt.title("Sobol Sensitivity Analysis")
    plt.show()
    """

