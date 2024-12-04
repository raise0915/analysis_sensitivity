import numpy as np
import matplotlib.pyplot as plt
from run_mcx import Runmcx
import pandas as pd
from calc_dvh import calc_log_dvh_with_log
from scipy.interpolate import interp1d
from cylinder import create_rotated_cylinder_mask
from openpyxl import load_workbook
import icecream as ic

class SobolAnalysis(Runmcx):
    def set_variables(self, variables, std_devs, num_samples, max_vals):
        """
        Generates samples for variables with given standard deviations.
        """
        samples = []
        variables = np.concatenate(variables)
        max_vals = np.concatenate(max_vals)
        for _ in range(num_samples):
            # pos_x, pos_y, pos_z, rot_x, rot_y, rot_z
            sample = [min(np.random.normal(var, std), max_val) for var, std, max_val in zip(variables, std_devs, max_vals)]
            samples.append(sample)
        return samples

    def run_simulation(self, vals, label):
        """
        Runs simulations for given values.
        """
        result = []
        for i, val in enumerate(vals):
            cylinder_pos = [val[0], val[1], val[2]]
            cylinder_rot= [15, -10, 95]
            # cylinder_rot = [val[3], val[4], val[5]]
            # opt_tumour = [val[6], val[7], val[8], val[9]]
            # opt_normal = [val[10], val[11], val[12], val[13]]

            self.run_mcx(cylinder_pos) #, cylinder_rot, opt_tumour, opt_normal)
            self.create_model(cylinder_pos, cylinder_rot)
            res_dict = self.evaluate_model()
            df = pd.DataFrame({
                'pos_x': [cylinder_pos[0]],
                'pos_y': [cylinder_pos[1]],
                'pos_z': [cylinder_pos[2]],
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
                          res_dict['dvh']['tumour']['d90'], res_dict['dvh']['tumour']['d50'], res_dict['dvh']['tumour']['d90'],
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


    def sobol_analysis(self, variables:list, std_devs:list, max_vals, num_samples:int):
        """
        Sobol sensitivity analysis
        """
        import copy
        
        A = pd.read_excel("/home/raise/mcx_simulation/analysis_sensitivity/input_A.xlsx").iloc[:, 0:3].values.tolist()
        # self.set_variables(variables, std_devs, num_samples, max_vals)
        samples =  []
        for i in range(len(A)):
            samples.append(A[i][0])
        plt.hist(samples, bins=100)
        plt.show()
        B = pd.read_excel("/home/raise/mcx_simulation/analysis_sensitivity/input_B.xlsx").iloc[:, 0:3].values.tolist()
        # self.set_variables(variables, std_devs, num_samples, max_vals)
        
        # Y_A = self.run_simulation(A, "A")
        # Y_B = self.run_simulation(B, "B")
        Y_A = pd.read_excel("/home/raise/mcx_simulation/analysis_sensitivity/input_A.xlsx").iloc[:, 3:].values.tolist()
        Y_B = pd.read_excel("/home/raise/mcx_simulation/analysis_sensitivity/input_B.xlsx").iloc[:, 3:].values.tolist()

        V_Y =  (np.var(Y_A, axis=0))
        ic.ic(V_Y)
        # V_Y = np.var(Y_A, ddof=1)

        sobol_first = {i: {} for i in range(3)}
        sobol_first_err = {i: {} for i in range(3)}

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
                
        return sobol_first, sobol_first_err    


# setting params
num_samples = 1000  # サンプル数
w = 0.05 # 30%/2
pos_x = 248
pos_y =  416
pos_z = 384
        
variables = [
    [pos_x, pos_y, pos_z] # position
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

# std_devs = np.concatenate(variables)

## 腫瘍
# 3sigma = 5 mm 
 
std_devs = np.array([1.67, 1.67, 1.67])
print(std_devs)

# run
sim = SobolAnalysis()
sobol_first, sobol_first_err = sim.sobol_analysis(variables, std_devs, max_vals, num_samples)

# DataFrameに変換
df_sobol_first = pd.DataFrame(sobol_first)
df_sobol_first_err = pd.DataFrame(sobol_first_err)

# エクセルに保存
with pd.ExcelWriter('sobol_analysis_results.xlsx') as writer:
    df_sobol_first.to_excel(writer, sheet_name='Sobol First')
    df_sobol_first_err.to_excel(writer, sheet_name='Sobol First Error')

# プロット
vals_flatten = np.array(variables).flatten()
labels = [f"x{i+1}" for i in range(len(vals_flatten))]
x = np.arange(len(variables))
plt.bar(x, sobol_first[0], yerr=sobol_first_err, capsize=5, color='steelblue')
plt.xticks(x, labels)
plt.ylabel("ST")
plt.title("Sobol Sensitivity Analysis")
plt.show()

