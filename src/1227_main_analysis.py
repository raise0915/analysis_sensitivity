import numpy as np
import pandas as pd
from numpy.random import default_rng
import json

from analysis_sensitivity import SobolAnalysis
from send_mail import send_email
from rotate_fiber import rotate_fiber


class SobolAnalysis_1227(SobolAnalysis):
    def set_variables(self, variables:dict, std_dev:dict, num_samples):
        """Generates samples for variables with given standard deviations."""
        samples = []
        covariance_matrix = np.diag(
            [std_dev['pos']**2, std_dev['pos']**2, std_dev['pos']**2]
        )  # 共分散行列（対角行列）
        variables_pos = variables['pos']
        rng = default_rng() 
        samples_pos = rng.multivariate_normal(variables_pos, covariance_matrix, num_samples)

        samples_rot = rotate_fiber(variables_pos, self.length, variables['pos'], std_dev['rot'], num_samples)


        variables_opt = np.array(variables['opt'])
        std_devs = np.array(std_dev['opt'])
        samples = []
        for val, std in zip(variables_opt, std_devs):
            sample = np.random.normal(val, std, num_samples)
            samples.append(sample)
        samples_opt = np.array(samples).T

        return {
            'pos': samples_pos,
            'rot': samples_rot,
            'opt': samples_opt
        }

    def run_simulation(self, vals, label):
        """Runs simulations for given values."""
        for i in range(num_samples):
                cylinder_pos = vals['pos'][i]
                cylinder_rot = vals['rot'][i]
                params = {
                    'position': cylinder_pos,
                    'rotation': cylinder_rot,
                    'mua_normal': vals['opt'][i][0],
                    'mus_normal': vals['opt'][i][1],
                    'mua_tumour': vals['opt'][i][2],
                    'mus_tumour': vals['opt'][i][3] 
                }
                try:
                    self.run_mcx(params)  # , cylinder_rot, opt_tumour, opt_normal)
                    self.create_model(cylinder_pos, cylinder_rot)
                    res_dict = self.evaluate_model()
                except Exception as e:
                    send_email('Error', f'Error occurred in {label} analysis\n{e}')
                    continue  
                res_dict.update({
                    'pos_x': params['position'][0],
                    'pos_y': params['position'][1],
                    'pos_z': params['position'][2],
                    'rot_x': params['rotation'][0],
                    'rot_y': params['rotation'][1],
                    'rot_z': params['rotation'][2],
                    'mua_normal': params['mua_normal'],
                    'mus_normal': params['mus_normal'],
                    'mua_tumour': params['mua_tumour'],
                    'mus_tumour': params['mus_tumour']
                })

                df = pd.DataFrame(res_dict, index=[0])

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
                    

if __name__ == "__main__":
    with open('/home/mbpl/morizane/analysis_sensitivity/src/inputs/default_value.json') as f:
        default_value = json.load(f)
    print(default_value)

    # setting params
    num_samples = 1000 # サンプル数
    mua_normal = 0.37
    mua_tumour = 0.27
    mus_normal = 27
    mus_tumour = 20
    opt_std_devs = np.array([0.14, 10, 0.05, 4.6])

    variables = {
        'pos': np.array(default_value['pos_cut']),
        'rot': default_value['rot_cut'],
        'opt': [mua_normal, mus_normal, mua_tumour, mus_tumour]
    }

    std_dev = {
        'opt': opt_std_devs*0.5 /3
    }

    ## 腫瘍
    # 3sigma = 5 mm
    # w = 0.05 # 30%/2
    # optical properties - 平均値から50%の範囲で
    for sigma_pos in [2, 4, 6, 8]:
        # sigma_pos = 1 sigma_rot = 15 のoptはまだ - 0122
        for sigma_rot in [2, 4, 6]:
            std_dev['pos']  = sigma_pos /3
            std_dev['rot'] = sigma_rot /3
            sim = SobolAnalysis_1227()
            sim.sobol_analysis(variables, std_dev, num_samples)
            send_email('Success', f'Position analysis sigma_pos:{sigma_pos} sigma_rot:{sigma_rot} is done')
    send_email('Success', 'All analysis is done')

