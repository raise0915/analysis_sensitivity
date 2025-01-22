import numpy as np
import pandas as pd
from numpy.random import default_rng

from analysis_sensitivity import SobolAnalysis
from send_mail import send_email

global today
today = ''

class SobolAnalysis_1227(SobolAnalysis):
    def set_variables(self, variables:dict, std_dev:dict, num_samples):
        """Generates samples for variables with given standard deviations."""
        samples = []
        covariance_matrix = np.diag(
            [std_dev['pos']**2, std_dev['pos']**2, std_dev['pos']**2]
        )  # 共分散行列（対角行列）
        variables_pos = np.concatenate(variables['pos'])
        rng = default_rng()
        samples_pos = rng.multivariate_normal(variables_pos, covariance_matrix, num_samples)

        rng = default_rng()
        def cartesian_to_spherical(x, y, z):
            r = np.sqrt(x**2 + y**2 + z**2)
            mtheta = np.arccos(z / r)
            phi = np.arctan2(y, x)
            return r, phi, mtheta

        r, _phi, _ = cartesian_to_spherical(float(variables['rot'][0]), float(variables['rot'][1]), float(variables['rot'][2]))
        theta = rng.normal(loc=_phi, scale=np.radians(std_dev['rot']), size=num_samples) 
        phi = rng.uniform(0, 2 * np.pi, num_samples)
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        samples_rot = np.array([x, y, z]).T


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
                    df.to_excel(f"{self.input_name}_{today}_{label}.xlsx", index=False, header=True)
                else:
                    with pd.ExcelWriter(
                        f"{self.input_name}_{today}_{label}.xlsx",
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
    # setting params
    num_samples = 1000 # サンプル数
    pos_x = 248
    pos_y = 416
    pos_z = 384

    mua_normal = 0.37
    mua_tumour = 0.27
    mus_normal = 27
    mus_tumour = 20
    opt_std_devs = np.array([0.14, 10, 0.05, 4.6])

    variables = {
        'pos': [[pos_x, pos_y, pos_z]],
        'rot': [15, -10, 95],
        'opt': [mua_normal, mus_normal, mua_tumour, mus_tumour]
    }

    std_dev = {
        'opt': opt_std_devs*0.5 /3
    }

    ## 腫瘍
    # 3sigma = 5 mm
    # w = 0.05 # 30%/2
    # optical properties - 平均値から50%の範囲で
    for sigma_pos in [1]:
        # sigma_pos = 1 sigma_rot = 15 のoptはまだ - 0122
        for sigma_rot in [15]:
            std_dev['pos']  = sigma_pos /3
            std_dev['rot'] = sigma_rot /3
            sim = SobolAnalysis_1227()
            sim.sobol_analysis(variables, std_dev, num_samples)
            send_email('Success', f'Position analysis sigma_pos:{sigma_pos} sigma_rot:{sigma_rot} is done')
    send_email('Success', 'All analysis is done')
