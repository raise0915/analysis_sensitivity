import numpy as np
from numpy.random import default_rng
from analysis_sensitivity import SobolAnalysis
import pandas as pd
from run_mcx import Runmcx
from send_mail import send_email

from SALib.sample import saltelli
from SALib.analyze import sobol



class DirAnalysis(SobolAnalysis, Runmcx):
    def set_variables(self, variables, std_dev:int, num_samples):
        """
        Generates samples for variables with given standard deviations.
        """
        # direction
        rng = default_rng()
        def cartesian_to_spherical(x, y, z):
            r = np.sqrt(x**2 + y**2 + z**2)
            mtheta = np.arccos(z / r)
            phi = np.arctan2(y, x)
            return r, phi, mtheta

        r, _phi, _ = cartesian_to_spherical(variables[0], variables[1], variables[2])
        theta = rng.normal(loc=_phi, scale=np.radians(std_dev), size=num_samples) 
        phi = rng.uniform(0, 2 * np.pi, num_samples)
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        return np.array([x, y, z]).T


class OptAnalysis(SobolAnalysis, Runmcx):
    def set_variables(self, variables, std_devs:list, num_samples):
        variables = np.array(variables)
        std_devs = np.array(std_devs)
        samples = []
        for val, std in zip(variables, std_devs):
            sample = np.random.normal(val, std, num_samples)
            samples.append(sample)
        samples = np.array(samples).T
        return samples

def export_res_to_excel(sobol_first, sobol_first_err, label, sigma):
    # DataFrameに変換
    df_sobol_first = pd.DataFrame(sobol_first)
    df_sobol_first_err = pd.DataFrame(sobol_first_err)
                # エクセルに保存
    with pd.ExcelWriter(f"/home/mbpl/morizane/analysis_sensitivity/sobol_analysis_results_{label}_{sigma}.xlsx") as writer:
        df_sobol_first.to_excel(writer, sheet_name="Sobol First")
        df_sobol_first_err.to_excel(writer, sheet_name="Sobol First Error")

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
    opt_std_devs = [0.14, 10, 0.05, 4.6]

    ## 腫瘍
    # 3sigma = 5 mm
    # w = 0.05 # 30%/2
    # optical properties - 平均値から50%の範囲で
    for label in ['optical_properties', 'direction', 'position']:
        try:
            if label == 'optical_properties':
                sim = OptAnalysis(label)
                sobol_first, sobol_first_err = sim.sobol_analysis(
                        [mua_normal, mus_normal, mua_tumour, mus_tumour], 
                        np.array([mua_normal, mus_normal, mua_tumour, mus_tumour])*0.5 // 3,
                        num_samples
                    )
                export_res_to_excel(sobol_first, sobol_first_err, label, '')
                send_email('Success', 'Optical properties analysis is done')
            
            if label == 'position':
                for sigma in [5, 10]:
                    sim = SobolAnalysis(label)
                    sobol_first, sobol_first_err = sim.sobol_analysis(
                            [[pos_x, pos_y, pos_z]], round(sigma/3, 2), num_samples
                        )
                    export_res_to_excel(sobol_first, sobol_first_err, label, sigma)
                    send_email('Success', f'Position analysis val{sigma} is done')
            
            if label =='direction':
                for sigma in [5, 10]:
                    sim = DirAnalysis(label)
                    sobol_first, sobol_first_err = sim.sobol_analysis(
                            [15, -10, 95], round(sigma/3, 2), num_samples
                        )
                    export_res_to_excel(sobol_first, sobol_first_err, label, sigma)
                    send_email('Success', f'Direction analysis val{sigma} is done')
        except Exception as e:
            send_email('Error', 
                       f"""
                       Error occurred in {label} analysis
                       {e}
                       """)
