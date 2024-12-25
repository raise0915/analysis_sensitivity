
import os
import json
import subprocess
import icecream as ic

class Runmcx():
    def __init__(self, move_type) -> None:
        self.save = 1
        json_open = open("/home/mbpl/morizane/analysis_sensitivity/src/inputs/settings.json", "r")
        settings = json.load(json_open)
        self.MCX_PATH = settings.get('MCX_PATH')
        self.radius = settings.get('raidus')
        self.length = settings.get('length')
        self.input_name = settings.get('INPUT_PATH')
        self.cfg = json.load(open("/home/mbpl/morizane/analysis_sensitivity/src/inputs/input.json"))
        self.HOME_PATH = settings.get('HOME_PATH')
        self.OUTPUT_PATH = settings.get('OUTPUT_PATH')
        self.model_path = self.cfg['Domain']['VolumeFile']
        self.type = move_type
   
    def set_conditions(self, params): # , rotation, opt_tumour, opt_normal):
        output_path = os.path.join(self.MCX_PATH, "test.json")
        self.cfg['Session']['ID'] =  self.OUTPUT_PATH
        self.cfg['Optode']['Source']['Type'] = 'gaussian'
        self.cfg['Optode']['Source']['Pos'] = params['position']
        self.cfg['Optode']['Source']['Param1'] = [self.radius, params['rotation'][0], params['rotation'][1], params['rotation'][2]] # gaussianの回転方向はここで指定する
        self.cfg['Optode']['Source']['Param2'] = [self.length, 0, 0, 0]
        self.cfg['Optode']['Source']['Lambda'] = 664
        self.cfg['Domain']['Media'] = [
        {
            "mua": 0,
            "mus": 0,
            "g": 1,
            "n": 1
        },
        {
        "mua": params['mua_normal'],
        "mus": params['mus_normal'],
        "g": 0.9,
        "n": 1.38
        },
        {
            "mua": 1e-5,
            "mus": 0.1,
            "g": 1.0,
            "n": 1.0
        },
        {
        "mua": params['mua_tumour'],
        "mus": params['mus_tumour'],
            "g": 0.9,
            "n": 1.38
          }
        ]

        ic.ic(self.cfg)
        
        # save for run
        json_file1 = open(output_path, mode="w")
        json.dump(self.cfg, json_file1, indent=4)
        json_file1.close()

    def run_mcx(self, params): # , rotation, opt_tumour, opt_normal):
        self.set_conditions(params) # , rotation, opt_tumour, opt_normal)
        os.chdir(self.MCX_PATH) 
        subprocess.run("./bin/mcx -f test.json", shell=True)

